"""Prune LangGraph checkpoint rows for idle threads.

Deletes from `checkpoints`, `checkpoint_writes`, and `checkpoint_blobs`
all rows belonging to threads whose latest checkpoint is older than
``settings.CHECKPOINT_TTL_DAYS``. Batched + transactional per batch so a
mid-run interruption is safe.

The checkpoint tables carry NO timestamp column, but `checkpoint_id` is
a UUIDv6 — its high 60 bits encode 100ns intervals since the Gregorian
epoch (1582-10-15 UTC), big-endian. Lexicographic string comparison of
UUIDv6 values therefore equals time ordering, so we compute a cutoff
UUIDv6 from `now - TTL` and compare in SQL with a plain `<`.

LOAD-BEARING ASSUMPTION — verified against installed deps:
  * langgraph-checkpoint 4.1.0 generates `checkpoint_id` via the
    canonical UUIDv6 algorithm in
    `langgraph/checkpoint/base/id.py::uuid6()` (high 60 bits = 100ns
    intervals since 1582-10-15, big-endian).
  * langgraph-checkpoint-postgres 3.1.0 declares the column as
    `checkpoint_id TEXT NOT NULL` in
    `langgraph/checkpoint/postgres/base.py::MIGRATIONS`. TEXT ordering
    in Postgres is byte-wise, which for ASCII hex matches UUIDv6 time
    ordering. RE-VERIFY this docstring after any langgraph upgrade —
    if either the generator or the column type changes, the script may
    delete current rows or miss stale ones. The unit test in
    tests/unit/test_prune_checkpoints.py pins the ordering invariant.

Usage (from backend/):
    uv run python -m scripts.prune_checkpoints
    uv run python -m scripts.prune_checkpoints --dry-run
    uv run python -m scripts.prune_checkpoints --ttl-days 3
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta

import psycopg

from app.settings import settings

logger = logging.getLogger("prune_checkpoints")

BATCH_SIZE = 500
SLEEP_BETWEEN_BATCHES_S = 0.1

# 1582-10-15 00:00:00 UTC — UUIDv6 epoch.
_UUID6_EPOCH = datetime(1582, 10, 15, tzinfo=UTC)


def uuid6_cutoff(dt: datetime) -> str:
    """Return the smallest UUIDv6 string whose embedded timestamp == dt.

    Any real UUIDv6 generated at time T compares lexicographically
    greater-or-equal to this value when T >= dt, so
    ``checkpoint_id < uuid6_cutoff(now - ttl)`` selects rows older than
    the TTL.
    """
    delta = dt - _UUID6_EPOCH
    ns100 = int(delta.total_seconds() * 10_000_000)
    ts60 = ns100 & ((1 << 60) - 1)
    time_high = (ts60 >> 28) & 0xFFFFFFFF
    time_mid = (ts60 >> 12) & 0xFFFF
    time_low = ts60 & 0xFFF
    # version nibble = 6; clock_seq/node set to all-zero so this is the
    # minimum UUIDv6 at the given timestamp.
    return f"{time_high:08x}-{time_mid:04x}-6{time_low:03x}-8000-000000000000"


def _conn_string() -> str:
    from app.agent.memory import checkpoint_conn_string

    return checkpoint_conn_string(settings.DATABASE_URL)


async def _stale_thread_ids(conn: psycopg.AsyncConnection, cutoff: str, limit: int) -> list[str]:
    sql = """
        SELECT thread_id
        FROM checkpoints
        GROUP BY thread_id
        HAVING MAX(checkpoint_id) < %s
        LIMIT %s
    """
    async with conn.cursor() as cur:
        await cur.execute(sql, (cutoff, limit))
        return [r[0] for r in await cur.fetchall()]


async def _delete_threads(
    conn: psycopg.AsyncConnection, thread_ids: list[str], dry_run: bool
) -> dict[str, int]:
    deleted = {"checkpoint_writes": 0, "checkpoint_blobs": 0, "checkpoints": 0}
    if not thread_ids:
        return deleted
    if dry_run:
        logger.info("dry_run.skip_delete thread_count=%d", len(thread_ids))
        return deleted
    async with conn.cursor() as cur:
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            await cur.execute(f"DELETE FROM {table} WHERE thread_id = ANY(%s)", (thread_ids,))
            deleted[table] = cur.rowcount or 0
    await conn.commit()
    return deleted


async def prune(ttl_days: int, dry_run: bool) -> dict[str, int]:
    start = time.monotonic()
    cutoff_dt = datetime.now(UTC) - timedelta(days=ttl_days)
    cutoff_uuid = uuid6_cutoff(cutoff_dt)
    logger.info(
        "prune.start ttl_days=%d cutoff_dt=%s cutoff_uuid=%s dry_run=%s",
        ttl_days,
        cutoff_dt.isoformat(),
        cutoff_uuid,
        dry_run,
    )
    totals = {"checkpoint_writes": 0, "checkpoint_blobs": 0, "checkpoints": 0, "threads": 0}
    try:
        async with await psycopg.AsyncConnection.connect(_conn_string(), autocommit=False) as conn:
            while True:
                thread_ids = await _stale_thread_ids(conn, cutoff_uuid, BATCH_SIZE)
                if not thread_ids:
                    break
                deleted = await _delete_threads(conn, thread_ids, dry_run)
                totals["threads"] += len(thread_ids)
                for k, v in deleted.items():
                    totals[k] += v
                logger.info(
                    "prune.batch batch_threads=%d writes=%d blobs=%d checkpoints=%d",
                    len(thread_ids),
                    deleted["checkpoint_writes"],
                    deleted["checkpoint_blobs"],
                    deleted["checkpoints"],
                )
                if dry_run:
                    break
                await asyncio.sleep(SLEEP_BETWEEN_BATCHES_S)
    except psycopg.errors.UndefinedTable:
        # The LangGraph checkpoint tables haven't been created — this
        # happens when LANGGRAPH_CHECKPOINTER_ENABLED was never True
        # so `saver.setup()` never ran. Treat as a no-op (nothing to
        # prune) rather than a hard error, so the CLI / the startup
        # hook can be invoked safely in any state.
        logger.info("prune.skipped reason=checkpoint_tables_not_initialised")
        return totals
    logger.info(
        "prune.done elapsed_s=%.2f threads=%d writes=%d blobs=%d checkpoints=%d",
        time.monotonic() - start,
        totals["threads"],
        totals["checkpoint_writes"],
        totals["checkpoint_blobs"],
        totals["checkpoints"],
    )
    return totals


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttl-days", type=int, default=settings.CHECKPOINT_TTL_DAYS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(prune(ttl_days=args.ttl_days, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
