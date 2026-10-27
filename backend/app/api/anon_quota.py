"""Lifetime per-IP request cap for anonymous /ask callers.

Persisted in Postgres (anon_usage table) so it survives restarts and
the frontend's sign-in modal is the only way past it. Raw IPs are
never stored — ip_hash = sha256(ip || ANON_HASH_SALT).

See docs/anon_hardening_plan.md §5.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import re
from functools import lru_cache
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import Identity
from app.settings import settings

logger = logging.getLogger(__name__)

# Standalone uppercase tokens (e.g. "TSLA", "JPM"). Lowercase forms and
# company names spelled out ("tesla") are intentionally missed here —
# the in-graph anon_gate node is the safety net for those, and a regex
# match on lowercase would false-positive on every sentence.
_TICKER_TOKEN = re.compile(r"\b[A-Z]{1,5}\b")


@lru_cache(maxsize=1)
def _known_tickers() -> frozenset[str]:
    """The covered universe, loaded once from config/tickers.csv.

    Used so the prefilter only flags real tickers — bare uppercase
    tokens like 'CEO' or 'USD' aren't tickers and shouldn't trigger
    a 403."""
    # __file__ → backend/app/api/anon_quota.py; parents[2] → backend/.
    # The config dir lives inside backend/ so it ships in the Docker image.
    csv_path = Path(__file__).resolve().parents[2] / "config" / "tickers.csv"
    if not csv_path.exists():
        return frozenset()
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        return frozenset(
            (row.get("ticker") or "").strip().upper()
            for row in reader
            if (row.get("ticker") or "").strip()
        )


def find_disallowed_ticker(question: str, allowlist: frozenset[str]) -> str | None:
    """Return the first known-but-not-allowlisted ticker mentioned in
    `question`, or None. Best-effort prefilter: catches "tell me about TSLA"
    so /ask can return 403 + LimitModal, while company-name and lowercase
    forms fall through to the in-graph anon_gate."""
    known = _known_tickers()
    if not known:
        return None
    for token in _TICKER_TOKEN.findall(question):
        if token in known and token not in allowlist:
            return token
    return None


def hash_ip(ip: str) -> str:
    """sha256(ip || salt). Salt presence is enforced at settings load."""
    h = hashlib.sha256()
    h.update(ip.encode("utf-8"))
    h.update(settings.ANON_HASH_SALT.encode("utf-8"))
    return h.hexdigest()


async def check_and_increment(
    identity: Identity,
    session: AsyncSession,
) -> tuple[bool, int, int]:
    """Atomically bump the anon counter for this identity's IP.

    Returns (allowed, count_after, limit).
      - User tier or feature disabled → (True, 0, 0), no DB I/O.
      - Within budget → (True, new_count, limit), counter incremented.
      - At/over budget → (False, current_count, limit), counter NOT
        incremented (the WHERE clause clamps it; the row sticks at the
        cap and acts as a permanent block).
      - DB error → fail open: (True, 0, limit), logged loudly. The
        sliding-window per-minute caps still bound blast radius.
    """
    if not settings.ANON_LIFETIME_ENABLED or identity.tier != "anon":
        return True, 0, 0

    limit = settings.ANON_LIFETIME_LIMIT
    ip_hash = hash_ip(identity.key)

    # Single round-trip: insert-or-increment, returning the post-write
    # count. The WHERE on DO UPDATE clamps increments at `limit` so a
    # capped row stops climbing — important because we want the row to
    # behave as a stable block, not unbounded telemetry.
    stmt = text(
        """
        INSERT INTO anon_usage (ip_hash, count, first_seen, last_seen)
        VALUES (:ip_hash, 1, now(), now())
        ON CONFLICT (ip_hash) DO UPDATE
          SET count = anon_usage.count + 1,
              last_seen = now()
          WHERE anon_usage.count < :limit
        RETURNING count
        """
    )
    try:
        result = await session.execute(stmt, {"ip_hash": ip_hash, "limit": limit})
        row = result.first()
        if row is None:
            # The WHERE on DO UPDATE failed → row already at the cap.
            # Read the current count for the response body.
            row = (
                await session.execute(
                    text("SELECT count FROM anon_usage WHERE ip_hash = :ip_hash"),
                    {"ip_hash": ip_hash},
                )
            ).first()
            current = int(row[0]) if row else limit
            await session.commit()
            return False, current, limit
        count_after = int(row[0])
        await session.commit()
        return True, count_after, limit
    except Exception:
        logger.exception("anon_quota.db_error", extra={"ip_hash_prefix": ip_hash[:8]})
        try:
            await session.rollback()
        except Exception:
            pass
        # Fail open — sliding-window per-minute caps remain in force.
        return True, 0, limit


__all__ = ["check_and_increment", "find_disallowed_ticker", "hash_ip"]
