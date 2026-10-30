"""uuid6_cutoff() ordering invariant — the load-bearing assumption that
makes the prune SQL `checkpoint_id < cutoff_uuid` correct.

If langgraph ever changes the checkpoint_id generator (away from
UUIDv6) or the column type (away from TEXT), one of these tests should
fail — see the docstring in `scripts/prune_checkpoints.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

# This is the canonical langgraph UUIDv6 generator — the same function
# `langgraph-checkpoint-postgres` calls when writing `checkpoint_id`.
# Pinning the import path also asserts the upstream layout we depend on.
from langgraph.checkpoint.base.id import uuid6

from scripts.prune_checkpoints import _UUID6_EPOCH, uuid6_cutoff

# --- shape ---------------------------------------------------------------


def test_cutoff_has_uuid_string_shape():
    s = uuid6_cutoff(datetime(2025, 1, 1, tzinfo=UTC))
    # 8-4-4-4-12 hex layout
    assert len(s) == 36
    parts = s.split("-")
    assert [len(p) for p in parts] == [8, 4, 4, 4, 12]
    # Version nibble is 6 (UUIDv6) — third group, first char.
    assert parts[2][0] == "6"


def test_cutoff_at_epoch_is_lowest_uuid6():
    # The minimum UUIDv6 has timestamp zero (== epoch).
    assert uuid6_cutoff(_UUID6_EPOCH).startswith("00000000-0000-6000-")


# --- ordering: the core invariant ---------------------------------------


def test_earlier_dt_yields_lexicographically_smaller_cutoff():
    """T1 < T2 ⇒ uuid6_cutoff(T1) < uuid6_cutoff(T2). If this fails the
    SQL `WHERE checkpoint_id < cutoff` won't select the older rows."""
    t1 = datetime(2025, 1, 1, tzinfo=UTC)
    t2 = datetime(2026, 1, 1, tzinfo=UTC)
    assert uuid6_cutoff(t1) < uuid6_cutoff(t2)


def test_strict_monotonicity_across_one_minute_step():
    base = datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC)
    prev = uuid6_cutoff(base)
    for i in range(1, 10):
        cur = uuid6_cutoff(base + timedelta(minutes=i))
        assert prev < cur, f"non-monotonic at step {i}: {prev} >= {cur}"
        prev = cur


# --- ordering vs REAL langgraph uuid6: closes the assumption -----------


def test_real_uuid6_at_time_t_is_greater_than_cutoff_at_t():
    """A checkpoint_id written by langgraph at time T must compare
    strictly greater than uuid6_cutoff(T - epsilon) — otherwise the
    prune query would delete current rows.

    Sampling the live `uuid6()` ties this test to the actual generator
    used by langgraph-checkpoint, so a future upstream change that
    breaks the invariant fails here loudly.
    """
    # Anchor "now" so the cutoff is well in the past of the uuid6() we
    # generate next — avoids 100ns-resolution flakiness around the boundary.
    cutoff_dt = datetime.now(UTC) - timedelta(seconds=1)
    cutoff = uuid6_cutoff(cutoff_dt)

    # The actual langgraph generator. Take a few samples; every one must
    # compare > cutoff.
    for _ in range(5):
        sample = str(uuid6())
        assert sample > cutoff, f"uuid6() {sample!r} is not > cutoff {cutoff!r}"


def test_real_uuid6_at_time_t_is_less_than_future_cutoff():
    """The mirror invariant — a checkpoint_id written now must compare
    less than a cutoff computed for a clearly-future time. This is what
    makes the prune *not* delete future rows (if any clock skew exists)."""
    sample = str(uuid6())
    future_cutoff = uuid6_cutoff(datetime.now(UTC) + timedelta(days=365 * 100))
    assert sample < future_cutoff
