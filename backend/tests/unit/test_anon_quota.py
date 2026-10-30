"""Unit tests for app.api.anon_quota.check_and_increment.

Session is mocked; these tests do not validate the SQL against PostgreSQL.
These tests cover the branching:
disabled flag, user tier, allowed/blocked, fail-open on DB error.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api import anon_quota
from app.api.auth import Identity


@pytest.fixture(autouse=True)
def enable_limiter(monkeypatch):
    from app.settings import settings

    monkeypatch.setattr(settings, "ANON_LIFETIME_ENABLED", True)
    monkeypatch.setattr(settings, "ANON_LIFETIME_LIMIT", 10)
    monkeypatch.setattr(settings, "ANON_HASH_SALT", "test-salt-1234")


def _mock_session(returning_count: int | None) -> MagicMock:
    """Build an AsyncSession double whose execute() returns a row with
    `returning_count`, or no row if None (= cap reached / UPDATE clamp)."""
    session = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    if returning_count is None:
        # First execute (INSERT) returns no row. Second execute (SELECT
        # for current count) returns the capped value.
        first = MagicMock()
        first.first.return_value = None
        second = MagicMock()
        second.first.return_value = (10,)
        session.execute = AsyncMock(side_effect=[first, second])
    else:
        result = MagicMock()
        result.first.return_value = (returning_count,)
        session.execute = AsyncMock(return_value=result)
    return session


async def test_user_tier_is_noop():
    identity = Identity(tier="user", key="user_abc")
    session = MagicMock()
    session.execute = AsyncMock()

    allowed, used, limit = await anon_quota.check_and_increment(identity, session)

    assert (allowed, used, limit) == (True, 0, 0)
    session.execute.assert_not_called()


async def test_disabled_is_noop(monkeypatch):
    from app.settings import settings

    monkeypatch.setattr(settings, "ANON_LIFETIME_ENABLED", False)
    identity = Identity(tier="anon", key="1.2.3.4")
    session = MagicMock()
    session.execute = AsyncMock()

    allowed, used, limit = await anon_quota.check_and_increment(identity, session)

    assert (allowed, used, limit) == (True, 0, 0)
    session.execute.assert_not_called()


async def test_first_request_allowed_count_one():
    identity = Identity(tier="anon", key="1.2.3.4")
    session = _mock_session(returning_count=1)

    allowed, used, limit = await anon_quota.check_and_increment(identity, session)

    assert allowed is True
    assert used == 1
    assert limit == 10
    session.commit.assert_awaited_once()


async def test_tenth_request_still_allowed():
    identity = Identity(tier="anon", key="1.2.3.4")
    session = _mock_session(returning_count=10)

    allowed, used, _ = await anon_quota.check_and_increment(identity, session)

    assert allowed is True
    assert used == 10


async def test_over_limit_blocked_and_not_incremented():
    """When the row is already at the cap, the WHERE clamp prevents the
    UPDATE so RETURNING produces no row. We then read the current count
    and return it as 'used' so the response body is accurate."""
    identity = Identity(tier="anon", key="1.2.3.4")
    session = _mock_session(returning_count=None)

    allowed, used, limit = await anon_quota.check_and_increment(identity, session)

    assert allowed is False
    assert used == 10
    assert limit == 10
    session.commit.assert_awaited_once()


async def test_db_error_fails_open():
    identity = Identity(tier="anon", key="1.2.3.4")
    session = MagicMock()
    session.execute = AsyncMock(side_effect=RuntimeError("conn dropped"))
    session.rollback = AsyncMock()
    session.commit = AsyncMock()

    allowed, used, limit = await anon_quota.check_and_increment(identity, session)

    assert allowed is True  # fail open
    assert used == 0
    assert limit == 10
    session.rollback.assert_awaited_once()


def test_hash_ip_is_deterministic_and_salted(monkeypatch):
    from app.settings import settings

    monkeypatch.setattr(settings, "ANON_HASH_SALT", "salt-a")
    a1 = anon_quota.hash_ip("9.9.9.9")
    a2 = anon_quota.hash_ip("9.9.9.9")
    assert a1 == a2
    assert len(a1) == 64

    monkeypatch.setattr(settings, "ANON_HASH_SALT", "salt-b")
    b = anon_quota.hash_ip("9.9.9.9")
    assert a1 != b  # different salt → different hash


# --- find_disallowed_ticker ----------------------------------------------

_ALLOW = frozenset({"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"})


def test_prefilter_matches_explicit_blocked_ticker():
    assert anon_quota.find_disallowed_ticker("tell me about TSLA", _ALLOW) == "TSLA"


def test_prefilter_passes_allowlisted_ticker():
    assert anon_quota.find_disallowed_ticker("What is AAPL revenue?", _ALLOW) is None


def test_prefilter_ignores_lowercase():
    # Lowercase falls through to the in-graph anon_gate by design.
    assert anon_quota.find_disallowed_ticker("tell me about tsla", _ALLOW) is None


def test_prefilter_ignores_non_ticker_uppercase_tokens():
    # CEO, USD, GDP are uppercase but not in the universe → no false 403.
    assert anon_quota.find_disallowed_ticker("Who is the CEO and what USD?", _ALLOW) is None


def test_prefilter_returns_first_blocked_when_mixed():
    # JPM is a known non-allowlisted ticker; AAPL is allowlisted.
    out = anon_quota.find_disallowed_ticker("Compare AAPL and JPM revenue", _ALLOW)
    assert out == "JPM"
