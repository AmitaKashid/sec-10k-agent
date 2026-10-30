"""Unit tests for app.api.token_budget. Uses a fake monotonic clock so
the 60s eviction is deterministic."""

from __future__ import annotations

import pytest

from app.api.auth import Identity
from app.api.token_budget import WINDOW_SECONDS, TokenBudget


class FakeClock:
    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def anon() -> Identity:
    return Identity(tier="anon", key="1.2.3.4")


@pytest.fixture
def user() -> Identity:
    return Identity(tier="user", key="user_abc")


@pytest.fixture
def fake_settings(monkeypatch):
    from app.settings import settings

    monkeypatch.setattr(settings, "ASK_TOKEN_LIMITER_ENABLED", True)
    monkeypatch.setattr(settings, "ASK_ANON_TOKENS_PER_MIN", 1000)
    monkeypatch.setattr(settings, "ASK_USER_TOKENS_PER_MIN", 10_000)
    return settings


async def test_reserve_under_budget_succeeds(fake_settings, anon):
    budget = TokenBudget(time_fn=FakeClock())
    res, used, cap = await budget.check_and_reserve(anon, est_tokens=300)
    assert res is not None
    assert used == 300
    assert cap == 1000


async def test_reserve_over_budget_rejects(fake_settings, anon):
    budget = TokenBudget(time_fn=FakeClock())
    await budget.check_and_reserve(anon, est_tokens=900)
    res, used, cap = await budget.check_and_reserve(anon, est_tokens=200)
    assert res is None
    # `used` returned on reject reflects state BEFORE the rejected reservation.
    assert used == 900
    assert cap == 1000


async def test_sliding_window_evicts_after_60s(fake_settings, anon):
    clock = FakeClock()
    budget = TokenBudget(time_fn=clock)
    await budget.check_and_reserve(anon, est_tokens=800)
    clock.advance(WINDOW_SECONDS + 1)
    # Old entry should be evicted, full budget available again.
    res, used, _ = await budget.check_and_reserve(anon, est_tokens=900)
    assert res is not None
    assert used == 900


async def test_finalize_replaces_estimate_with_actual(fake_settings, anon):
    budget = TokenBudget(time_fn=FakeClock())
    # Reserve 800; actual usage was 200. The 600 over-estimate should be
    # released so the next request can use it.
    res, _, _ = await budget.check_and_reserve(anon, est_tokens=800)
    assert res is not None
    await budget.finalize(res, actual_tokens=200)
    used = await budget.used(anon)
    assert used == 200
    # The 800 reservation is gone — a 700-token request should now fit.
    res2, used2, _ = await budget.check_and_reserve(anon, est_tokens=700)
    assert res2 is not None
    assert used2 == 900


async def test_finalize_after_eviction_records_fresh_entry(fake_settings, anon):
    clock = FakeClock()
    budget = TokenBudget(time_fn=clock)
    res, _, _ = await budget.check_and_reserve(anon, est_tokens=500)
    assert res is not None
    clock.advance(WINDOW_SECONDS + 1)
    # Reservation entry has been evicted; finalize must still record the
    # real cost so it counts against the current window.
    await budget.finalize(res, actual_tokens=300)
    assert await budget.used(anon) == 300


async def test_tiers_do_not_share_budget(fake_settings, anon, user):
    budget = TokenBudget(time_fn=FakeClock())
    await budget.check_and_reserve(anon, est_tokens=900)  # near anon cap
    # User tier has its own bucket — must succeed with a big reservation.
    res, used, cap = await budget.check_and_reserve(user, est_tokens=5000)
    assert res is not None
    assert used == 5000
    assert cap == 10_000


async def test_disabled_passthrough(monkeypatch, anon):
    from app.settings import settings

    monkeypatch.setattr(settings, "ASK_TOKEN_LIMITER_ENABLED", False)
    monkeypatch.setattr(settings, "ASK_ANON_TOKENS_PER_MIN", 1)  # absurdly low
    budget = TokenBudget(time_fn=FakeClock())
    res, used, cap = await budget.check_and_reserve(anon, est_tokens=999_999)
    assert res is not None
    assert res.enabled is False
    # finalize is a no-op for disabled reservations.
    await budget.finalize(res, actual_tokens=999_999)
    # used/cap returned as 0 in disabled mode — caller doesn't need them.
    assert (used, cap) == (0, 0)


async def test_signed_out_and_signed_in_same_principal_do_not_bleed(fake_settings):
    """A user who was signed in but then signs out (or vice versa) must
    not inherit the prior bucket's usage."""
    budget = TokenBudget(time_fn=FakeClock())
    same_principal_signed_in = Identity(tier="user", key="alice")
    same_principal_signed_out = Identity(tier="anon", key="alice")
    await budget.check_and_reserve(same_principal_signed_in, est_tokens=9000)
    res, used, cap = await budget.check_and_reserve(same_principal_signed_out, est_tokens=900)
    assert res is not None
    assert used == 900  # anon bucket starts fresh
    assert cap == 1000
