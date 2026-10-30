"""Route-level tests for the two anon /ask short-circuits that bypass
the graph: lifetime cap (429) and ticker prefilter (403).

Both paths return early via `limit_response`, so the test app doesn't
need a real DB or graph — `anon_check_and_increment` is monkeypatched
and `get_session` is a stub."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes
from app.api.auth import Identity
from app.data.db import get_session


@pytest.fixture
def anon_client(monkeypatch) -> TestClient:
    monkeypatch.setattr(routes.settings, "ANON_TICKER_ALLOWLIST", ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"])
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.resolve_identity] = lambda: Identity(tier="anon", key="1.2.3.4")

    async def _session():
        s = MagicMock()
        s.execute = AsyncMock()
        s.commit = AsyncMock()
        s.rollback = AsyncMock()
        yield s

    app.dependency_overrides[get_session] = _session
    return TestClient(app)


# --- Ticker prefilter ----------------------------------------------------


async def _allow_lifetime(_identity, _session):
    """Stub: lifetime check passes with one credit used out of ten."""
    return True, 1, 10


def test_ask_anon_blocked_ticker_returns_403(anon_client, monkeypatch):
    monkeypatch.setattr(routes, "anon_check_and_increment", _allow_lifetime)
    r = anon_client.post(
        "/ask",
        json={"question": "tell me about TSLA"},
        headers={"Accept": "application/json"},
    )
    assert r.status_code == 403
    body = r.json()
    assert body["reason"] == "anon_ticker_not_allowed"
    assert body["auth_required"] is True
    assert "TSLA" in body["detail"]


# Pass-through behavior (allowlisted ticker not blocked) is covered at the
# function level by test_prefilter_passes_allowlisted_ticker in
# test_anon_quota.py. A route-level test would need to mock the entire
# agent graph (or supply an OPENAI_API_KEY in CI) — overkill for what's
# essentially just `find_disallowed_ticker(q, allow) is None`.


# --- Lifetime cap --------------------------------------------------------


async def _block_lifetime(_identity, _session):
    """Stub: lifetime check trips at the cap."""
    return False, 10, 10


def test_ask_anon_lifetime_returns_429(anon_client, monkeypatch):
    monkeypatch.setattr(routes, "anon_check_and_increment", _block_lifetime)
    r = anon_client.post(
        "/ask",
        json={"question": "What is AAPL revenue?"},
        headers={"Accept": "application/json"},
    )
    assert r.status_code == 429
    body = r.json()
    assert body["reason"] == "anonymous_lifetime_limit"
    assert body["auth_required"] is True
    assert body["limit"]["used"] == 10
    assert body["limit"]["budget"] == 10
    # The lifetime check runs before the ticker prefilter, so a blocked
    # ticker on a capped account still surfaces as 429, not 403.


def test_ask_anon_lifetime_trips_before_ticker_check(anon_client, monkeypatch):
    monkeypatch.setattr(routes, "anon_check_and_increment", _block_lifetime)
    r = anon_client.post(
        "/ask",
        json={"question": "tell me about TSLA"},
        headers={"Accept": "application/json"},
    )
    assert r.status_code == 429
    assert r.json()["reason"] == "anonymous_lifetime_limit"
