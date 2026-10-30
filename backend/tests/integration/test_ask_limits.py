"""Integration tests for tiered limits on /ask.

Covers the 413 (input too large) and 429 (request-count and token-budget)
paths. The 429 tests deliberately set the relevant tier limit to 0 so
the very first request trips — keeps the test fast and avoids touching
the graph.
"""

from __future__ import annotations

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import Identity, resolve_identity
from app.api.token_budget import BUDGET, REQUESTS, WINDOW_SECONDS
from app.main import app
from app.settings import settings


@pytest.fixture(autouse=True)
def _clear_budgets(monkeypatch, offline_graph_services):
    """Module-level budget singletons persist across tests in the same
    process. Clear before each test for determinism."""
    monkeypatch.setattr(settings, "ASK_TOKEN_LIMITER_ENABLED", True)
    monkeypatch.setattr(settings, "ANON_LIFETIME_ENABLED", False)
    monkeypatch.setattr(settings, "ASK_ANON_REQS_PER_MIN", 10)
    monkeypatch.setattr(settings, "ASK_USER_REQS_PER_MIN", 10)
    monkeypatch.setattr(settings, "ASK_ANON_TOKENS_PER_MIN", 100000)
    monkeypatch.setattr(settings, "ASK_USER_TOKENS_PER_MIN", 100000)
    BUDGET._buckets.clear()
    REQUESTS._buckets.clear()
    yield
    BUDGET._buckets.clear()
    REQUESTS._buckets.clear()


async def _post_ask(headers: dict[str, str] | None = None, question: str = "What?"):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            "/ask",
            json={"question": question},
            headers={"Accept": "application/json", **(headers or {})},
        )


# --- 413 input too large -------------------------------------------------


async def test_ask_413_when_question_exceeds_anon_max(monkeypatch, agent_session: AsyncSession):
    monkeypatch.setattr(settings, "ASK_ANON_MAX_INPUT_TOKENS", 0)
    r = await _post_ask()
    assert r.status_code == 413
    body = r.json()
    assert body["reason"] == "input_too_large"
    assert body["auth_required"] is True
    assert body["limit"]["window_s"] == WINDOW_SECONDS


# --- 429 request-count ---------------------------------------------------


async def test_ask_429_anonymous_req_limit_shape(monkeypatch, agent_session: AsyncSession):
    monkeypatch.setattr(settings, "ASK_ANON_REQS_PER_MIN", 0)
    r = await _post_ask()
    assert r.status_code == 429
    assert r.headers.get("Retry-After") == str(WINDOW_SECONDS)
    body = r.json()
    assert body["reason"] == "anonymous_req_limit"
    assert body["auth_required"] is True
    assert body["limit"]["budget"] == 0


async def test_ask_429_authed_req_limit_shape(monkeypatch, agent_session: AsyncSession):
    # Override the resolve_identity dep to return a user tier without
    # needing a real Clerk JWT — narrower than monkeypatching the auth
    # module and matches FastAPI's idiomatic test pattern.
    app.dependency_overrides[resolve_identity] = lambda: Identity(tier="user", key="user_test")
    try:
        monkeypatch.setattr(settings, "ASK_USER_REQS_PER_MIN", 0)
        r = await _post_ask()
    finally:
        app.dependency_overrides.pop(resolve_identity, None)
    assert r.status_code == 429
    body = r.json()
    assert body["reason"] == "authed_req_limit"
    assert body["auth_required"] is False


# --- 429 token-budget ----------------------------------------------------


async def test_ask_429_anonymous_token_budget_shape(monkeypatch, agent_session: AsyncSession):
    # Zero token budget → any reservation > 0 trips immediately. The
    # request-count cap is left at default 5 so it doesn't fire first.
    monkeypatch.setattr(settings, "ASK_ANON_TOKENS_PER_MIN", 0)
    r = await _post_ask()
    assert r.status_code == 429
    body = r.json()
    assert body["reason"] == "anonymous_limit"
    assert body["auth_required"] is True


async def test_ask_429_authed_token_budget_shape(monkeypatch, agent_session: AsyncSession):
    app.dependency_overrides[resolve_identity] = lambda: Identity(tier="user", key="user_test")
    try:
        monkeypatch.setattr(settings, "ASK_USER_TOKENS_PER_MIN", 0)
        r = await _post_ask()
    finally:
        app.dependency_overrides.pop(resolve_identity, None)
    assert r.status_code == 429
    body = r.json()
    assert body["reason"] == "authed_limit"
    assert body["auth_required"] is False


async def test_token_budget_rejection_refunds_request_slot(
    monkeypatch, agent_session: AsyncSession
):
    # When the token-budget check fails, the request-count slot consumed
    # in the preceding step must be refunded — otherwise the two caps
    # stop reconciling independently (plan §Token budget).
    app.dependency_overrides[resolve_identity] = lambda: Identity(tier="user", key="user_test")
    try:
        monkeypatch.setattr(settings, "ASK_USER_TOKENS_PER_MIN", 0)
        r = await _post_ask()
        assert r.status_code == 429
        assert r.json()["reason"] == "authed_limit"
        # The request-bucket must be empty — the rejected request did
        # not get to use the service, so it should not have burned a slot.
        dq = REQUESTS._buckets.get("user:user_test")
        assert not dq, f"expected empty request bucket, got {list(dq) if dq else None}"
    finally:
        app.dependency_overrides.pop(resolve_identity, None)
