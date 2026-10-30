"""Route-level test: /companies + /companies/stats reject anonymous callers
with 401 auth_required, and accept user-tier callers.

The DB session is a stub — we never reach the SQL path for the anon case
(handler short-circuits), and for the user case we just need to confirm
the dependency resolves and the handler runs to completion."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes
from app.api.auth import Identity
from app.data.db import get_session


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(routes.router)
    return app


@pytest.fixture
def anon_client(app: FastAPI) -> TestClient:
    app.dependency_overrides[routes.resolve_identity] = lambda: Identity(tier="anon", key="1.2.3.4")

    async def _session():
        s = MagicMock()
        s.execute = AsyncMock()
        s.commit = AsyncMock()
        s.rollback = AsyncMock()
        yield s

    app.dependency_overrides[get_session] = _session
    return TestClient(app)


def test_companies_anon_returns_401(anon_client: TestClient):
    r = anon_client.get("/companies")
    assert r.status_code == 401
    assert r.json() == {"detail": "auth_required"}


def test_companies_stats_anon_returns_401(anon_client: TestClient):
    r = anon_client.get("/companies/stats")
    assert r.status_code == 401
    assert r.json() == {"detail": "auth_required"}


@pytest.mark.parametrize("path", ["/companies", "/companies/stats"])
@pytest.mark.parametrize("access", ["user", "admin"])
def test_companies_accepts_signed_in_user_or_valid_admin_key(app, monkeypatch, path, access):
    from datetime import date, datetime, timezone
    monkeypatch.setattr(routes.settings, "API_KEY", "test-admin-key")
    identity = Identity(tier="user" if access == "user" else "anon", key="test-principal")
    app.dependency_overrides[routes.resolve_identity] = lambda: identity
    result = MagicMock()
    result.mappings.return_value.all.return_value = [{
        "ticker": "AAPL", "name": "Apple Inc.", "latest_10k_period": 2024,
        "latest_10k_filed_at": date(2024, 11, 1), "filings_count": 3,
        "has_chunks": True, "has_xbrl": True,
    }]
    result.mappings.return_value.one.return_value = {
        "issuers": 1, "filings_indexed": 3, "xbrl_facts": 10,
        "coverage_complete": 1, "last_refresh": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)
    async def fake_session():
        yield session
    app.dependency_overrides[get_session] = fake_session
    with TestClient(app) as client:
        headers = {"X-API-Key": "test-admin-key"} if access == "admin" else {}
        response = client.get(path, headers=headers)
    assert response.status_code == 200, response.text
    if path == "/companies":
        assert response.json()[0]["ticker"] == "AAPL"
        assert response.json()[0]["coverage_status"] == "complete"
    else:
        assert response.json()["issuers"] == 1
        assert response.json()["coverage_complete"] == 1
    session.execute.assert_awaited_once()


@pytest.mark.parametrize("path", ["/companies", "/companies/stats"])
def test_companies_rejects_invalid_admin_key(anon_client, monkeypatch, path):
    monkeypatch.setattr(routes.settings, "API_KEY", "test-admin-key")
    response = anon_client.get(path, headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401
