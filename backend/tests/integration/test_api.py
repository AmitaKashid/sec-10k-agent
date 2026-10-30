"""HTTP integration tests for /ask, /companies, /health.

Real Postgres, mocked LLMs (same pattern as test_agent_graph). Uses the
`agent_session` fixture to seed AAPL FY2024 inline.
"""

from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.messages import AIMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.nodes import plan as plan_module
from app.agent.nodes import validate as validate_module
from app.agent.nodes.validate import _ClaimByIdx, _ValidateLLMResponse
from app.agent.state import PlanResult
from app.main import app
from app.settings import settings
from tests.integration.conftest import SKIP_REASON, db_reachable

pytestmark = pytest.mark.skipif(not db_reachable(), reason=SKIP_REASON)


class _StructuredFake:
    def __init__(self, response, include_raw: bool = False):
        self._response = response
        self._include_raw = include_raw

    async def ainvoke(self, prompt):
        if self._include_raw:
            return {"raw": None, "parsed": self._response, "parsing_error": None}
        return self._response


class _ChatFake:
    def __init__(self, response):
        self._response = response

    def bind(self, **kwargs):
        return self

    async def ainvoke(self, prompt):
        if isinstance(self._response, PlanResult):
            content = self._response.model_dump_json()
        else:
            content = json.dumps({"selected_ids": sorted({c.source_idx for c in self._response.claims})})
        return AIMessage(content=content, response_metadata={"finish_reason": "stop"})

    def with_structured_output(self, schema, *, include_raw: bool = False, **_):
        return _StructuredFake(self._response, include_raw=include_raw)


def _patch_plan(monkeypatch, plan_response: PlanResult):
    monkeypatch.setattr(plan_module, "get_chat_model", lambda *a, **kw: _ChatFake(plan_response))


def _patch_validate(monkeypatch, validate_response: _ValidateLLMResponse):
    monkeypatch.setattr(
        validate_module, "get_chat_model", lambda *a, **kw: _ChatFake(validate_response)
    )


def _aapl_revenue_plan() -> PlanResult:
    return PlanResult(
        question_type="A_numerical",
        ticker="AAPL",
        fiscal_year=2024,
        concept_key="revenue",
    )


def _aapl_revenue_validate() -> _ValidateLLMResponse:
    return _ValidateLLMResponse(
        draft_answer="Apple's FY2024 revenue was $391,035 million.",
        claims=[
            _ClaimByIdx(
                text="Apple's FY2024 revenue was $391,035 million.",
                kind="number",
                source_idx=0,
            )
        ],
    )


# --- /health -------------------------------------------------------------


async def test_health_ok(agent_session: AsyncSession):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["db"] == "ok"
    assert body["graph"] == "ok"


# --- /companies ----------------------------------------------------------


async def test_companies_requires_api_key(agent_session: AsyncSession):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/companies")
    assert r.status_code == 401


async def test_companies_returns_seeded_aapl(agent_session: AsyncSession):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/companies", headers={"X-API-Key": settings.API_KEY})
    assert r.status_code == 200
    rows = r.json()
    tickers = [c["ticker"] for c in rows]
    assert "AAPL" in tickers


# --- /ask ----------------------------------------------------------------


async def test_ask_rejects_invalid_bearer_token(agent_session: AsyncSession):
    # /ask now allows anonymous access (anon tier) per
    # docs/auth_and_rate_limit_plan.md. A *present* but invalid Bearer
    # token must still be rejected — silently downgrading would mask a
    # misconfigured frontend.
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/ask",
            json={"question": "what?"},
            headers={"Authorization": "Bearer not-a-real-jwt"},
        )
    assert r.status_code == 401


async def test_ask_sync_happy_path(monkeypatch, agent_session: AsyncSession):
    _patch_plan(monkeypatch, _aapl_revenue_plan())
    _patch_validate(monkeypatch, _aapl_revenue_validate())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/ask",
            json={"question": "What was Apple's FY2024 revenue?"},
            headers={"X-API-Key": settings.API_KEY, "Accept": "application/json"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["refused"] is False
    assert body["verified"] is True
    assert "391,035" in body["answer"]
    assert body["request_id"]


async def test_ask_sse_stream_terminates_with_done(monkeypatch, agent_session: AsyncSession):
    _patch_plan(monkeypatch, _aapl_revenue_plan())
    _patch_validate(monkeypatch, _aapl_revenue_validate())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as client:
        async with client.stream(
            "POST",
            "/ask",
            json={"question": "What was Apple's FY2024 revenue?"},
            headers={"X-API-Key": settings.API_KEY, "Accept": "text/event-stream"},
        ) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            raw = await _collect(r)

    # We emit one `event: status` per LangGraph node (loading-UI signal
    # only — no state on the wire) and a terminal `event: done` with the
    # verified answer. Frontend simulates typing on the final text.
    assert "event: status" in raw
    assert "event: done" in raw
    for step in ("plan", "fetch", "locate", "retrieve", "validate", "cite_check"):
        assert f'"step": "{step}"' in raw, f"missing status event: {step}"
    # Intermediate (unverified) state must not leak.
    assert "draft_answer" not in raw
    # Terminal payload markers.
    assert '"refused": false' in raw
    assert '"verified": true' in raw
    assert "391,035" in raw


async def test_ask_refusal_short_circuits(monkeypatch, agent_session: AsyncSession):
    _patch_plan(
        monkeypatch,
        PlanResult(question_type="REFUSE", refusal_reason="out of scope"),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/ask",
            json={"question": "what is the weather"},
            headers={"X-API-Key": settings.API_KEY, "Accept": "application/json"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] is True
    assert body["answer"] is None
    assert "out of scope" in (body["refusal_reason"] or "")


# --- helpers -------------------------------------------------------------


async def _collect(response: httpx.Response) -> str:
    chunks: list[str] = []
    async for raw in response.aiter_text():
        chunks.append(raw)
    return "".join(chunks)


def _parse_sse(payload: str) -> list[dict]:
    """Minimal SSE parser: split on blank-line boundaries, take event/data."""
    events: list[dict] = []
    for block in payload.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event = "message"
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith(":"):
                continue  # SSE comment / keepalive
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if not data_lines:
            continue
        try:
            data = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            data = {"raw": "\n".join(data_lines)}
        events.append({"event": event, "data": data})
    return events
