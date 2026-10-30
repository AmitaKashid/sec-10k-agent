"""Plan node — LLM mocked, asserts state-merge correctness."""

from __future__ import annotations

import pytest

from app.agent.nodes import plan as plan_module
from app.agent.state import AgentState, PlanResult


class _FakeStructured:
    def __init__(self, response, include_raw: bool = False):
        self._response = response
        self._include_raw = include_raw

    async def ainvoke(self, prompt):
        if self._include_raw:
            return {"raw": None, "parsed": self._response, "parsing_error": None}
        return self._response


class _FakeChat:
    def __init__(self, response):
        self._response = response

    def with_structured_output(self, schema, *, include_raw: bool = False, **_):
        return _FakeStructured(self._response, include_raw=include_raw)


def _patch_llm(monkeypatch, response: PlanResult):
    monkeypatch.setattr(plan_module, "get_chat_model", lambda *a, **kw: _FakeChat(response))


@pytest.mark.asyncio
async def test_plan_classifies_a_numerical(monkeypatch):
    _patch_llm(
        monkeypatch,
        PlanResult(
            question_type="A_numerical",
            ticker="aapl",
            fiscal_year=2024,
            concept_key="revenue",
        ),
    )
    state = AgentState(question="What was Apple's revenue in 2024?", request_id="r")
    out = await plan_module.plan(state)
    assert out["question_type"] == "A_numerical"
    assert out["ticker"] == "AAPL"  # uppercased
    assert out["fiscal_year"] == 2024
    assert out["concept_key"] == "revenue"
    assert out["refusal_reason"] is None


@pytest.mark.asyncio
async def test_plan_classifies_b_qualitative(monkeypatch):
    _patch_llm(
        monkeypatch,
        PlanResult(
            question_type="B_qualitative",
            ticker="AAPL",
            item="1A",
        ),
    )
    state = AgentState(question="What does Apple say about supply chain risk?", request_id="r")
    out = await plan_module.plan(state)
    assert out["question_type"] == "B_qualitative"
    assert out["item"] == "1A"


@pytest.mark.asyncio
async def test_plan_refuses_off_topic(monkeypatch):
    _patch_llm(
        monkeypatch,
        PlanResult(
            question_type="REFUSE",
            refusal_reason="I only answer SEC 10-K questions.",
        ),
    )
    state = AgentState(question="what's the weather?", request_id="r")
    out = await plan_module.plan(state)
    assert out["question_type"] == "REFUSE"
    assert "10-K" in out["refusal_reason"]


@pytest.mark.asyncio
async def test_plan_starts_fresh_trace_events(monkeypatch):
    """plan is the entry node and the per-turn boundary: it resets
    trace_events (part of RESET_DEFAULTS) so a prior turn's SSE events
    never bleed into this turn's stream when the checkpointer carries
    state across turns. See conversation_memory_plan.md §7."""
    _patch_llm(monkeypatch, PlanResult(question_type="A_numerical", ticker="AAPL"))
    state = AgentState(question="q", request_id="r", trace_events=[{"node": "previous"}])
    out = await plan_module.plan(state)
    assert len(out["trace_events"]) == 1
    assert out["trace_events"][0]["node"] == "plan"


@pytest.mark.asyncio
async def test_plan_handles_null_ticker(monkeypatch):
    """REFUSE with no ticker shouldn't crash on .upper()."""
    _patch_llm(monkeypatch, PlanResult(question_type="REFUSE", refusal_reason="off-topic"))
    state = AgentState(question="hi", request_id="r")
    out = await plan_module.plan(state)
    assert out["ticker"] is None
