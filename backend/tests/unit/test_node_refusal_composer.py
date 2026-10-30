"""Refusal composer node — short-circuit, happy path, fallback."""

from __future__ import annotations

import pytest

from app.agent.nodes import refusal_composer as composer_module
from app.agent.state import AgentState


class _FakeMsg:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChat:
    """LLM stub: records the prompt it received and returns a canned reply."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.last_prompt: str | None = None

    async def ainvoke(self, prompt: str) -> _FakeMsg:
        self.last_prompt = prompt
        return _FakeMsg(self._response)


class _RaisingChat:
    async def ainvoke(self, prompt: str) -> _FakeMsg:
        raise RuntimeError("simulated LLM outage")


def _patch_chat(monkeypatch, chat) -> None:
    monkeypatch.setattr(composer_module, "get_chat_model", lambda *_a, **_kw: chat)


@pytest.mark.asyncio
async def test_off_topic_uses_llm(monkeypatch) -> None:
    """OFF_TOPIC must flow through the composer LLM, not a static string —
    the user wants every refusal to feel agentic. Cost is bounded by the
    upstream rate limits, not by skipping the LLM here."""
    fake = _FakeChat(
        "I focus on SEC 10-K filings — try asking what Apple disclosed about "
        "supply chain risk in its 10-K."
    )
    _patch_chat(monkeypatch, fake)

    state = AgentState(
        question="what's the weather?",
        request_id="r1",
        question_type="REFUSE",
        refusal_code="OFF_TOPIC",
        refusal_reason="Off-topic.",
    )
    out = await composer_module.refusal_composer(state)
    # LLM was invoked and its output reached the user.
    assert fake.last_prompt is not None
    assert "OFF_TOPIC" in fake.last_prompt
    assert "Apple" in out["refusal_message"]
    # Trace event records the code; no short_circuit flag any more.
    assert any(
        e.get("node") == "refusal_composer" and e.get("refusal_code") == "OFF_TOPIC"
        for e in out["trace_events"]
    )


@pytest.mark.asyncio
async def test_missing_filing_uses_available_years(monkeypatch) -> None:
    """MISSING_FILING with available years should pass them to the prompt."""
    fake = _FakeChat("We don't have AAPL's 2019 10-K. Try fiscal year 2024 — we have that one.")
    _patch_chat(monkeypatch, fake)

    state = AgentState(
        question="What was AAPL revenue in 2019?",
        request_id="r2",
        question_type="REFUSE",
        ticker="AAPL",
        fiscal_year=2019,
        refusal_code="MISSING_FILING",
        refusal_reason="No 10-K for AAPL FY2019.",
        available_fiscal_years=[2024, 2023, 2022],
    )
    out = await composer_module.refusal_composer(state)
    assert "2024" in out["refusal_message"]
    # The prompt must carry the available years so the LLM can ground its
    # suggestion. Verify they appear in the rendered prompt.
    assert fake.last_prompt is not None
    assert "2024" in fake.last_prompt
    assert "2023" in fake.last_prompt


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_refusal_reason(monkeypatch) -> None:
    """Refusal must never depend on a successful LLM call."""
    _patch_chat(monkeypatch, _RaisingChat())

    state = AgentState(
        question="What was XYZ revenue?",
        request_id="r3",
        question_type="REFUSE",
        ticker="XYZ",
        refusal_code="UNKNOWN_TICKER",
        refusal_reason="XYZ isn't in our 50-company universe.",
    )
    out = await composer_module.refusal_composer(state)
    # Falls back to the deterministic short hint.
    assert "XYZ" in out["refusal_message"]
    assert out["refusal_message"].startswith("XYZ isn't")


@pytest.mark.asyncio
async def test_no_op_when_not_refusing(monkeypatch) -> None:
    """Composer routed in error on a non-refuse state should no-op."""
    monkeypatch.setattr(
        composer_module,
        "get_chat_model",
        lambda *_a, **_kw: pytest.fail("LLM should not be called"),
    )
    state = AgentState(
        question="What was AAPL revenue in 2024?",
        request_id="r4",
        question_type="A_numerical",
    )
    out = await composer_module.refusal_composer(state)
    assert out == {}


@pytest.mark.asyncio
async def test_happy_path_writes_message_and_trace(monkeypatch) -> None:
    fake = _FakeChat("I can't compare two companies in one query — try them separately.")
    _patch_chat(monkeypatch, fake)

    state = AgentState(
        question="Compare AAPL and MSFT revenue",
        request_id="r5",
        question_type="REFUSE",
        refusal_code="CROSS_COMPANY",
        refusal_reason="Cross-company.",
    )
    out = await composer_module.refusal_composer(state)
    assert "compare" in out["refusal_message"].lower()
    assert any(
        e.get("node") == "refusal_composer" and e.get("refusal_code") == "CROSS_COMPANY"
        for e in out["trace_events"]
    )
