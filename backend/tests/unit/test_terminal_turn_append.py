"""Each terminal node must append exactly one Turn to `history` so the
next turn's `plan` can resolve follow-ups. Tests the REAL nodes (not
stubs) — closes the gap in test_graph_followup_grounding.py which stubs
plan/retrieve. See conversation_memory_plan.md §6, §13.
"""

from __future__ import annotations

import pytest

from app.agent.nodes import concierge as concierge_module
from app.agent.nodes import refusal_composer as composer_module
from app.agent.state import AgentState


class _FakeMsg:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChat:
    def __init__(self, response: str) -> None:
        self._response = response

    async def ainvoke(self, prompt: str) -> _FakeMsg:
        return _FakeMsg(self._response)


@pytest.mark.asyncio
async def test_refusal_composer_appends_turn_with_refusal_code(monkeypatch):
    monkeypatch.setattr(
        composer_module, "get_chat_model", lambda *_a, **_kw: _FakeChat("Sorry, can't.")
    )
    state = AgentState(
        question="compare AAPL and MSFT",
        request_id="r",
        question_type="REFUSE",
        refusal_code="CROSS_COMPANY",
        refusal_reason="one company at a time",
    )
    out = await composer_module.refusal_composer(state)
    assert "history" in out
    assert len(out["history"]) == 1
    turn = out["history"][0]
    # The refused turn carries its code so the next plan can apply rule 3.
    assert turn.refusal_code == "CROSS_COMPANY"
    assert turn.question == "compare AAPL and MSFT"
    assert turn.final_answer == "Sorry, can't."


@pytest.mark.asyncio
async def test_concierge_appends_turn(monkeypatch):
    monkeypatch.setattr(
        concierge_module, "get_chat_model", lambda *_a, **_kw: _FakeChat("Hi there!")
    )
    state = AgentState(question="hello", request_id="r", question_type="META")
    out = await concierge_module.concierge(state)
    assert "history" in out
    assert len(out["history"]) == 1
    assert out["history"][0].final_answer == "Hi there!"
    # META turns carry no ticker — nothing to inherit, by design.
    assert out["history"][0].ticker is None


@pytest.mark.asyncio
async def test_refusal_composer_appends_turn_even_on_llm_failure(monkeypatch):
    """Refusal must never depend on a successful LLM call — and the Turn
    must still be appended so history stays consistent."""

    class _Raise:
        async def ainvoke(self, prompt):
            raise RuntimeError("LLM down")

    monkeypatch.setattr(composer_module, "get_chat_model", lambda *_a, **_kw: _Raise())
    state = AgentState(
        question="weather?",
        request_id="r",
        question_type="REFUSE",
        refusal_code="OFF_TOPIC",
        refusal_reason="I only do 10-K questions",
    )
    out = await composer_module.refusal_composer(state)
    assert len(out["history"]) == 1
    assert out["history"][0].refusal_code == "OFF_TOPIC"
    # Fell back to the structured reason rather than silence.
    assert out["history"][0].final_answer == "I only do 10-K questions"
