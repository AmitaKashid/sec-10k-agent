"""Follow-up resolution: PlanResult validator + plan-node propagation +
prompt construction. See conversation_memory_plan.md §8, §11.

The LLM's *decision* (rules 1–5) is an eval concern — here the LLM is
mocked. These tests pin the deterministic pieces:
  * the self-contained-rewrite validator,
  * propagation of is_followup / rewritten_question into the delta,
  * history block construction (refused turns filtered, window of 3).
"""

from __future__ import annotations

import pytest

from app.agent.nodes import plan as plan_module
from app.agent.prompts import build_plan_prompt
from app.agent.state import AgentState, PlanResult, Turn


class _FakeStructured:
    def __init__(self, response, sink, include_raw: bool = False):
        self._response = response
        self._sink = sink
        self._include_raw = include_raw

    async def ainvoke(self, prompt):
        self._sink["prompt"] = prompt
        if self._include_raw:
            return {"raw": None, "parsed": self._response, "parsing_error": None}
        return self._response


class _FakeChat:
    def __init__(self, response, sink):
        self._response = response
        self._sink = sink

    def with_structured_output(self, schema, *, include_raw: bool = False, **_):
        return _FakeStructured(self._response, self._sink, include_raw=include_raw)


def _patch_llm(monkeypatch, response: PlanResult):
    sink: dict = {}
    monkeypatch.setattr(plan_module, "get_chat_model", lambda *a, **kw: _FakeChat(response, sink))
    return sink


# --- PlanResult self-contained validator ---------------------------------


def test_followup_without_ticker_is_downgraded():
    r = PlanResult(
        question_type="A_numerical",
        is_followup=True,
        rewritten_question="revenue in 2024",  # no ticker
        ticker=None,
    )
    assert r.is_followup is False
    assert r.rewritten_question is None


def test_followup_without_rewrite_is_downgraded():
    r = PlanResult(
        question_type="A_numerical",
        is_followup=True,
        rewritten_question=None,
        ticker="AAPL",
    )
    assert r.is_followup is False


def test_valid_followup_survives():
    r = PlanResult(
        question_type="A_numerical",
        is_followup=True,
        rewritten_question="What was AAPL revenue in 2024?",
        ticker="AAPL",
        fiscal_year=2024,
        concept_key="revenue",
    )
    assert r.is_followup is True
    assert r.rewritten_question == "What was AAPL revenue in 2024?"


# --- plan-node propagation ----------------------------------------------


@pytest.mark.asyncio
async def test_plan_propagates_followup_fields(monkeypatch):
    _patch_llm(
        monkeypatch,
        PlanResult(
            question_type="A_numerical",
            ticker="AAPL",
            fiscal_year=2024,
            concept_key="revenue",
            is_followup=True,
            rewritten_question="What was AAPL revenue in 2024?",
        ),
    )
    state = AgentState(
        question="and 2024?",
        request_id="r",
        history=[Turn(question="AAPL revenue 2023?", ticker="AAPL", fiscal_year=2023)],
    )
    out = await plan_module.plan(state)
    assert out["is_followup"] is True
    assert out["rewritten_question"] == "What was AAPL revenue in 2024?"
    assert out["ticker"] == "AAPL"
    assert out["fiscal_year"] == 2024


@pytest.mark.asyncio
async def test_plan_standalone_when_no_history(monkeypatch):
    _patch_llm(
        monkeypatch,
        PlanResult(question_type="A_numerical", ticker="AAPL", fiscal_year=2023),
    )
    state = AgentState(question="AAPL revenue 2023?", request_id="r")
    out = await plan_module.plan(state)
    assert out["is_followup"] is False
    assert out["rewritten_question"] is None


# --- prompt construction ------------------------------------------------


def test_prompt_omits_history_block_when_empty():
    p = build_plan_prompt("AAPL revenue 2023?", [])
    assert "No prior turn history" in p
    assert "Prior turns in this conversation" not in p


def test_prompt_includes_recent_turns():
    hist = [
        Turn(question="AAPL revenue 2023?", ticker="AAPL", fiscal_year=2023, concept_key="revenue"),
    ]
    p = build_plan_prompt("and 2024?", hist)
    assert "Prior turns in this conversation" in p
    assert "ticker=AAPL" in p
    assert "fiscal_year=2023" in p


def test_prompt_filters_refused_turns():
    # Distinctive question string that does not appear in the static
    # PLAN_PROMPT (which itself contains "compare AAPL and MSFT").
    refused_q = "ZZQ_UNIQUE_REFUSED_MARKER_42"
    hist = [Turn(question=refused_q, ticker=None, refusal_code="CROSS_COMPANY")]
    p = build_plan_prompt("and just AAPL?", hist)
    # All prior turns were refused → treated as no usable history.
    assert "filtered out" in p
    # The refused turn's question must not be rendered as a history line.
    assert refused_q not in p
    assert "Prior turns in this conversation" not in p


def test_prompt_windows_to_last_three_turns():
    hist = [Turn(question=f"q{i}", ticker="AAPL", fiscal_year=2020 + i) for i in range(6)]
    p = build_plan_prompt("and the next year?", hist)
    # Only the last 3 (q3, q4, q5) should appear.
    assert "'q5'" in p and "'q4'" in p and "'q3'" in p
    assert "'q0'" not in p and "'q2'" not in p
