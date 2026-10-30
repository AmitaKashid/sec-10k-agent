"""Per-turn reset: `plan` must wipe working-memory carried by the
checkpointer from a prior turn, while NEVER touching `history`.

See conversation_memory_plan.md §7.
"""

from __future__ import annotations

import pytest

from app.agent.nodes import plan as plan_module
from app.agent.state import RESET_DEFAULTS, AgentState, ChunkHit, Claim, PlanResult, Turn


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


def test_reset_defaults_excludes_history_and_inputs():
    # history accumulates via reducer — resetting it would wipe the
    # conversation. question/request_id are set fresh by the caller.
    assert "history" not in RESET_DEFAULTS
    assert "question" not in RESET_DEFAULTS
    assert "request_id" not in RESET_DEFAULTS


def test_reset_defaults_covers_all_working_memory():
    # Every per-turn field that a prior turn could have dirtied must be
    # in RESET_DEFAULTS, else it leaks across turns.
    for field in (
        "chunks",
        "claims",
        "xbrl_fact",
        "final_answer",
        "verified",
        "cache_hit",
        "refusal_reason_final",
    ):
        assert field in RESET_DEFAULTS, f"{field} missing from RESET_DEFAULTS"


@pytest.mark.asyncio
async def test_plan_delta_resets_prior_turn_working_memory(monkeypatch):
    _patch_llm(
        monkeypatch,
        PlanResult(
            question_type="A_numerical",
            ticker="MSFT",
            fiscal_year=2024,
            concept_key="revenue",
        ),
    )
    # Simulate a state dirtied by a prior turn (checkpointer carried it).
    dirty = AgentState(
        question="MSFT revenue 2024?",
        request_id="r2",
        chunks=[
            ChunkHit(
                chunk_id=1,
                accession_no="OLD-ACC",
                item="7",
                seq=0,
                text="stale",
                score=0.1,
                char_start=0,
                char_end=5,
            )
        ],
        claims=[Claim(text="stale claim", kind="prose")],
        final_answer="stale answer from turn 1",
        verified=True,
        history=[Turn(question="AAPL revenue 2023?", ticker="AAPL")],
    )

    out = await plan_module.plan(dirty)

    # Working memory wiped...
    assert out["chunks"] == []
    assert out["claims"] == []
    assert out["final_answer"] is None
    assert out["verified"] is False
    # ...new classification applied...
    assert out["ticker"] == "MSFT"
    assert out["fiscal_year"] == 2024
    # ...and history is NOT in the plan delta (so the reducer keeps it).
    assert "history" not in out
    # trace_events starts fresh with just plan's event.
    assert len(out["trace_events"]) == 1
    assert out["trace_events"][0]["node"] == "plan"
