"""Headline grounding test for multi-turn memory.

The load-bearing guarantee (conversation_memory_plan.md §10): history
can change WHICH filing a turn queries, but can never itself become a
source of grounding. Concretely:

  * `Turn` carries no chunks / claims / XBRL facts — nothing citable.
  * `plan` applies RESET_DEFAULTS so turn N never inherits turn N-1's
    retrieved chunks; retrieval re-runs fresh on the rewritten question.

Built as a minimal checkpointed graph with stub nodes that exercise the
*real* RESET_DEFAULTS and build_turn_from_state machinery.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from app.agent.state import (
    RESET_DEFAULTS,
    AgentState,
    ChunkHit,
    Turn,
    build_turn_from_state,
)


def test_turn_model_carries_no_grounding_fields():
    """Static invariant: a Turn cannot smuggle citable content forward."""
    fields = set(Turn.model_fields)
    forbidden = {"chunks", "claims", "xbrl_fact", "source_ref", "evidence_quote"}
    assert fields.isdisjoint(forbidden), f"Turn leaks grounding: {fields & forbidden}"


def _filing_for_year(fy: int) -> str:
    return f"ACC-{fy}"


def _build_graph(checkpointer):
    """plan → retrieve → terminal. Stubs, but real reset + Turn helpers."""

    async def plan_stub(state: AgentState) -> dict:
        # Resolve a follow-up: "and 2024?" inherits AAPL from history and
        # bumps the year. Mirrors what the real plan LLM would produce.
        if state.history and "2024" in state.question:
            classification = {
                "question_type": "A_numerical",
                "ticker": "AAPL",
                "fiscal_year": 2024,
                "concept_key": "revenue",
                "is_followup": True,
                "rewritten_question": "What was AAPL revenue in 2024?",
            }
        else:
            classification = {
                "question_type": "A_numerical",
                "ticker": "AAPL",
                "fiscal_year": 2023,
                "concept_key": "revenue",
                "is_followup": False,
                "rewritten_question": None,
            }
        return {**RESET_DEFAULTS, **classification, "trace_events": []}

    async def retrieve_stub(state: AgentState) -> dict:
        assert state.chunks == []
        assert state.claims == []
        assert state.final_answer is None
        # Retrieval keys off the resolved fiscal_year — proving the
        # rewritten question (not turn 1's stale chunks) drives it.
        acc = _filing_for_year(state.fiscal_year)
        return {
            "chunks": [
                ChunkHit(
                    chunk_id=state.fiscal_year,
                    accession_no=acc,
                    item="7",
                    seq=0,
                    text=f"revenue discussion for FY{state.fiscal_year}",
                    score=0.05,
                    char_start=0,
                    char_end=10,
                )
            ]
        }

    async def terminal_stub(state: AgentState) -> dict:
        answer = f"AAPL FY{state.fiscal_year}: cited {state.chunks[0].accession_no}"
        return {
            "final_answer": answer,
            "verified": True,
            "history": [build_turn_from_state(state, answer)],
        }

    g = StateGraph(AgentState)
    g.add_node("plan", plan_stub)
    g.add_node("retrieve", retrieve_stub)
    g.add_node("terminal", terminal_stub)
    g.set_entry_point("plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "terminal")
    g.add_edge("terminal", END)
    return g.compile(checkpointer=checkpointer)


@pytest.mark.asyncio
async def test_followup_cites_its_own_turn_filing_not_the_prior():
    graph = _build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "T"}}

    # Turn 1: AAPL revenue 2023 → cites the 2023 filing.
    out1 = await graph.ainvoke(
        AgentState(question="AAPL revenue 2023?", request_id="r1"), config=config
    )
    assert out1["chunks"][0].accession_no == "ACC-2023"
    assert "ACC-2023" in out1["final_answer"]
    assert [t.fiscal_year for t in out1["history"]] == [2023]

    # Turn 2: "and 2024?" — resolves via history to AAPL FY2024.
    out2 = await graph.ainvoke(AgentState(question="and 2024?", request_id="r2"), config=config)

    # Grounding contract: turn 2 cites the 2024 filing ONLY. Turn 1's
    # 2023 chunk must not survive RESET_DEFAULTS into turn 2.
    assert len(out2["chunks"]) == 1
    assert out2["chunks"][0].accession_no == "ACC-2024"
    assert "ACC-2023" not in out2["final_answer"]
    assert "ACC-2024" in out2["final_answer"]

    # History accumulated both turns (reducer), in order.
    assert [t.fiscal_year for t in out2["history"]] == [2023, 2024]
    # And the follow-up was actually recognised as one.
    assert out2["is_followup"] is True
    assert out2["rewritten_question"] == "What was AAPL revenue in 2024?"


@pytest.mark.asyncio
async def test_reset_defaults_clears_stale_chunks_before_retrieve():
    """Even if a checkpointer hands plan a state still holding turn-1
    chunks, the plan delta must zero them so retrieve starts clean."""
    graph = _build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "T2"}}

    await graph.ainvoke(AgentState(question="AAPL revenue 2023?", request_id="r1"), config=config)
    out2 = await graph.ainvoke(AgentState(question="and 2024?", request_id="r2"), config=config)
    # No 2023 leakage anywhere in turn 2's chunk set.
    assert all(c.accession_no == "ACC-2024" for c in out2["chunks"])
