"""The `history` reducer must CONCATENATE Turn lists across turns of a
checkpointed thread, not overwrite. See conversation_memory_plan.md §6.

Pure unit test: a minimal StateGraph over AgentState with an
InMemorySaver. No DB, no LLM.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from app.agent.state import AgentState, Turn


def _build_graph(checkpointer):
    async def terminal(state: AgentState) -> dict:
        # Each run appends exactly one Turn for the current question.
        return {"history": [Turn(question=state.question, ticker="AAPL")]}

    g = StateGraph(AgentState)
    g.add_node("terminal", terminal)
    g.set_entry_point("terminal")
    g.add_edge("terminal", END)
    return g.compile(checkpointer=checkpointer)


@pytest.mark.asyncio
async def test_history_concatenates_across_turns_same_thread():
    graph = _build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "T1"}}

    out1 = await graph.ainvoke(AgentState(question="q1", request_id="r1"), config=config)
    assert [t.question for t in out1["history"]] == ["q1"]

    out2 = await graph.ainvoke(AgentState(question="q2", request_id="r2"), config=config)
    # Reducer concatenated — q1 survives into turn 2.
    assert [t.question for t in out2["history"]] == ["q1", "q2"]

    out3 = await graph.ainvoke(AgentState(question="q3", request_id="r3"), config=config)
    assert [t.question for t in out3["history"]] == ["q1", "q2", "q3"]


@pytest.mark.asyncio
async def test_history_isolated_per_thread():
    graph = _build_graph(InMemorySaver())

    await graph.ainvoke(
        AgentState(question="a1", request_id="r"),
        config={"configurable": {"thread_id": "A"}},
    )
    out_b = await graph.ainvoke(
        AgentState(question="b1", request_id="r"),
        config={"configurable": {"thread_id": "B"}},
    )
    # Thread B must not see thread A's turn.
    assert [t.question for t in out_b["history"]] == ["b1"]


@pytest.mark.asyncio
async def test_no_checkpointer_history_never_accumulates():
    """Without a checkpointer every invoke is a fresh thread — history
    is whatever the single run produced, never carried."""
    graph = _build_graph(None)
    out1 = await graph.ainvoke(AgentState(question="q1", request_id="r1"))
    out2 = await graph.ainvoke(AgentState(question="q2", request_id="r2"))
    assert [t.question for t in out1["history"]] == ["q1"]
    assert [t.question for t in out2["history"]] == ["q2"]


@pytest.mark.asyncio
async def test_thread_config_is_harmless_on_no_checkpointer_graph():
    """The route always passes config={'configurable': {'thread_id': ...}}
    when thread_id is truthy, regardless of whether the compiled graph
    has a checkpointer. A no-checkpointer graph must accept that config
    without error and simply not persist — verifies the route is safe
    when LANGGRAPH_CHECKPOINTER_ENABLED=false but a client still sends
    thread_id."""
    graph = _build_graph(None)
    config = {"configurable": {"thread_id": "client-sent-this"}}
    out1 = await graph.ainvoke(AgentState(question="q1", request_id="r1"), config=config)
    out2 = await graph.ainvoke(AgentState(question="q2", request_id="r2"), config=config)
    # No error, and no cross-invocation persistence (no checkpointer).
    assert [t.question for t in out1["history"]] == ["q1"]
    assert [t.question for t in out2["history"]] == ["q2"]
