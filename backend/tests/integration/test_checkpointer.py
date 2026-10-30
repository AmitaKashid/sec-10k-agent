"""Real PostgreSQL checkpoint persistence and user isolation.

Uses only the isolated TEST_DATABASE_URL. CI requires that database to
be reachable before pytest; local runs skip when it is unavailable.
"""
from __future__ import annotations

import uuid

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph

from app.agent.memory import checkpoint_conn_string, checkpoint_key
from app.agent.state import CHECKPOINTED_TYPES, AgentState, Turn
from app.settings import settings
from tests.integration.conftest import (
    SKIP_REASON,
    _assert_safe_to_wipe,
    db_reachable,
)

pytestmark = pytest.mark.skipif(not db_reachable(), reason=SKIP_REASON)


def test_conn_string_strips_asyncpg_driver():
    conn_string = checkpoint_conn_string(settings.TEST_DATABASE_URL)
    assert "+asyncpg" not in conn_string
    assert conn_string.startswith("postgresql://")


def tiny_graph(checkpointer):
    async def terminal(state: AgentState):
        return {"history": [Turn(question=state.question, ticker="MSFT")]}

    builder = StateGraph(AgentState)
    builder.add_node("terminal", terminal)
    builder.set_entry_point("terminal")
    builder.add_edge("terminal", END)
    return builder.compile(checkpointer=checkpointer)


def _connection_string():
    # Never connect this test to the application's development/production DB.
    url = settings.TEST_DATABASE_URL
    _assert_safe_to_wipe(url)
    return checkpoint_conn_string(url)


def saver():
    serde = JsonPlusSerializer(allowed_msgpack_modules=list(CHECKPOINTED_TYPES))
    return AsyncPostgresSaver.from_conn_string(_connection_string(), serde=serde)


async def test_postgres_saver_persists_after_close_and_reopen():
    thread = checkpoint_key("test-user", "itest-" + uuid.uuid4().hex)
    config = {"configurable": {"thread_id": thread}}
    async with saver() as first_saver:
        await first_saver.setup()
        graph = tiny_graph(first_saver)
        one = await graph.ainvoke(AgentState(question="turn-one", request_id="r1"), config)
        assert [turn.question for turn in one["history"]] == ["turn-one"]
        two = await graph.ainvoke(AgentState(question="turn-two", request_id="r2"), config)
        assert [turn.question for turn in two["history"]] == ["turn-one", "turn-two"]

    # A new database connection, serializer and compiled graph must load
    # history from PostgreSQL, rather than retaining an in-memory handle.
    async with saver() as reopened:
        try:
            graph = tiny_graph(reopened)
            three = await graph.ainvoke(AgentState(question="turn-three", request_id="r3"), config)
            assert [turn.question for turn in three["history"]] == [
                "turn-one", "turn-two", "turn-three"
            ]
            assert all(isinstance(turn, Turn) for turn in three["history"])
        finally:
            await reopened.adelete_thread(thread)


async def test_postgres_saver_isolates_users_with_same_client_thread_id():
    client_thread = "itest-" + uuid.uuid4().hex
    alice_key = checkpoint_key("alice", client_thread)
    bob_key = checkpoint_key("bob", client_thread)
    async with saver() as checkpoints:
        await checkpoints.setup()
        try:
            graph = tiny_graph(checkpoints)
            await graph.ainvoke(
                AgentState(question="alice-private", request_id="a"),
                {"configurable": {"thread_id": alice_key}},
            )
            bob = await graph.ainvoke(
                AgentState(question="bob-first", request_id="b"),
                {"configurable": {"thread_id": bob_key}},
            )
            assert [turn.question for turn in bob["history"]] == ["bob-first"]
            alice = await graph.aget_state({"configurable": {"thread_id": alice_key}})
            assert [turn.question for turn in alice.values["history"]] == ["alice-private"]
        finally:
            await checkpoints.adelete_thread(alice_key)
            await checkpoints.adelete_thread(bob_key)
