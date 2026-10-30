"""Exercise real /ask thread routing with a real in-memory checkpointer.

Only graph nodes, identity resolution and token counting are replaced.
No database, Clerk credentials, tokenizer downloads or LLM calls are used.
"""
from __future__ import annotations

import json
from contextlib import nullcontext

import httpx
import pytest
from fastapi import FastAPI, Request
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from app.agent.memory import checkpoint_key
from app.agent.state import AgentState, Turn
from app.api import routes
from app.api.auth import Identity
from app.data.db import get_session


def tiny_graph(checkpointer):
    async def terminal(state: AgentState):
        return {
            "question_type": "META",
            "meta_message": f"prior-turns:{len(state.history)}",
            "history": [Turn(question=state.question)],
        }

    builder = StateGraph(AgentState)
    builder.add_node("terminal", terminal)
    builder.set_entry_point("terminal")
    builder.add_edge("terminal", END)
    return builder.compile(checkpointer=checkpointer)


@pytest.fixture
def memory_app(monkeypatch):
    monkeypatch.setattr(routes.settings, "ANON_LIFETIME_ENABLED", False)
    monkeypatch.setattr(routes.settings, "ASK_TOKEN_LIMITER_ENABLED", False)
    monkeypatch.setattr(routes.settings, "ASK_ANON_MAX_INPUT_TOKENS", 100)
    monkeypatch.setattr(routes.settings, "ASK_USER_MAX_INPUT_TOKENS", 100)
    monkeypatch.setattr(routes.settings, "EXPOSE_TRACE_URLS", False)
    monkeypatch.setattr(routes, "_count_tokens", lambda text: 1)
    monkeypatch.setattr(routes, "request_trace", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(routes, "update_trace_output", lambda *a: None)
    monkeypatch.setattr(routes, "langfuse_flush", lambda: None)

    async def no_db():
        yield None

    async def identity(request: Request):
        # A test-only identity dependency, never installed in the real app.
        user = request.headers.get("x-test-user")
        return Identity(tier="user" if user else "anon", key=user or "test-ip")

    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_session] = no_db
    app.dependency_overrides[routes.resolve_identity] = identity
    app.state.graph = tiny_graph(InMemorySaver())
    app.state.memory_active = True
    monkeypatch.setattr(routes, "GRAPH", tiny_graph(None))
    return app


async def ask(app, user, thread, question, accept):
    headers = {"Accept": accept}
    if user:
        headers["x-test-user"] = user
    body = {"question": question}
    if thread is not None:
        body["thread_id"] = thread
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/ask", json=body, headers=headers)
    assert response.status_code == 200, response.text
    if accept == "application/json":
        return response.json()
    for block in response.text.replace("\r\n", "\n").split("\n\n"):
        lines = block.splitlines()
        if "event: done" in lines:
            return json.loads("\n".join(line[6:] for line in lines if line.startswith("data: ")))
    pytest.fail("Stream did not contain a terminal done event: " + response.text)


@pytest.mark.parametrize("accept", ["application/json", "text/event-stream"])
async def test_same_user_and_thread_resume_history(memory_app, accept):
    first = await ask(memory_app, "alice", "conversation", "first", accept)
    second = await ask(memory_app, "alice", "conversation", "second", accept)
    assert first["meta_message"] == "prior-turns:0"
    assert second["meta_message"] == "prior-turns:1"
    assert first["thread_id"] == second["thread_id"] == "conversation"


@pytest.mark.parametrize("accept", ["application/json", "text/event-stream"])
async def test_different_users_cannot_share_client_thread_id(memory_app, accept):
    await ask(memory_app, "alice", "shared-id", "alice-private", accept)
    bob = await ask(memory_app, "bob", "shared-id", "bob-first", accept)
    assert bob["meta_message"] == "prior-turns:0"
    alice_state = await memory_app.state.graph.aget_state(
        {"configurable": {"thread_id": checkpoint_key("alice", "shared-id")}}
    )
    assert [turn.question for turn in alice_state.values["history"]] == ["alice-private"]


@pytest.mark.parametrize("accept", ["application/json", "text/event-stream"])
async def test_new_thread_has_no_previous_history(memory_app, accept):
    await ask(memory_app, "alice", "old-thread", "first", accept)
    fresh = await ask(memory_app, "alice", "new-thread", "follow-up", accept)
    assert fresh["meta_message"] == "prior-turns:0"


@pytest.mark.parametrize("accept", ["application/json", "text/event-stream"])
async def test_anonymous_requests_stay_stateless(memory_app, accept):
    await ask(memory_app, None, "same-thread", "first", accept)
    second = await ask(memory_app, None, "same-thread", "second", accept)
    assert second["thread_id"] is None
    assert second["meta_message"] == "prior-turns:0"


@pytest.mark.parametrize("accept", ["application/json", "text/event-stream"])
async def test_memory_inactive_does_not_echo_thread_id(memory_app, accept):
    memory_app.state.memory_active = False
    memory_app.state.graph = tiny_graph(None)
    first = await ask(memory_app, "alice", "client-thread", "first", accept)
    second = await ask(memory_app, "alice", "client-thread", "second", accept)
    assert first["thread_id"] is second["thread_id"] is None
    assert second["meta_message"] == "prior-turns:0"
