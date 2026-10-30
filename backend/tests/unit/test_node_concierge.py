"""Concierge node — META-question handler."""

from __future__ import annotations

import pytest

from app.agent.nodes import concierge as concierge_module
from app.agent.state import AgentState


class _FakeMsg:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChat:
    def __init__(self, response: str) -> None:
        self._response = response
        self.last_prompt: str | None = None
        self.streaming_kw: bool | None = None

    async def ainvoke(self, prompt: str) -> _FakeMsg:
        self.last_prompt = prompt
        return _FakeMsg(self._response)


class _RaisingChat:
    async def ainvoke(self, prompt: str) -> _FakeMsg:
        raise RuntimeError("simulated LLM outage")


def _patch_chat(monkeypatch, chat) -> None:
    captured = {}

    def factory(*_a, **kw):
        captured["streaming"] = kw.get("streaming")
        return chat

    monkeypatch.setattr(concierge_module, "get_chat_model", factory)
    return captured


@pytest.mark.asyncio
async def test_happy_path_writes_meta_message(monkeypatch) -> None:
    fake = _FakeChat("Hi! I'm an SEC 10-K analyst. Try asking about Apple's revenue in FY2024.")
    captured = _patch_chat(monkeypatch, fake)

    state = AgentState(
        question="hi",
        request_id="r1",
        question_type="META",
    )
    out = await concierge_module.concierge(state)
    assert "10-K" in out["meta_message"]
    # Streaming must be enabled so SSE can emit per-token chunks.
    assert captured["streaming"] is True
    # Prompt got the user's question.
    assert "hi" in (fake.last_prompt or "")
    assert any(e.get("node") == "concierge" for e in out["trace_events"])


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_static_welcome(monkeypatch) -> None:
    """Concierge must never raise — fall back to the static welcome on outage."""
    _patch_chat(monkeypatch, _RaisingChat())

    state = AgentState(
        question="who are you?",
        request_id="r2",
        question_type="META",
    )
    out = await concierge_module.concierge(state)
    # The static fallback is non-empty and mentions the agent's purpose.
    assert out["meta_message"]
    assert "10-K" in out["meta_message"]


@pytest.mark.asyncio
async def test_no_op_when_not_meta(monkeypatch) -> None:
    """Concierge guards against being routed in error on a non-META state."""
    monkeypatch.setattr(
        concierge_module,
        "get_chat_model",
        lambda *_a, **_kw: pytest.fail("LLM should not be called for non-META"),
    )
    state = AgentState(
        question="What was AAPL revenue in 2024?",
        request_id="r3",
        question_type="A_numerical",
    )
    out = await concierge_module.concierge(state)
    assert out == {}
