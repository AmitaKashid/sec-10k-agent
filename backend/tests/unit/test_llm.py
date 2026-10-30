"""LLM factory dispatch tests. We don't construct real clients —
instantiating ChatAnthropic etc. would require API keys at import time
and try to validate them. We patch the constructors and assert dispatch.
"""

from __future__ import annotations

import pytest

from app.agent import llm


@pytest.fixture
def patched_clients(monkeypatch):
    calls = {}

    class _Sentinel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_anthropic(**kwargs):
        calls["anthropic"] = kwargs
        return _Sentinel(**kwargs)

    def fake_openai(**kwargs):
        calls["openai"] = kwargs
        return _Sentinel(**kwargs)

    def fake_groq(**kwargs):
        calls["groq"] = kwargs
        return _Sentinel(**kwargs)

    monkeypatch.setattr(llm, "ChatAnthropic", fake_anthropic)
    monkeypatch.setattr(llm, "ChatOpenAI", fake_openai)
    monkeypatch.setattr(llm, "ChatGroq", fake_groq)
    return calls


def test_claude_prefix_routes_to_anthropic(patched_clients):
    llm.get_chat_model("claude-haiku-4-5")
    assert "anthropic" in patched_clients
    assert patched_clients["anthropic"]["model_name"] == "claude-haiku-4-5"
    assert patched_clients["anthropic"]["temperature"] == 0


def test_gpt_prefix_routes_to_openai(patched_clients):
    llm.get_chat_model("gpt-4o-mini", streaming=True)
    assert "openai" in patched_clients
    assert patched_clients["openai"]["model"] == "gpt-4o-mini"
    assert patched_clients["openai"]["streaming"] is True


def test_llama_prefix_routes_to_groq(patched_clients):
    llm.get_chat_model("llama-3.3-70b-versatile")
    assert "groq" in patched_clients
    assert patched_clients["groq"]["model"] == "llama-3.3-70b-versatile"


def test_mixtral_prefix_routes_to_groq(patched_clients):
    llm.get_chat_model("mixtral-8x7b")
    assert "groq" in patched_clients


def test_gemma_prefix_routes_to_groq(patched_clients):
    llm.get_chat_model("gemma-7b-it")
    assert "groq" in patched_clients


def test_unknown_prefix_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        llm.get_chat_model("bedrock-cohere")


def test_temperature_always_zero(patched_clients):
    """Determinism — required by the eval harness gate."""
    llm.get_chat_model("gpt-4o-mini")
    llm.get_chat_model("claude-haiku-4-5")
    llm.get_chat_model("llama-3.1-8b-instant")
    assert patched_clients["openai"]["temperature"] == 0
    assert patched_clients["anthropic"]["temperature"] == 0
    assert patched_clients["groq"]["temperature"] == 0
