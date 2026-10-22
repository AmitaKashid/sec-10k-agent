"""LLM factory — provider-agnostic chat-model construction.

Dispatches by name prefix so we can swap providers via env var without
touching call sites. See docs/lld.md §LLD 7 — "llm.py" section.

Allowed model names are gated by `settings.ALLOWED_MODELS` at the
API edge (LLD 8); this factory only enforces the prefix→provider map.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from app.settings import settings


def get_chat_model(name: str, streaming: bool = False) -> BaseChatModel:
    """Return a configured chat model. Temperature is always 0 — agent
    behavior must be deterministic for the eval harness."""
    if name.startswith("claude-"):
        return ChatAnthropic(
            model_name=name,
            streaming=streaming,
            temperature=0,
            api_key=settings.ANTHROPIC_API_KEY or None,
            timeout=60,
            stop=None,
        )
    if name.startswith("gpt-"):
        return ChatOpenAI(
            model=name,
            streaming=streaming,
            temperature=0,
            api_key=settings.OPENAI_API_KEY or None,
            timeout=60,
        )
    if name.startswith(("llama-", "mixtral-", "gemma", "openai/gpt-oss-")):
        return ChatGroq(
            model=name,
            streaming=streaming,
            temperature=0,
            api_key=settings.GROQ_API_KEY or None,
            timeout=60,
        )
    raise ValueError(f"Unknown model: {name!r}")


__all__ = ["get_chat_model"]
