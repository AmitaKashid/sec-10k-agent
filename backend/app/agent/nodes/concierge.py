"""Concierge — META-question handler.

Greetings, identity, capability, and help questions hit this node
instead of the refuse path. Streams a warm, scope-describing reply via
the cheap-tier LLM (`settings.REFUSAL_MODEL` — same cheap tier the
composer uses; we don't need a separate setting).

Hard contract: this node never answers domain questions. The prompt
has explicit guardrails, and on any LLM failure we fall back to a
static welcome string — never to silence and never to a domain answer.
"""

from __future__ import annotations

import logging

from app.agent.llm import get_chat_model
from app.agent.prompts import CONCIERGE_PROMPT
from app.agent.state import AgentState, build_turn_from_state
from app.api.usage_tracker import extract_langchain_usage
from app.api.usage_tracker import record as record_usage
from app.settings import settings

logger = logging.getLogger(__name__)

# Static fallback used when the LLM call fails or returns empty. Designed
# to be useful on its own — names a concrete starter question so the user
# can keep moving even if the cheap tier is down.
_STATIC_WELCOME = (
    "Hi! I'm an SEC 10-K analyst — I answer questions about US public "
    "companies grounded in their 10-K filings. Try: \"What was Apple's "
    'revenue in FY2024?" or "What does Microsoft say about AI risks?"'
)

# How many recent turns to surface to the concierge prompt. Small on
# purpose: META is short-lived, and the model only needs enough context
# to vary wording / avoid repeating itself.
_HISTORY_WINDOW = 4


def _format_recent_turns(state: AgentState) -> str:
    """Compact view of recent turns for the concierge prompt.

    Returns "(none)" on the first turn so the prompt placeholder is
    never empty. Each turn is one line: user question + a truncated
    agent reply (final_answer or refusal/meta — whichever was set).
    """
    turns = state.history[-_HISTORY_WINDOW:]
    if not turns:
        return "(none)"
    lines = []
    for t in turns:
        reply = (t.final_answer or "").strip().replace("\n", " ")
        if len(reply) > 200:
            reply = reply[:200] + "…"
        lines.append(f'- User: "{t.question}" | Agent: "{reply}"')
    return "\n".join(lines)


async def concierge(state: AgentState) -> dict:
    """Produce `meta_message` for greetings / capability / identity questions."""
    # Defensive: this node should only run when plan classified META.
    if state.question_type != "META":
        return {}

    prompt = CONCIERGE_PROMPT.format(
        question=state.question,
        recent_turns=_format_recent_turns(state),
    )

    try:
        # streaming=True so LangGraph's `messages` stream mode emits per-
        # token chunks; the `/ask` SSE route forwards them as `event:
        # token`. `.ainvoke()` still returns the assembled message for
        # the fallback / final-state path.
        chat = get_chat_model(settings.REFUSAL_MODEL, streaming=True)
        result = await chat.ainvoke(prompt)
        message = (result.content or "").strip()
        usage = extract_langchain_usage(result)
        if usage is not None:
            record_usage(state.request_id, "concierge", *usage)
    except Exception as exc:  # noqa: BLE001 — fall back, never raise
        logger.warning("concierge.llm_failed", extra={"error": str(exc)})
        message = ""

    if not message:
        message = _STATIC_WELCOME

    logger.info("concierge.composed", extra={"message_len": len(message)})
    return {
        "meta_message": message,
        "trace_events": state.trace_events + [{"node": "concierge", "message_len": len(message)}],
        # Terminal node on the META path — append a Turn so the next
        # turn's `plan` sees this thread had a META exchange. Useful
        # mostly for logging; META turns have no ticker to inherit.
        "history": [build_turn_from_state(state, message)],
    }


__all__ = ["concierge"]
