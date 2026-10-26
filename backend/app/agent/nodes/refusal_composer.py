"""Refusal composer — last node on the refuse path before END.

Takes the structured refusal context (code, reason, ticker, fiscal year,
available years) and produces user-facing prose in `refusal_message`.
The deterministic `refusal_reason` is preserved untouched for evals
and logs.

Every refusal — including OFF_TOPIC — flows through the cheap-tier LLM
(`settings.REFUSAL_MODEL`) so the response feels conversational rather
than canned. Cost is bounded by the upstream rate limits (slowapi per
API key, Vercel upstash per IP) and the cheap-tier model choice.

On any LLM failure, falls back to the existing `refusal_reason` (or a
generic fallback for OFF_TOPIC) so the user always gets *something* —
refusal must never depend on a successful LLM call.
"""

from __future__ import annotations

import logging

from app.agent.llm import get_chat_model
from app.agent.prompts import COMPOSE_REFUSAL_PROMPT
from app.agent.state import AgentState, build_turn_from_state
from app.api.usage_tracker import extract_langchain_usage
from app.api.usage_tracker import record as record_usage
from app.settings import settings

logger = logging.getLogger(__name__)

# Last-resort fallback for the LLM-failure path when no `refusal_reason`
# is available (e.g. OFF_TOPIC where the plan LLM didn't shape one and
# the composer LLM also failed).
_GENERIC_FALLBACK = (
    "I only answer questions about US public-company SEC 10-K filings. "
    'Try something like: "What was Apple\'s revenue in FY2024?" or '
    '"What does Microsoft say about AI risks in its 10-K?"'
)


async def refusal_composer(state: AgentState) -> dict:
    """Produce `refusal_message` from the structured refusal context."""
    # Defensive: composer should only run on the refuse path. If routed
    # here in error, no-op rather than fabricating a refusal.
    if state.question_type != "REFUSE" and not state.refusal_reason_final:
        return {}

    code = state.refusal_code
    reason = state.refusal_reason_final or state.refusal_reason or ""

    # Do not call a failing/quota-limited provider again to explain its failure.
    if code == "INTERNAL_ERROR":
        return {
            "refusal_message": reason or "The model service is temporarily unavailable.",
            "history": [build_turn_from_state(state, reason)],
        }

    prompt = COMPOSE_REFUSAL_PROMPT.format(
        question=state.rewritten_question or state.question,
        refusal_code=code or "UNKNOWN",
        refusal_reason=reason,
        ticker=state.ticker or "—",
        fiscal_year=state.fiscal_year if state.fiscal_year is not None else "—",
        available_fiscal_years=(
            ", ".join(str(y) for y in state.available_fiscal_years)
            if state.available_fiscal_years
            else "—"
        ),
    )

    try:
        # streaming=True so LangGraph's `messages` stream mode picks up
        # per-token chunks; the `/ask` SSE route forwards them as
        # `event: token`. `.ainvoke()` still returns the assembled message.
        chat = get_chat_model(settings.REFUSAL_MODEL, streaming=True)
        result = await chat.ainvoke(prompt)
        message = (result.content or "").strip()
        usage = extract_langchain_usage(result)
        if usage is not None:
            record_usage(state.request_id, "refusal_composer", *usage)
    except Exception as exc:  # noqa: BLE001 — fall back, never raise
        logger.warning(
            "refusal_composer.llm_failed",
            extra={"refusal_code": code, "error": str(exc)},
        )
        message = ""

    if not message:
        # Fallback: keep the structured reason so the user sees *something*.
        message = reason or _GENERIC_FALLBACK

    logger.info(
        "refusal_composer.composed",
        extra={"refusal_code": code, "message_len": len(message)},
    )
    return {
        "refusal_message": message,
        "trace_events": state.trace_events
        + [
            {
                "node": "refusal_composer",
                "refusal_code": code,
                "message_len": len(message),
            }
        ],
        # Terminal node on the refuse path — append the Turn so the
        # next turn's `plan` can apply rule 3 (don't inherit context
        # from refused turns).
        "history": [build_turn_from_state(state, message)],
    }


__all__ = ["refusal_composer"]
