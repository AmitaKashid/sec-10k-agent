"""Plan node — classifies the question and extracts structured fields.

LLM (cheap, structured output) → PlanResult. Sets state.question_type,
ticker, fiscal_year, concept_key/item, refusal_reason.

Refuses for: off-topic, no ticker, YoY, cross-company, unknown concept,
non-{1A,7,8} item.

Multi-turn: when `state.history` is non-empty (only possible with the
checkpointer enabled), plan also decides `is_followup` and may emit a
self-contained `rewritten_question` carrying forward the prior turn's
ticker / fiscal_year / concept. Downstream nodes read
`state.rewritten_question or state.question`.

Per-turn reset: plan is the entry node, so it returns `RESET_DEFAULTS`
merged with the new classification. Without this, fields persisted by
the checkpointer from turn N-1 (chunks, claims, final_answer, ...) would
leak into turn N's working memory.
"""

from __future__ import annotations

import logging

from app.agent.groq_runtime import MODEL_ERRORS, failure, invoke_strict
from app.agent.llm import get_chat_model
from app.agent.prompts import build_plan_prompt
from app.agent.state import RESET_DEFAULTS, AgentState, PlanResult
from app.api.usage_tracker import extract_langchain_usage
from app.api.usage_tracker import record as record_usage
from app.settings import settings

logger = logging.getLogger(__name__)


async def plan(state: AgentState) -> dict:
    model_name = state.plan_model_override or settings.PLAN_MODEL
    logger.info(
        "plan.llm.request",
        extra={
            "question": state.question,
            "model": model_name,
            "history_len": len(state.history),
        },
    )
    prompt = build_plan_prompt(state.question, state.history)
    try:
        chat = get_chat_model(model_name, streaming=False)
        if model_name.startswith("openai/gpt-oss-"):
            result = await invoke_strict(chat, PlanResult, prompt, state.request_id, "plan")
        else:
            structured = chat.with_structured_output(PlanResult, include_raw=True)
            raw_and_parsed = await structured.ainvoke(prompt)
            result = raw_and_parsed.get("parsed")
            if result is None:
                raise ValueError("Invalid planner response")
            usage = extract_langchain_usage(raw_and_parsed.get("raw"))
            if usage is not None:
                record_usage(state.request_id, "plan", *usage)
    except MODEL_ERRORS as exc:
        return {**RESET_DEFAULTS, **failure(state, "plan", exc)}

    logger.info(
        "plan.llm.response",
        extra={
            "question_type": result.question_type,
            "ticker": result.ticker.upper() if result.ticker else None,
            "fiscal_year": result.fiscal_year,
            "concept_key": result.concept_key,
            "item": result.item,
            "refusal_reason": result.refusal_reason,
            "refusal_code": result.refusal_code,
            "is_followup": result.is_followup,
            "rewritten_question": result.rewritten_question,
        },
    )

    event = {
        "node": "plan",
        "question_type": result.question_type,
        "ticker": result.ticker.upper() if result.ticker else None,
        "refused": result.question_type == "REFUSE",
        "refusal_code": result.refusal_code,
        "is_followup": result.is_followup,
    }
    # RESET_DEFAULTS wipes per-turn working memory carried by the
    # checkpointer from a prior turn. With no checkpointer the state is
    # already fresh and the spread is a no-op. `history` is intentionally
    # absent — it accumulates via the operator.add reducer.
    classification = {
        "question_type": result.question_type,
        "ticker": result.ticker.upper() if result.ticker else None,
        "fiscal_year": result.fiscal_year,
        "concept_key": result.concept_key,
        "item": result.item,
        "refusal_reason": result.refusal_reason,
        "refusal_code": result.refusal_code,
        "is_followup": result.is_followup,
        "rewritten_question": result.rewritten_question,
    }
    return {
        **RESET_DEFAULTS,
        **classification,
        # `trace_events` is in RESET_DEFAULTS as []. Plan owns the first
        # event of the turn, so start the list with just this event.
        "trace_events": [event],
    }
