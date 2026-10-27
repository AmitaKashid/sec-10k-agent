"""Validate node — pre-validate cache check + frontier LLM with structured output.

Cache hit short-circuits to END (graph routes accordingly). Cache miss
calls the frontier LLM, which emits a draft_answer + structured claims
each pointing back to a source by index. The node maps indices to the
actual XbrlFact / ChunkHit objects so cite_check sees fully-resolved
source_refs.

Streaming: this node is the only one that streams tokens back via SSE.
For 7b we keep the node body non-streaming; the streaming-aware wrapper
lives in the graph layer (LLD 8 SSE route consumes `astream`).
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.agent.groq_runtime import MODEL_ERRORS, failure, select_evidence
from app.agent.llm import get_chat_model
from app.agent.prompts import VALIDATE_PROMPT
from app.agent.state import AgentState, ChunkHit, Claim, XbrlFact, build_turn_from_state
from app.api.usage_tracker import extract_langchain_usage
from app.api.usage_tracker import record as record_usage
from app.data.db import async_session_factory
from app.data.models import AnswerCache
from app.settings import settings


class _ClaimByIdx(BaseModel):
    text: str
    kind: Literal["number", "prose"]
    source_idx: int  # 0-based into the sources list passed to the LLM
    # For prose claims: a verbatim substring of the cited chunk that
    # grounds `text`. Empty for number claims (the XBRL value is the
    # structural ground truth).
    evidence_quote: str = ""


class _ValidateLLMResponse(BaseModel):
    draft_answer: str
    claims: list[_ClaimByIdx] = Field(default_factory=list)


def _compute_cache_key(state: AgentState) -> str:
    if state.xbrl_fact:
        sources_repr = (
            f"xbrl:{state.xbrl_fact.accession_no}:"
            f"{state.xbrl_fact.concept_key}:{state.xbrl_fact.value}"
        )
    elif state.chunks:
        sources_repr = "chunks:" + ",".join(str(c.chunk_id) for c in state.chunks)
    else:
        sources_repr = "no-sources"

    plan_model = state.plan_model_override or settings.PLAN_MODEL
    validate_model = state.validate_model_override or settings.VALIDATE_MODEL
    # Cache key uses the rewritten (self-contained) form when a
    # follow-up was resolved — otherwise the raw question. This keeps
    # cache hits stable across "what was AAPL revenue 2024?" and a
    # follow-up "and 2024?" that resolves to the same form.
    effective_question = (state.rewritten_question or state.question).lower().strip()
    components = [
        "grounding-v3",
        effective_question,
        state.question_type or "",
        state.cik or "",
        str(state.fiscal_year or ""),
        sources_repr,
        plan_model,
        validate_model,
    ]
    return hashlib.sha256("|".join(components).encode()).hexdigest()


def _format_sources(state: AgentState) -> tuple[str, list[XbrlFact | ChunkHit]]:
    """Return (printable_for_prompt, sources_list_indexed_0)."""
    sources: list[XbrlFact | ChunkHit] = []
    lines: list[str] = []

    if state.question_type == "A_numerical" and state.xbrl_fact:
        f = state.xbrl_fact
        sources.append(f)
        lines.append(
            f"[0] XBRL FACT — concept={f.concept_key} unit={f.unit} "
            f"value={f.value} fiscal_year={f.fiscal_year} "
            f"accession_no={f.accession_no}"
        )
    elif state.question_type == "B_qualitative":
        for i, c in enumerate(state.chunks):
            sources.append(c)
            lines.append(f"[{i}] CHUNK item={c.item} accession_no={c.accession_no}\n{c.text}")

    return "\n\n".join(lines), sources


async def validate(state: AgentState) -> dict:
    if state.question_type == "REFUSE":
        return {}

    cache_key = _compute_cache_key(state)

    # --- Pre-validate cache check ---
    async with async_session_factory() as session:
        cached = (
            await session.execute(select(AnswerCache).where(AnswerCache.cache_key == cache_key))
        ).scalar_one_or_none()

        if cached is not None:
            await session.execute(
                update(AnswerCache)
                .where(AnswerCache.cache_key == cache_key)
                .values(hit_count=AnswerCache.hit_count + 1)
            )
            await session.commit()
            return {
                "cache_hit": True,
                "cache_key": cache_key,
                "final_answer": cached.final_answer,
                "verified": True,
                "claims": [Claim.model_validate(c) for c in cached.claims_json],
                "trace_events": state.trace_events + [{"node": "validate", "cache_hit": True}],
                # Cache-hit is the one terminal exit that bypasses
                # cite_check (route_after_validate → END). Append the
                # Turn here too, else a cached answer is invisible to
                # the next turn's history. See conversation_memory_plan §13.
                "history": [build_turn_from_state(state, cached.final_answer)],
            }

    # --- Cache miss: call the LLM ---
    sources_str, sources = _format_sources(state)
    model_name = state.validate_model_override or settings.VALIDATE_MODEL
    try:
        if model_name.startswith("openai/gpt-oss-"):
            if state.question_type == "A_numerical" and state.xbrl_fact:
                # A single structured fact needs formatting, not generative reasoning.
                fact = state.xbrl_fact
                value = format(fact.value, ",f")
                if "." in value:
                    value = value.rstrip("0").rstrip(".")
                display = f"${value}" if fact.unit == "USD" else f"{value} {fact.unit}"
                if fact.unit == "USD/shares":
                    display = f"${value} per share"
                text = f"{state.ticker}'s {fact.concept_key.replace('_', ' ')} was {display}."
                full_claims = [Claim(text=text, kind="number", source_ref=fact)]
            else:
                chat = get_chat_model(model_name, streaming=False)
                full_claims = await select_evidence(chat, state)
            return {
                "cache_key": cache_key,
                "draft_answer": " ".join(c.text for c in full_claims),
                "claims": full_claims,
                "trace_events": state.trace_events
                + [
                    {
                        "node": "validate",
                        "cache_hit": False,
                        "mode": "extractive"
                        if state.question_type == "B_qualitative"
                        else "structured_fact",
                        "claims": len(full_claims),
                    }
                ],
            }
        chat = get_chat_model(model_name, streaming=False)
        structured = chat.with_structured_output(_ValidateLLMResponse, include_raw=True)
        raw_and_parsed = await structured.ainvoke(
            VALIDATE_PROMPT.format(
                question=state.rewritten_question or state.question,
                question_type=state.question_type,
                sources=sources_str or "(no sources)",
            )
        )
        usage = extract_langchain_usage(raw_and_parsed.get("raw"))
        if usage is not None:
            record_usage(state.request_id, "validate", *usage)
        raw = raw_and_parsed.get("raw")
        if raw is not None and raw.response_metadata.get("finish_reason") in (
            "length",
            "content_filter",
        ):
            raise ValueError("Incomplete generation")
        resp = raw_and_parsed.get("parsed")
        if resp is None:
            raise ValueError("Invalid structured response")
    except MODEL_ERRORS as exc:
        return failure(state, "validate", exc)

    # If LLM emitted empty claims, that's the canonical "I can't ground this" signal.
    if not resp.claims or not resp.draft_answer.strip():
        return {
            "cache_key": cache_key,
            "draft_answer": "",
            "claims": [],
            "trace_events": state.trace_events
            + [{"node": "validate", "cache_hit": False, "claims": 0}],
        }

    # Map indices → real source objects.
    full_claims: list[Claim] = []
    for c in resp.claims:
        ref: XbrlFact | ChunkHit | None = None
        if 0 <= c.source_idx < len(sources):
            ref = sources[c.source_idx]
        quote = c.evidence_quote.strip() or None if c.kind == "prose" else None
        full_claims.append(Claim(text=c.text, kind=c.kind, source_ref=ref, evidence_quote=quote))

    return {
        "cache_key": cache_key,
        "draft_answer": resp.draft_answer,
        "claims": full_claims,
        "trace_events": state.trace_events
        + [
            {
                "node": "validate",
                "cache_hit": False,
                "claims": len(full_claims),
            }
        ],
    }


async def write_to_cache(cache_key: str, final_answer: str, claims: list[Claim]) -> None:
    """Called by cite_check after a successful citation pass."""
    claims_json = [c.model_dump(mode="json") for c in claims]
    async with async_session_factory() as session:
        stmt = (
            pg_insert(AnswerCache)
            .values(
                cache_key=cache_key,
                final_answer=final_answer,
                claims_json=claims_json,
            )
            .on_conflict_do_nothing(index_elements=["cache_key"])
        )
        await session.execute(stmt)
        await session.commit()
