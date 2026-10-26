"""Retrieve node — branches on question_type.

A_numerical → get_xbrl_fact (single Decimal-typed row)
B_qualitative → multi-query hybrid search (semantic + keyword, top-k merged)

Refuses if the relevant fact/chunks aren't found.
"""

from __future__ import annotations

import logging

from app.agent.state import AgentState, RefusalCode
from app.agent.tools import get_xbrl_fact, rrf_merge, vector_search
from app.data.db import async_session_factory

logger = logging.getLogger(__name__)

# k per sub-query when searching all filings (no year filter)
_SUB_QUERY_K = 10
# k per sub-query when filtered to one specific filing — larger to compensate
# for the smaller chunk pool when only one year's chunks are searched
_SUB_QUERY_K_FILTERED = 15
# Final merged cap returned to the validate node
_MERGED_CAP = 4


async def retrieve(state: AgentState) -> dict:
    if state.question_type == "REFUSE":
        return {}

    if state.question_type == "A_numerical":
        return await _retrieve_a(state)
    return await _retrieve_b(state)


async def _retrieve_a(state: AgentState) -> dict:
    if not (state.cik and state.concept_key):
        return _refuse(state, "Numerical retrieve missing cik or concept_key.", "INTERNAL_ERROR")

    async with async_session_factory() as session:
        fact = await get_xbrl_fact(
            cik=state.cik,
            concept_key=state.concept_key,
            fiscal_year=state.fiscal_year,
            session=session,
        )

    if fact is None:
        return _refuse(
            state,
            f"No {state.concept_key} value found for {state.ticker} "
            f"in fiscal year {state.fiscal_year or 'most-recent'}.",
            "NO_RETRIEVAL_MATCH",
        )

    return {
        "xbrl_fact": fact,
        "trace_events": state.trace_events
        + [
            {
                "node": "retrieve",
                "path": "A",
                "concept_key": state.concept_key,
                "value": str(fact.value),
                "fiscal_year": fact.fiscal_year,
            }
        ],
    }


async def _retrieve_b(state: AgentState) -> dict:
    if not (state.cik and state.item):
        return _refuse(state, "Qualitative retrieve missing cik or item.", "INTERNAL_ERROR")

    # Multi-turn: use plan's self-contained rewrite when this is a
    # follow-up; otherwise fall back to the raw question. Retrieval must
    # run on the rewritten form so the embedding query carries the
    # inherited ticker / fiscal year, not the elided original.
    effective_question = state.rewritten_question or state.question
    sub_queries = await _decompose_question(effective_question)

    # When the user explicitly requested a fiscal year, restrict the search
    # to that filing's chunks so citations always match the requested year.
    # When no year is specified, search all years to maximise relevance.
    # Use a higher k when filtering to compensate for the smaller pool.
    accession_no = (
        state.filing.accession_no if state.filing and state.fiscal_year is not None else None
    )
    k = _SUB_QUERY_K_FILTERED if accession_no else _SUB_QUERY_K

    async with async_session_factory() as session:
        # Each sub-query produces a ranked list via hybrid RRF (semantic+keyword).
        # We then RRF-merge those lists so chunks ranked consistently across
        # multiple sub-queries surface above chunks that only match one.
        result_lists = []
        for q in sub_queries:
            hits = await vector_search(
                cik=state.cik,
                item=state.item,
                question=q,
                session=session,
                k=k,
                accession_no=accession_no,
            )
            result_lists.append(hits)

    hits = rrf_merge(result_lists)[:_MERGED_CAP]

    if not hits:
        return _refuse(
            state,
            f"No relevant passages found in Item {state.item} of "
            f"{state.ticker}'s 10-K for this question.",
            "NO_RETRIEVAL_MATCH",
        )

    return {
        "chunks": hits,
        "trace_events": state.trace_events
        + [
            {
                "node": "retrieve",
                "path": "B",
                "item": state.item,
                "k": len(hits),
                "sub_queries": len(sub_queries),
                "top_score": hits[0].score,
            }
        ],
    }


async def _decompose_question(question: str) -> list[str]:
    """Break a compound question into 2-3 focused sub-queries.

    Compound questions like "cybersecurity AND competition risks" retrieve
    better when split: a single embedding blends both topics and may not
    rank well against chunks that cover only one. Always includes the
    original question so nothing is lost if decomposition is unhelpful.

    Falls back to [question] on any model error so retrieval still works.
    """
    from app.agent.llm import get_chat_model
    from app.settings import settings

    try:
        chat = get_chat_model(settings.PLAN_MODEL, streaming=False)
        prompt = (
            "Split the following research question into 2-3 short, focused "
            "sub-queries optimised for searching SEC 10-K text. "
            "Output ONLY the sub-queries, one per line, no bullets or numbers.\n\n"
            f"Question: {question}"
        )
        result = await chat.ainvoke(prompt)
        lines = [ln.strip() for ln in result.content.strip().splitlines() if ln.strip()]
        # Deduplicate while preserving order; always keep the original first.
        seen: dict[str, None] = {question: None}
        for ln in lines[:2]:
            seen[ln] = None
        return list(seen.keys())
    except Exception as exc:  # noqa: BLE001
        logger.warning("question decomposition failed (%s), using original", exc)
        return [question]


def _refuse(state: AgentState, reason: str, code: RefusalCode) -> dict:
    return {
        "question_type": "REFUSE",
        "refusal_reason": reason,
        "refusal_code": code,
        "trace_events": state.trace_events
        + [{"node": "retrieve", "refused": True, "reason": reason, "refusal_code": code}],
    }
