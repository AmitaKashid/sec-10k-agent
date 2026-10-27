"""Plain async functions called by the retrieve node — *not* LLM tools.

The graph routing decides when each is invoked; the LLM never picks.
This is intentional: tool-call agents can hallucinate freely between
calls, but a fixed graph + structured Claim/source_ref makes the
hallucination-proofing in cite_check actually enforceable.

See docs/lld.md §LLD 7.
"""

from __future__ import annotations

import logging

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.state import ChunkHit
from app.data.embedder import embed_texts
from app.data.models import XbrlFact as XbrlFactRow
from app.data.xbrl import XbrlFact

logger = logging.getLogger(__name__)

# RRF smoothing constant — standard value from the original paper.
# Prevents top-1 results from dominating; at k=60 the score difference
# between rank-1 and rank-2 is only ~0.03%.
_RRF_K = 60


async def get_xbrl_fact(
    cik: str,
    concept_key: str,
    fiscal_year: int | None,
    session: AsyncSession,
) -> XbrlFact | None:
    """Lookup one FY XBRL fact for (cik, concept_key, year). Returns None if absent.

    `concept_key` is the friendly key ("revenue", "net_income", …) — the
    DB stores friendly keys in the `concept` column (LLD 6 decision).
    `fiscal_year=None` → most recent FY available for that concept.
    """
    stmt = select(XbrlFactRow).where(
        XbrlFactRow.cik == cik,
        XbrlFactRow.concept == concept_key,
        XbrlFactRow.fiscal_period == "FY",
    )
    if fiscal_year is not None:
        stmt = stmt.where(XbrlFactRow.fiscal_year == fiscal_year)
    stmt = stmt.order_by(XbrlFactRow.period_end.desc()).limit(1)

    row = (await session.execute(stmt)).scalar_one_or_none()
    logger.info(
        "retrieve.xbrl_fact",
        extra={
            "cik": cik,
            "concept_key": concept_key,
            "fiscal_year": fiscal_year,
            "found": row is not None,
        },
    )
    if row is None:
        return None

    return XbrlFact(
        cik=row.cik,
        concept=row.concept,
        concept_key=row.concept,
        unit=row.unit,
        value=row.value,
        period_end=row.period_end,
        fiscal_year=row.fiscal_year,
        fiscal_period="FY",
        accession_no=row.accession_no,
        form=row.form,
    )


def rrf_merge(ranked_lists: list[list[ChunkHit]], k: int = _RRF_K) -> list[ChunkHit]:
    """Reciprocal Rank Fusion over multiple ranked lists of ChunkHits.

    Each chunk receives a score of 1/(k + rank) for every list it appears
    in, where rank is 0-based. Scores are summed across lists so chunks
    that appear consistently near the top of multiple lists rank highest.

    Example — why this beats "best score wins":
        chunk A: rank 0 in list-1 only       → 1/61  = 0.0164
        chunk B: rank 2 in all 3 lists        → 3/63  = 0.0476  ← wins

    Returns a new list sorted by descending RRF score. The .score field
    on each ChunkHit retains the original cosine distance from the
    semantic search (lower = better) so trace events stay interpretable.
    """
    rrf_scores: dict[int, float] = {}
    chunks: dict[int, ChunkHit] = {}

    for result_list in ranked_lists:
        for rank, hit in enumerate(result_list):
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            if hit.chunk_id not in chunks:
                chunks[hit.chunk_id] = hit

    return sorted(chunks.values(), key=lambda h: -rrf_scores[h.chunk_id])


async def vector_search(
    cik: str,
    item: str,
    question: str,
    session: AsyncSession,
    k: int = 5,
    accession_no: str | None = None,
) -> list[ChunkHit]:
    """Hybrid retrieval: RRF fusion of semantic (cosine) + keyword (ts_rank).

    Fetches k candidates from each signal independently, then merges with
    Reciprocal Rank Fusion. This avoids the scale-mismatch problem of a
    linear weighted sum (cosine distance is 0–2; ts_rank is 0–1) and lets
    each signal contribute proportionally to how it ranks, not how it scores.

    A chunk that ranks #3 in both semantic and keyword lists scores higher
    than one that ranks #1 in only one — consistent relevance wins.
    """
    if not question.strip():
        return []

    semantic = await _semantic_search(cik, item, question, session, k=k, accession_no=accession_no)
    keyword = await _keyword_search(cik, item, question, session, k=k, accession_no=accession_no)
    merged = rrf_merge([semantic, keyword])[:k]

    logger.info(
        "retrieve.hybrid_search",
        extra={
            "cik": cik,
            "item": item,
            "accession_no": accession_no,
            "k": k,
            "semantic_hits": len(semantic),
            "keyword_hits": len(keyword),
            "merged_hits": len(merged),
            "top_semantic_score": float(semantic[0].score) if semantic else None,
        },
    )
    return merged


# ── Internal search primitives ────────────────────────────────────────────────


def _chunk_from_row(row: object) -> ChunkHit:
    return ChunkHit(
        chunk_id=row.id,  # type: ignore[attr-defined]
        accession_no=row.accession_no,
        item=row.item,
        seq=row.seq,
        text=row.text,
        score=float(row.score),
        char_start=row.char_start,
        char_end=row.char_end,
        ticker=row.ticker,
        company_name=row.company_name,
        cik=row.cik,
        fiscal_year=row.fiscal_year,
        filing_date=row.filing_date,
        section_title=row.section_title,
    )


_SQL_SEMANTIC = text(
    """
    SELECT id, accession_no, item, seq, text, char_start, char_end,
           ticker, company_name, cik, fiscal_year, filing_date, section_title,
           embedding <=> CAST(:q_vec AS vector) AS score
    FROM chunks
    WHERE cik = :cik AND item = :item
    ORDER BY score ASC
    LIMIT :k
    """
)

_SQL_SEMANTIC_FILTERED = text(
    """
    SELECT id, accession_no, item, seq, text, char_start, char_end,
           ticker, company_name, cik, fiscal_year, filing_date, section_title,
           embedding <=> CAST(:q_vec AS vector) AS score
    FROM chunks
    WHERE cik = :cik AND item = :item AND accession_no = :accession_no
    ORDER BY score ASC
    LIMIT :k
    """
)

_SQL_KEYWORD = text(
    """
    SELECT id, accession_no, item, seq, text, char_start, char_end,
           ticker, company_name, cik, fiscal_year, filing_date, section_title,
           ts_rank(to_tsvector('english', text),
                   plainto_tsquery('english', :keywords), 1) AS score
    FROM chunks
    WHERE cik = :cik AND item = :item
      AND to_tsvector('english', text) @@ plainto_tsquery('english', :keywords)
    ORDER BY score DESC
    LIMIT :k
    """
)

_SQL_KEYWORD_FILTERED = text(
    """
    SELECT id, accession_no, item, seq, text, char_start, char_end,
           ticker, company_name, cik, fiscal_year, filing_date, section_title,
           ts_rank(to_tsvector('english', text),
                   plainto_tsquery('english', :keywords), 1) AS score
    FROM chunks
    WHERE cik = :cik AND item = :item AND accession_no = :accession_no
      AND to_tsvector('english', text) @@ plainto_tsquery('english', :keywords)
    ORDER BY score DESC
    LIMIT :k
    """
)


async def _semantic_search(
    cik: str,
    item: str,
    question: str,
    session: AsyncSession,
    k: int,
    accession_no: str | None = None,
) -> list[ChunkHit]:
    """Pure cosine-similarity search. Returns ascending score (lower = closer)."""
    [q_vec] = await embed_texts([question])

    if accession_no:
        sql = _SQL_SEMANTIC_FILTERED
        params: dict = {
            "q_vec": str(q_vec),
            "cik": cik,
            "item": item,
            "k": k,
            "accession_no": accession_no,
        }
    else:
        sql = _SQL_SEMANTIC
        params = {"q_vec": str(q_vec), "cik": cik, "item": item, "k": k}

    res = await session.execute(sql, params)
    return [_chunk_from_row(row) for row in res]


async def _keyword_search(
    cik: str,
    item: str,
    question: str,
    session: AsyncSession,
    k: int,
    accession_no: str | None = None,
) -> list[ChunkHit]:
    """BM25-style keyword search via Postgres ts_rank.

    Only returns rows that match the query (uses @@ operator). The GIN index
    on to_tsvector('english', text) makes this O(log n) instead of a seq scan.
    The .score field is set to 1.0 since ChunkHit.score is semantically "cosine
    distance" (lower=better) — keyword-only hits must not skew trace events.
    Returns empty list on any query-parse error (e.g. plainto_tsquery rejection).
    """
    keywords = question[:500]  # guard against extremely long questions

    if accession_no:
        sql = _SQL_KEYWORD_FILTERED
        params: dict = {
            "cik": cik,
            "item": item,
            "keywords": keywords,
            "k": k,
            "accession_no": accession_no,
        }
    else:
        sql = _SQL_KEYWORD
        params = {"cik": cik, "item": item, "keywords": keywords, "k": k}

    try:
        res = await session.execute(sql, params)
        rows = list(res)
        return [
            ChunkHit(
                chunk_id=row.id,
                accession_no=row.accession_no,
                item=row.item,
                seq=row.seq,
                text=row.text,
                score=1.0,
                char_start=row.char_start,
                char_end=row.char_end,
                ticker=row.ticker,
                company_name=row.company_name,
                cik=row.cik,
                fiscal_year=row.fiscal_year,
                filing_date=row.filing_date,
                section_title=row.section_title,
            )
            for row in rows
        ]
    except Exception as exc:  # noqa: BLE001
        logger.error("keyword search failed (cik=%s item=%s): %s", cik, item, exc)
        return []


__all__ = ["get_xbrl_fact", "vector_search", "rrf_merge"]
