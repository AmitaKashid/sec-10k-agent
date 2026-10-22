"""Ingestion orchestrator — wires SEC fetch → parse → chunk → embed → DB.

See docs/lld.md §LLD 6. Idempotent across runs via per-filing
`ingestion_status` gate + `ON CONFLICT DO NOTHING` on chunks/xbrl_facts.
Filing-level errors (parser, EDGAR 404, unexpected bugs) are logged and
the filing is marked `failed` so a re-run with `retry_failed=True` can
pick it up. Systemic errors (rate limits, OpenAI quota/connectivity, DB
outage) propagate so the operator can resume from a clean state.
"""

from __future__ import annotations

import asyncio
import csv
import logging
import time
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError
from pydantic import BaseModel
from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.data import edgar_client, parser, xbrl
from app.data.chunker import Chunk as ChunkDTO
from app.data.chunker import chunk_section
from app.data.db import async_session_factory
from app.data.edgar_client import (
    EdgarError,
    EdgarRateLimited,
    EdgarServerError,
    FilingMeta,
)
from app.data.models import Chunk, Company, Filing, XbrlFact
from app.data.parser import extract_headings

logger = logging.getLogger(__name__)

# OpenAI text-embedding-3-small published rate (USD per 1M tokens).
EMBEDDING_USD_PER_1M = Decimal("0.02")

ITEM_KEYS: tuple[str, ...] = ("1A", "7", "8")
DEFAULT_YEARS_BACK = 3
DEFAULT_CONCURRENCY = 5

_ITEM_NAMES = {
    "1A": "Risk Factors",
    "7": "Management Discussion and Analysis",
    "8": "Financial Statements and Supplementary Data",
}


def _heading_for_offset(char_start: int, headings: list[tuple[int, str]]) -> str | None:
    """Return the last heading whose offset is <= char_start, or None."""
    result = None
    for offset, title in headings:
        if offset <= char_start:
            result = title
        else:
            break
    return result


def _embedding_text(
    chunk_text: str, company_name: str, ticker: str, fiscal_year: int, item: str
) -> str:
    """Prefix chunk text with filing context before embedding.

    Contextual embeddings dramatically improve retrieval: a chunk that
    says "nation-state actors undertake attacks" embeds without context,
    so a query about "Microsoft cybersecurity risks" may not find it.
    Adding "Microsoft (MSFT) 10-K FY2024 — Item 1A Risk Factors:" to the
    embedded text aligns the chunk's vector with queries that mention the
    company and topic, without changing the stored chunk text (which is
    still used for verbatim cite-check matching).
    """
    section = _ITEM_NAMES.get(item, f"Item {item}")
    return f"{company_name} ({ticker}) 10-K FY{fiscal_year} — Item {item} {section}:\n{chunk_text}"


Status = Literal["success", "partial", "failed", "skipped"]


class IngestionResult(BaseModel):
    ticker: str
    status: Status
    filings_ingested: int = 0
    filings_failed: int = 0
    xbrl_facts_inserted: int = 0
    embedding_cost_usd: Decimal = Decimal("0")
    error: str | None = None


# --- Per-company orchestration -------------------------------------------


async def ingest_company(ticker: str, retry_failed: bool = False) -> IngestionResult:
    ticker = ticker.upper()
    cik = await edgar_client.get_cik_for_ticker(ticker)
    if cik is None:
        return IngestionResult(
            ticker=ticker, status="failed", error=f"ticker not in EDGAR map: {ticker}"
        )
    cik = cik.zfill(10)

    try:
        submissions = await edgar_client.get_submissions(cik, "10-K")
    except (EdgarRateLimited, EdgarServerError):
        raise
    except EdgarError as e:
        return IngestionResult(ticker=ticker, status="failed", error=str(e))

    if not submissions:
        return IngestionResult(ticker=ticker, status="failed", error="no 10-K filings found")

    # Top-N by filing_date desc (decision #6 in LLD).
    recent = sorted(submissions, key=lambda f: f.filing_date, reverse=True)[:DEFAULT_YEARS_BACK]

    # Fetch facts up front: gives us company name + lets a bad facts URL fail
    # before we burn quota on filing fetches.
    facts_json = await edgar_client.get_company_facts(cik)
    company_name = facts_json.get("entityName") or ticker

    ingested = 0
    failed = 0
    skipped_already_done = 0
    cost = Decimal("0")
    xbrl_inserted = 0

    async with async_session_factory() as session:
        await _upsert_company(session, cik=cik, ticker=ticker, name=company_name)
        await session.commit()

        # Pre-insert filing rows so XBRL's FK filter sees them, and so
        # XBRL ingestion is durable even if the embedding loop crashes.
        to_process: list[FilingMeta] = []
        for meta in recent:
            existing = await _get_filing_status(session, meta.accession_no)
            if existing == "embedded":
                skipped_already_done += 1
                continue
            if existing == "failed" and not retry_failed:
                skipped_already_done += 1
                continue
            if existing is None:
                await _insert_filing_pending(session, cik=cik, meta=meta)
            elif existing == "failed" and retry_failed:
                await _set_filing_status(session, meta.accession_no, "pending")
            to_process.append(meta)
        await session.commit()

        # XBRL is independent of embeddings — insert it before the fragile
        # OpenAI loop so a mid-run crash doesn't empty the numerical table.
        xbrl_inserted = await _insert_xbrl_facts(session, cik=cik, facts_json=facts_json)
        await session.commit()

        for meta in to_process:
            try:
                filing_cost = await _process_filing(
                    session,
                    cik=cik,
                    meta=meta,
                    company_name=company_name,
                    ticker=ticker,
                )
                cost += filing_cost
                ingested += 1
            except (
                EdgarRateLimited,
                EdgarServerError,
                RateLimitError,
                APIError,
                APIConnectionError,
                APITimeoutError,
            ):
                # Systemic — let it propagate. Already-committed work persists.
                # APIConnectionError/APITimeoutError are included because
                # `_embed_with_usage`'s tenacity wrapper retries them and
                # re-raises after exhausting attempts; treating them as
                # filing-level "failed" would silently burn retries on a
                # global outage.
                raise
            except Exception as e:  # noqa: BLE001
                # Any other failure (parser, embed, transient OpenAI, ...): mark
                # the filing failed so it's visible and a re-run can retry it,
                # rather than leaving it stuck in 'parsed' silently.
                logger.warning(
                    "[%s %s] filing failed: %s: %s",
                    ticker,
                    meta.accession_no,
                    type(e).__name__,
                    e,
                )
                await session.rollback()
                await _set_filing_status(session, meta.accession_no, "failed")
                await session.commit()
                failed += 1

    if ingested == 0 and failed == 0:
        status: Status = "skipped"
    elif failed == 0:
        status = "success"
    elif ingested > 0:
        status = "partial"
    else:
        status = "failed"

    return IngestionResult(
        ticker=ticker,
        status=status,
        filings_ingested=ingested,
        filings_failed=failed,
        xbrl_facts_inserted=xbrl_inserted,
        embedding_cost_usd=cost.quantize(Decimal("0.000001")),
    )


# --- Filing pipeline ------------------------------------------------------


async def _process_filing(
    session: AsyncSession,
    *,
    cik: str,
    meta: FilingMeta,
    company_name: str,
    ticker: str,
) -> Decimal:
    """Fetch HTML → parse → chunk → embed → bulk insert. Returns USD cost.

    On success, marks filing 'embedded' and commits. On any failure
    (EdgarNotFound, ParserAllTiersFailed, transient embedding errors,
    unexpected bugs) the caller catches the exception, rolls back any
    uncommitted work, and marks the filing 'failed'. Systemic errors
    (rate limits, OpenAI connectivity) bypass this and propagate out.
    """
    html = await edgar_client.get_filing_html(cik, meta.accession_no, meta.primary_doc)
    parsed = parser.parse_10k(html, meta.accession_no)

    # Mark 'parsed' so a crash before embedding can resume from here on next run.
    await _set_filing_status(session, meta.accession_no, "parsed")
    await session.commit()

    # Chunk the 3 whitelisted items. Missing items are simply absent.
    all_chunks: list[ChunkDTO] = []
    for item in ITEM_KEYS:
        text = parsed.items.get(item, "").strip()
        if not text:
            continue
        headings = extract_headings(text)  # [(offset, title), ...]
        for chunk in chunk_section(text, item=item):
            # Assign the last heading whose offset <= chunk.char_start
            chunk.section_title = _heading_for_offset(chunk.char_start, headings)
            all_chunks.append(chunk)

    if not all_chunks:
        # Filing had none of the three items — record as embedded (nothing to embed)
        # so we don't keep retrying. Counts as success.
        await _mark_embedded(session, meta.accession_no)
        await session.commit()
        return Decimal("0")

    fiscal_year = meta.period_of_report.year
    embed_inputs = [
        _embedding_text(c.text, company_name, ticker, fiscal_year, c.item) for c in all_chunks
    ]
    vectors, total_tokens = await _embed_with_usage(embed_inputs)
    for chunk, vec in zip(all_chunks, vectors, strict=True):
        chunk.embedding = vec

    # Re-ingestion must be idempotent: drop any prior chunks for this
    # accession before inserting. Otherwise stale rows (e.g. fixture
    # leftovers, or a partial run with a since-revised chunker) collide
    # on (accession_no, item, seq) and ON CONFLICT DO NOTHING silently
    # blocks the real chunks. Filings are immutable, so deletion is
    # safe — we either re-insert the same content or strictly fresher.
    await session.execute(delete(Chunk).where(Chunk.accession_no == meta.accession_no))

    inserted = await _bulk_insert_chunks(
        session,
        cik=cik,
        accession_no=meta.accession_no,
        ticker=ticker,
        company_name=company_name,
        fiscal_year=fiscal_year,
        filing_date=meta.filing_date,
        chunks=all_chunks,
    )
    logger.info(
        "[%s] inserted %d chunks (%d tokens)",
        meta.accession_no,
        inserted,
        total_tokens,
    )

    await _mark_embedded(session, meta.accession_no)
    await session.commit()

    from app.settings import settings

    if settings.EMBEDDING_PROVIDER == "gemini":
        # Free-tier configuration; this is an estimate, not a billing receipt.
        return Decimal("0")
    return (Decimal(total_tokens) / Decimal(1_000_000)) * EMBEDDING_USD_PER_1M


async def _embed_with_usage(texts: list[str]) -> tuple[list[list[float]], int]:
    """embed_texts() returns vectors only — for cost tracking we need token
    counts. Re-implement the batched call here using OpenAI's response.usage.
    """
    from openai import APIConnectionError, AsyncOpenAI
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    from app.data.embedder import DEFAULT_BATCH_SIZE, DEFAULT_MODEL
    from app.settings import settings

    if not texts:
        return [], 0

    if settings.EMBEDDING_PROVIDER == "gemini":
        from app.data.gemini_embeddings import embed_with_usage

        return await embed_with_usage(texts)
    if settings.EMBEDDING_PROVIDER != "openai":
        raise ValueError("EMBEDDING_PROVIDER must be openai or gemini")

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _one(batch: list[str]) -> tuple[list[list[float]], int]:
        resp = await client.embeddings.create(model=DEFAULT_MODEL, input=batch)
        return [d.embedding for d in resp.data], resp.usage.total_tokens

    out_vecs: list[list[float]] = []
    total_tokens = 0
    for start in range(0, len(texts), DEFAULT_BATCH_SIZE):
        batch = texts[start : start + DEFAULT_BATCH_SIZE]
        vecs, tokens = await _one(batch)
        out_vecs.extend(vecs)
        total_tokens += tokens
    return out_vecs, total_tokens


# --- DB helpers -----------------------------------------------------------


async def _upsert_company(session: AsyncSession, *, cik: str, ticker: str, name: str) -> None:
    stmt = (
        pg_insert(Company)
        .values(cik=cik, ticker=ticker, name=name)
        .on_conflict_do_update(index_elements=["cik"], set_={"ticker": ticker, "name": name})
    )
    await session.execute(stmt)


async def _get_filing_status(session: AsyncSession, accession_no: str) -> str | None:
    res = await session.execute(
        select(Filing.ingestion_status).where(Filing.accession_no == accession_no)
    )
    return res.scalar_one_or_none()


async def _insert_filing_pending(session: AsyncSession, *, cik: str, meta: FilingMeta) -> None:
    stmt = (
        pg_insert(Filing)
        .values(
            accession_no=meta.accession_no,
            cik=cik,
            form_type=meta.form_type,
            filing_date=meta.filing_date,
            period_of_report=meta.period_of_report,
            fiscal_year=meta.period_of_report.year,
            primary_doc_url=meta.primary_doc_url,
            ingestion_status="pending",
        )
        .on_conflict_do_nothing(index_elements=["accession_no"])
    )
    await session.execute(stmt)


async def _set_filing_status(session: AsyncSession, accession_no: str, status: str) -> None:
    await session.execute(
        update(Filing).where(Filing.accession_no == accession_no).values(ingestion_status=status)
    )


async def _mark_embedded(session: AsyncSession, accession_no: str) -> None:
    await session.execute(
        update(Filing)
        .where(Filing.accession_no == accession_no)
        .values(
            ingestion_status="embedded",
            ingested_at=datetime.now(UTC),
        )
    )


async def _bulk_insert_chunks(
    session: AsyncSession,
    *,
    cik: str,
    accession_no: str,
    ticker: str,
    company_name: str,
    fiscal_year: int,
    filing_date: date,
    chunks: list[ChunkDTO],
) -> int:
    if not chunks:
        return 0
    rows = [
        {
            "accession_no": accession_no,
            "cik": cik,
            "ticker": ticker,
            "company_name": company_name,
            "fiscal_year": fiscal_year,
            "filing_date": filing_date,
            "item": c.item,
            "seq": c.seq,
            "text": c.text,
            "token_count": c.token_count,
            "embedding": c.embedding,
            "char_start": c.char_start,
            "char_end": c.char_end,
            "section_title": c.section_title,
        }
        for c in chunks
    ]
    stmt = (
        pg_insert(Chunk)
        .values(rows)
        .on_conflict_do_nothing(index_elements=["accession_no", "item", "seq"])
    )
    res = await session.execute(stmt)
    return res.rowcount or 0


async def _insert_xbrl_facts(session: AsyncSession, *, cik: str, facts_json: dict) -> int:
    facts = xbrl.extract_facts(facts_json, cik=cik, years_back=DEFAULT_YEARS_BACK)
    if not facts:
        return 0

    # Skip facts whose accession_no isn't in our filings table — keeps the FK
    # happy when XBRL references filings we didn't ingest (older filings,
    # non-10-K forms occasionally appear in company_facts).
    accession_nos = await _known_accession_nos(session, cik)
    rows = [
        {
            "cik": f.cik,
            "concept": f.concept_key,  # store friendly key; tag is derivable from whitelist
            "unit": f.unit,
            "value": f.value,
            "period_end": f.period_end,
            "fiscal_year": f.fiscal_year,
            "fiscal_period": f.fiscal_period,
            "accession_no": f.accession_no,
            "form": f.form,
        }
        for f in facts
        if f.accession_no in accession_nos
    ]
    if not rows:
        return 0

    stmt = (
        pg_insert(XbrlFact)
        .values(rows)
        .on_conflict_do_nothing(index_elements=["cik", "concept", "period_end", "accession_no"])
    )
    res = await session.execute(stmt)
    inserted = res.rowcount or 0
    logger.info(
        "[%s] xbrl: extracted=%d kept=%d inserted=%d",
        cik,
        len(facts),
        len(rows),
        inserted,
    )
    return inserted


async def _known_accession_nos(session: AsyncSession, cik: str) -> set[str]:
    res = await session.execute(select(Filing.accession_no).where(Filing.cik == cik))
    return {row[0] for row in res}


# --- Concurrent runner & CLI ---------------------------------------------


async def run_ingestion(
    tickers: list[str],
    concurrency: int = DEFAULT_CONCURRENCY,
    retry_failed: bool = False,
) -> list[IngestionResult]:
    sem = asyncio.Semaphore(concurrency)

    async def one(t: str) -> IngestionResult:
        async with sem:
            try:
                return await ingest_company(t, retry_failed=retry_failed)
            except (EdgarRateLimited, EdgarServerError, RateLimitError, APIError):
                # Systemic — re-raise so the caller can stop the run.
                raise
            except Exception as e:  # noqa: BLE001
                logger.exception("[%s] unexpected error", t)
                return IngestionResult(ticker=t, status="failed", error=str(e))

    # return_exceptions=True so a systemic raise in one task does not cancel
    # siblings mid-transaction; we re-raise after all in-flight work settles.
    raw = await asyncio.gather(*[one(t) for t in tickers], return_exceptions=True)
    out: list[IngestionResult] = []
    systemic: BaseException | None = None
    for r in raw:
        if isinstance(r, BaseException):
            systemic = systemic or r
            continue
        out.append(r)
    if systemic is not None:
        raise systemic
    return out


def _read_tickers_csv(path: Path) -> list[str]:
    out: list[str] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = (row.get("ticker") or "").strip().upper()
            if t:
                out.append(t)
    return out


def _print_summary(results: list[IngestionResult], elapsed_s: float) -> None:
    success = sum(1 for r in results if r.status == "success")
    partial = sum(1 for r in results if r.status == "partial")
    failed = sum(1 for r in results if r.status == "failed")
    skipped = sum(1 for r in results if r.status == "skipped")
    total_filings = sum(r.filings_ingested for r in results)
    total_failed_filings = sum(r.filings_failed for r in results)
    total_facts = sum(r.xbrl_facts_inserted for r in results)
    total_cost = sum((r.embedding_cost_usd for r in results), Decimal("0"))

    print("=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Companies:       {len(results)} total")
    print(f"  success:       {success}")
    print(f"  partial:       {partial}")
    print(f"  failed:        {failed}")
    print(f"  skipped:       {skipped}")
    print(f"Filings:         {total_filings} ingested, {total_failed_filings} failed")
    print(f"XBRL facts:      {total_facts} inserted")
    print(f"Embedding cost:  ${total_cost:.6f}")
    print(f"Elapsed:         {elapsed_s:.1f}s")
    print("=" * 60)
    for r in results:
        if r.status in ("failed", "partial"):
            print(f"  {r.ticker:<6} {r.status:<8} {r.error or ''}")


async def main(
    tickers_csv: Path,
    concurrency: int = DEFAULT_CONCURRENCY,
    retry_failed: bool = False,
) -> None:
    tickers = _read_tickers_csv(tickers_csv)
    if not tickers:
        raise SystemExit(f"no tickers in {tickers_csv}")
    logger.info("ingesting %d tickers (concurrency=%d)", len(tickers), concurrency)

    t0 = time.monotonic()
    results = await run_ingestion(tickers, concurrency=concurrency, retry_failed=retry_failed)
    _print_summary(results, time.monotonic() - t0)
