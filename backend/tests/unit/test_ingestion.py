"""Unit tests for ingestion orchestration. All I/O mocked.

DB helpers in ingestion are stubbed out per-test so we exercise the
orchestration logic (status gating, error routing, cost summing,
concurrency cap) without standing up Postgres. The actual SQL is
covered by tests/integration/.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import date
from decimal import Decimal

import pytest

from app.data import ingestion
from app.settings import settings
from app.data.edgar_client import EdgarRateLimited, FilingMeta
from app.data.parser import ParsedFiling, ParserAllTiersFailed

# --- Fixtures: stub out the data layer + DB session ----------------------


def _filing_meta(accession: str, fy: int) -> FilingMeta:
    return FilingMeta(
        accession_no=accession,
        form_type="10-K",
        filing_date=date(fy, 11, 1),
        period_of_report=date(fy, 9, 30),
        primary_doc=f"{accession}.htm",
        primary_doc_url=f"https://example.com/{accession}.htm",
    )


def _parsed_filing() -> ParsedFiling:
    return ParsedFiling(
        items={
            "1A": "Risk factor body. " * 50,
            "7": "MD&A body. " * 50,
            "8": "Financial body. " * 50,
        },
        tier="edgartools",
        char_counts={"1A": 1000, "7": 1000, "8": 1000},
    )


@pytest.fixture
def stub_session(monkeypatch):
    """Replace async_session_factory with a no-op async context manager.

    DB helpers are individually monkeypatched per-test; the session
    object itself is just a sentinel that gets passed through.
    """
    sentinel = object()

    @asynccontextmanager
    async def fake_factory():
        class _S:
            async def commit(self):
                pass

            async def rollback(self):
                pass

            async def close(self):
                pass

            async def execute(self, *args, **kwargs):
                # No-op for the idempotency `delete(Chunk)...` call in
                # _process_filing; bulk inserts go through monkeypatched
                # helpers, not through this method.
                return None

        yield _S()

    monkeypatch.setattr(ingestion, "async_session_factory", fake_factory)
    return sentinel


@pytest.fixture
def state():
    """Per-test mutable state to drive filing-status responses + capture writes."""
    return {
        "statuses": {},  # accession_no -> status
        "chunks_inserted": [],  # list of accession_no
        "facts_inserted": 0,
        "embed_token_count": 1000,
        "filings_to_return": [
            _filing_meta("0000-24-001", 2024),
            _filing_meta("0000-23-001", 2023),
            _filing_meta("0000-22-001", 2022),
        ],
        "facts_json": {"entityName": "Test Co", "facts": {"us-gaap": {}}},
        "cik": "0000000123",
    }


@pytest.fixture
def patched(monkeypatch, state):
    """Monkeypatch every external call ingest_company makes."""
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "openai")

    async def fake_get_cik(ticker):
        return state["cik"] if ticker.upper() != "BAD" else None

    async def fake_get_submissions(cik, form_type="10-K"):
        return state["filings_to_return"]

    async def fake_get_company_facts(cik):
        return state["facts_json"]

    async def fake_get_filing_html(cik, accession_no, primary_doc):
        return b"<html></html>"

    def fake_parse_10k(html, accession_no):
        if state.get("parser_fails_for") == accession_no:
            raise ParserAllTiersFailed(["all tiers failed"])
        return _parsed_filing()

    async def fake_embed_with_usage(texts):
        vecs = [[0.0] * 1536 for _ in texts]
        return vecs, state["embed_token_count"]

    def fake_extract_facts(facts_json, cik, years_back=3):
        return []

    # DB helpers
    async def fake_upsert_company(session, *, cik, ticker, name):
        state["company"] = (cik, ticker, name)

    async def fake_get_filing_status(session, accession_no):
        return state["statuses"].get(accession_no)

    async def fake_insert_filing_pending(session, *, cik, meta):
        state["statuses"][meta.accession_no] = "pending"

    async def fake_set_filing_status(session, accession_no, status):
        state["statuses"][accession_no] = status

    async def fake_mark_embedded(session, accession_no):
        state["statuses"][accession_no] = "embedded"

    async def fake_bulk_insert_chunks(
        session,
        *,
        cik,
        accession_no,
        chunks,
        ticker=None,
        company_name=None,
        fiscal_year=None,
        filing_date=None,
    ):
        state["chunks_inserted"].append(accession_no)
        return len(chunks)

    async def fake_insert_xbrl_facts(session, *, cik, facts_json):
        state["facts_inserted"] += 1
        return 0

    monkeypatch.setattr(ingestion.edgar_client, "get_cik_for_ticker", fake_get_cik)
    monkeypatch.setattr(ingestion.edgar_client, "get_submissions", fake_get_submissions)
    monkeypatch.setattr(ingestion.edgar_client, "get_company_facts", fake_get_company_facts)
    monkeypatch.setattr(ingestion.edgar_client, "get_filing_html", fake_get_filing_html)
    monkeypatch.setattr(ingestion.parser, "parse_10k", fake_parse_10k)
    monkeypatch.setattr(ingestion, "_embed_with_usage", fake_embed_with_usage)
    monkeypatch.setattr(ingestion.xbrl, "extract_facts", fake_extract_facts)

    monkeypatch.setattr(ingestion, "_upsert_company", fake_upsert_company)
    monkeypatch.setattr(ingestion, "_get_filing_status", fake_get_filing_status)
    monkeypatch.setattr(ingestion, "_insert_filing_pending", fake_insert_filing_pending)
    monkeypatch.setattr(ingestion, "_set_filing_status", fake_set_filing_status)
    monkeypatch.setattr(ingestion, "_mark_embedded", fake_mark_embedded)
    monkeypatch.setattr(ingestion, "_bulk_insert_chunks", fake_bulk_insert_chunks)
    monkeypatch.setattr(ingestion, "_insert_xbrl_facts", fake_insert_xbrl_facts)
    return state


# --- Tests ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_company_end_to_end(stub_session, patched):
    res = await ingestion.ingest_company("AAPL")
    assert res.status == "success"
    assert res.filings_ingested == 3
    assert res.filings_failed == 0
    assert len(patched["chunks_inserted"]) == 3
    # 3 filings × 1000 tokens × $0.02/1M = $0.00006
    assert res.embedding_cost_usd == Decimal("0.000060")
    assert all(s == "embedded" for s in patched["statuses"].values())


@pytest.mark.asyncio
async def test_idempotent_rerun_zero_inserts(stub_session, patched):
    # Pre-mark every filing as embedded.
    for f in patched["filings_to_return"]:
        patched["statuses"][f.accession_no] = "embedded"

    res = await ingestion.ingest_company("AAPL")
    assert res.status == "skipped"
    assert res.filings_ingested == 0
    assert res.filings_failed == 0
    assert patched["chunks_inserted"] == []  # no embeds called


@pytest.mark.asyncio
async def test_filing_parse_failure_marks_failed_continues(stub_session, patched):
    patched["parser_fails_for"] = "0000-23-001"
    res = await ingestion.ingest_company("AAPL")
    assert res.status == "partial"
    assert res.filings_ingested == 2
    assert res.filings_failed == 1
    assert patched["statuses"]["0000-23-001"] == "failed"
    assert patched["statuses"]["0000-24-001"] == "embedded"


@pytest.mark.asyncio
async def test_failed_status_skipped_by_default(stub_session, patched):
    patched["statuses"]["0000-22-001"] = "failed"
    res = await ingestion.ingest_company("AAPL", retry_failed=False)
    # 2 succeed (24 + 23), 22 skipped because already failed
    assert res.filings_ingested == 2
    assert res.filings_failed == 0
    assert patched["statuses"]["0000-22-001"] == "failed"  # unchanged


@pytest.mark.asyncio
async def test_retry_failed_flag_resets_and_retries(stub_session, patched):
    patched["statuses"]["0000-22-001"] = "failed"
    res = await ingestion.ingest_company("AAPL", retry_failed=True)
    assert res.filings_ingested == 3
    assert patched["statuses"]["0000-22-001"] == "embedded"


@pytest.mark.asyncio
async def test_unknown_ticker_failed_result_continues(stub_session, patched):
    res = await ingestion.ingest_company("BAD")
    assert res.status == "failed"
    assert "ticker not in EDGAR map" in (res.error or "")
    assert res.filings_ingested == 0
    # No DB writes attempted.
    assert patched["chunks_inserted"] == []


@pytest.mark.asyncio
async def test_systemic_error_propagates(stub_session, patched, monkeypatch):
    async def boom(cik, form_type="10-K"):
        raise EdgarRateLimited("429 after retries")

    monkeypatch.setattr(ingestion.edgar_client, "get_submissions", boom)
    with pytest.raises(EdgarRateLimited):
        await ingestion.ingest_company("AAPL")


@pytest.mark.asyncio
async def test_concurrency_cap_respected(monkeypatch, stub_session):
    in_flight = 0
    peak = 0

    async def fake_ingest(ticker, retry_failed=False):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return ingestion.IngestionResult(ticker=ticker, status="success")

    monkeypatch.setattr(ingestion, "ingest_company", fake_ingest)
    tickers = [f"T{i}" for i in range(20)]
    results = await ingestion.run_ingestion(tickers, concurrency=5)
    assert len(results) == 20
    assert peak <= 5


@pytest.mark.asyncio
async def test_gemini_ingestion_does_not_report_openai_embedding_cost(stub_session, patched, monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "gemini")
    result = await ingestion.ingest_company("AAPL")
    assert result.status == "success"
    assert result.filings_ingested == 3
    assert result.embedding_cost_usd == Decimal("0")
