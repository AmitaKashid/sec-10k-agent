"""Opt-in SEC/Gemini ingestion smoke test on the isolated test database.

Run with scripts/run_tests.py --suite live. Requires a real SEC contact
email and private provider keys. Consumes real SEC/embedding quota.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.data import ingestion
from app.data.models import Chunk, Company, Filing, XbrlFact
from app.settings import settings
from tests.integration.conftest import SKIP_REASON, db_reachable

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not db_reachable(), reason=SKIP_REASON),
]


async def test_aapl_end_to_end(session: AsyncSession, monkeypatch):
    """Single-company seed run + re-run idempotency check.

    The `session` fixture truncates all tables and runs migrations, so this
    is the first and only AAPL run in this DB session.
    """
    if not settings.GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY not set")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "gemini")
    # --- First run: full pipeline -------------------------------------
    result = await ingestion.ingest_company("AAPL")

    assert result.status == "success", f"ingest failed: {result.error}"
    assert result.filings_ingested == 3, f"expected 3 filings, got {result.filings_ingested}"
    assert result.filings_failed == 0
    assert result.embedding_cost_usd == 0  # Free-provider reporting policy

    # --- DB state assertions ------------------------------------------
    companies = (await session.execute(select(Company))).scalars().all()
    assert len(companies) == 1
    assert companies[0].ticker == "AAPL"
    assert companies[0].cik == "0000320193"

    filings = (await session.execute(select(Filing))).scalars().all()
    assert len(filings) == 3
    assert all(f.ingestion_status == "embedded" for f in filings)
    assert all(f.form_type == "10-K" for f in filings)
    assert all(f.ingested_at is not None for f in filings)

    chunks = (await session.execute(select(Chunk))).scalars().all()
    assert len(chunks) > 0
    # Every chunk has a 1536-dim embedding.
    assert all(c.embedding is not None and len(c.embedding) == 1536 for c in chunks)
    # Every chunk's item is one of 1A/7/8.
    assert {c.item for c in chunks} <= {"1A", "7", "8"}

    facts = (await session.execute(select(XbrlFact))).scalars().all()
    assert len(facts) > 0
    found_keys = {f.concept for f in facts}
    assert "revenue" in found_keys


    initial_chunk_count = len(chunks)
    initial_fact_count = len(facts)

    # --- Second run: must be a no-op ----------------------------------
    rerun = await ingestion.ingest_company("AAPL")
    assert rerun.status == "skipped"
    assert rerun.filings_ingested == 0
    assert rerun.filings_failed == 0
    assert rerun.embedding_cost_usd == 0

    # No new rows.
    chunks_after = (await session.execute(select(Chunk))).scalars().all()
    facts_after = (await session.execute(select(XbrlFact))).scalars().all()
    assert len(chunks_after) == initial_chunk_count
    assert len(facts_after) == initial_fact_count
