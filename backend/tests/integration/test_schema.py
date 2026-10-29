"""Integration tests for the Postgres schema (LLD 5 test plan).

Requires a live Postgres with pgvector. Auto-skipped via conftest.py
when the DB is unreachable.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.models import Chunk, Company, Filing, XbrlFact
from tests.integration.conftest import SKIP_REASON, db_reachable

pytestmark = pytest.mark.skipif(not db_reachable(), reason=SKIP_REASON)

EMBEDDING_DIM = 1536


def _vec(value: float) -> list[float]:
    return [value] * EMBEDDING_DIM


async def _seed_company_filing(session: AsyncSession) -> tuple[Company, Filing]:
    company = Company(cik="0000320193", ticker="AAPL", name="Apple Inc.")
    filing = Filing(
        accession_no="0000320193-24-000123",
        cik="0000320193",
        form_type="10-K",
        filing_date=date(2024, 11, 1),
        period_of_report=date(2024, 9, 28),
        fiscal_year=2024,
        primary_doc_url="https://www.sec.gov/example.htm",
        ingestion_status="parsed",
    )
    session.add_all([company, filing])
    await session.flush()
    return company, filing


async def test_alembic_upgrade_head_succeeds(session: AsyncSession):
    """All 6 tables exist and pgvector extension is installed."""
    res = await session.execute(
        text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        )
    )
    tables = {row[0] for row in res}
    expected = {
        "alembic_version",
        "answer_cache",
        "chunks",
        "companies",
        "filings",
        "query_log",
        "xbrl_facts",
    }
    assert expected <= tables, f"missing tables: {expected - tables}"

    ext = await session.execute(text("SELECT extname FROM pg_extension WHERE extname='vector'"))
    assert ext.scalar() == "vector"


async def test_company_filing_relationship_cascade(session: AsyncSession):
    company, filing = await _seed_company_filing(session)
    await session.commit()

    await session.delete(company)
    await session.commit()

    res = await session.execute(select(Filing).where(Filing.accession_no == filing.accession_no))
    assert res.scalar_one_or_none() is None


async def test_filing_chunks_cascade(session: AsyncSession):
    _, filing = await _seed_company_filing(session)
    chunk = Chunk(
        accession_no=filing.accession_no,
        cik=filing.cik,
        item="1A",
        seq=0,
        text="risk factor text",
        token_count=4,
        embedding=_vec(0.1),
        char_start=0,
        char_end=16,
    )
    session.add(chunk)
    await session.commit()

    await session.delete(filing)
    await session.commit()

    res = await session.execute(select(Chunk))
    assert res.scalars().all() == []


async def test_chunk_unique_constraint(session: AsyncSession):
    _, filing = await _seed_company_filing(session)
    session.add(
        Chunk(
            accession_no=filing.accession_no,
            cik=filing.cik,
            item="1A",
            seq=0,
            text="a",
            token_count=1,
            embedding=_vec(0.1),
            char_start=0,
            char_end=1,
        )
    )
    await session.commit()

    session.add(
        Chunk(
            accession_no=filing.accession_no,
            cik=filing.cik,
            item="1A",
            seq=0,
            text="b",
            token_count=1,
            embedding=_vec(0.2),
            char_start=0,
            char_end=1,
        )
    )
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_xbrl_unique_constraint(session: AsyncSession):
    _, filing = await _seed_company_filing(session)
    base = dict(
        cik=filing.cik,
        concept="Revenues",
        unit="USD",
        period_end=date(2024, 9, 28),
        fiscal_year=2024,
        fiscal_period="FY",
        accession_no=filing.accession_no,
        form="10-K",
    )
    session.add(XbrlFact(**base, value=1))
    await session.commit()

    session.add(XbrlFact(**base, value=2))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_vector_cosine_search_returns_relevant(session: AsyncSession):
    _, filing = await _seed_company_filing(session)
    target = _vec(1.0)
    other = [0.0] * EMBEDDING_DIM
    other[0] = 1.0  # cosine-distant from `target`
    session.add_all(
        [
            Chunk(
                accession_no=filing.accession_no,
                cik=filing.cik,
                item="1A",
                seq=0,
                text="match",
                token_count=1,
                embedding=target,
                char_start=0,
                char_end=5,
            ),
            Chunk(
                accession_no=filing.accession_no,
                cik=filing.cik,
                item="1A",
                seq=1,
                text="other",
                token_count=1,
                embedding=other,
                char_start=5,
                char_end=10,
            ),
        ]
    )
    await session.commit()

    res = await session.execute(
        text("SELECT text FROM chunks ORDER BY embedding <=> CAST(:q AS vector) LIMIT 1"),
        {"q": str(target)},
    )
    assert res.scalar() == "match"


async def test_check_constraint_rejects_bad_item(session: AsyncSession):
    _, filing = await _seed_company_filing(session)
    session.add(
        Chunk(
            accession_no=filing.accession_no,
            cik=filing.cik,
            item="99",
            seq=0,
            text="x",
            token_count=1,
            embedding=_vec(0.1),
            char_start=0,
            char_end=1,
        )
    )
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_hnsw_index_used(session: AsyncSession):
    _, filing = await _seed_company_filing(session)
    # Need enough rows for the planner to prefer the index over seq scan.
    session.add_all(
        [
            Chunk(
                accession_no=filing.accession_no,
                cik=filing.cik,
                item="1A",
                seq=i,
                text=f"c{i}",
                token_count=1,
                embedding=_vec(0.001 * i),
                char_start=i,
                char_end=i + 1,
            )
            for i in range(50)
        ]
    )
    await session.commit()
    # Disable seq scan so the planner is forced to consider the index;
    # if the index is missing or unusable this query would fall back differently.
    await session.execute(text("SET LOCAL enable_seqscan = off"))
    res = await session.execute(
        text(
            "EXPLAIN (FORMAT TEXT) SELECT id FROM chunks "
            "ORDER BY embedding <=> CAST(:q AS vector) LIMIT 5"
        ),
        {"q": str(_vec(0.01))},
    )
    plan = "\n".join(row[0] for row in res)
    assert "ix_chunks_embedding_hnsw" in plan, plan
