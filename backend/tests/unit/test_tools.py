"""Tools layer — vector_search SQL shape and get_xbrl_fact mapping.

These hit the *interface* between agent and DB. Real SQL execution is
covered by integration tests; here we mock the session and assert the
right query is built.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.agent import tools
from app.agent.state import ChunkHit
from app.data.xbrl import XbrlFact as XbrlFactDTO


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self, result_rows):
        self._rows = result_rows
        self.last_stmt = None
        self.last_params = None

    async def execute(self, stmt, params=None):
        self.last_stmt = stmt
        self.last_params = params
        return _FakeResult(self._rows)


@pytest.mark.asyncio
async def test_get_xbrl_fact_returns_dto(monkeypatch):
    fake_row = SimpleNamespace(
        cik="0000320193",
        concept="revenue",
        unit="USD",
        value=Decimal("391035000000"),
        period_end=date(2024, 9, 28),
        fiscal_year=2024,
        accession_no="0000320193-24-000123",
        form="10-K",
    )
    session = _FakeSession([fake_row])

    fact = await tools.get_xbrl_fact("0000320193", "revenue", 2024, session)
    assert isinstance(fact, XbrlFactDTO)
    assert fact.value == Decimal("391035000000")
    assert fact.concept_key == "revenue"
    assert fact.fiscal_year == 2024


@pytest.mark.asyncio
async def test_get_xbrl_fact_returns_none_when_absent():
    session = _FakeSession([])
    fact = await tools.get_xbrl_fact("0000000000", "revenue", 2024, session)
    assert fact is None


@pytest.mark.asyncio
async def test_get_xbrl_fact_latest_when_year_is_none():
    """fiscal_year=None must NOT inject a year filter — caller wants the
    most recent FY available. We just check the call doesn't blow up
    and a row comes back."""
    fake_row = SimpleNamespace(
        cik="0000320193",
        concept="revenue",
        unit="USD",
        value=Decimal("383285000000"),
        period_end=date(2023, 9, 30),
        fiscal_year=2023,
        accession_no="0000320193-23-000106",
        form="10-K",
    )
    session = _FakeSession([fake_row])
    fact = await tools.get_xbrl_fact("0000320193", "revenue", None, session)
    assert fact is not None
    assert fact.fiscal_year == 2023


@pytest.mark.asyncio
async def test_vector_search_passes_cik_and_item_filter(monkeypatch):
    """The cik+item pre-filter is what makes citations section-correct."""

    async def fake_embed_texts(texts):
        return [[0.1] * 1536]

    monkeypatch.setattr(tools, "embed_texts", fake_embed_texts)

    fake_rows = [
        SimpleNamespace(
            id=1,
            accession_no="acc1",
            item="1A",
            seq=0,
            text="t1",
            char_start=0,
            char_end=2,
            score=0.1,
            ticker=None,
            company_name=None,
            cik=None,
            fiscal_year=None,
            filing_date=None,
            section_title=None,
        ),
        SimpleNamespace(
            id=2,
            accession_no="acc1",
            item="1A",
            seq=1,
            text="t2",
            char_start=2,
            char_end=4,
            score=0.2,
            ticker=None,
            company_name=None,
            cik=None,
            fiscal_year=None,
            filing_date=None,
            section_title=None,
        ),
    ]
    session = _FakeSession(fake_rows)

    hits = await tools.vector_search(
        cik="0000320193", item="1A", question="risk?", session=session, k=5
    )
    assert all(isinstance(h, ChunkHit) for h in hits)
    assert len(hits) == 2
    assert hits[0].score == 0.1
    # Verify SQL params include the cik + item + k
    assert session.last_params["cik"] == "0000320193"
    assert session.last_params["item"] == "1A"
    assert session.last_params["k"] == 5


@pytest.mark.asyncio
async def test_vector_search_empty_question_returns_empty():
    session = _FakeSession([])
    hits = await tools.vector_search("c", "1A", "   ", session)
    assert hits == []


@pytest.mark.asyncio
async def test_vector_search_embeds_question_once(monkeypatch):
    """k>1 results must come from a single embedding call, not k of them."""
    call_count = 0

    async def fake_embed_texts(texts):
        nonlocal call_count
        call_count += 1
        assert len(texts) == 1
        return [[0.5] * 1536]

    monkeypatch.setattr(tools, "embed_texts", fake_embed_texts)
    session = _FakeSession([])
    await tools.vector_search("c", "1A", "x", session, k=5)
    assert call_count == 1
