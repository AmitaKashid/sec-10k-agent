"""Retrieve node — branches on question_type, calls get_xbrl_fact or
vector_search."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
from decimal import Decimal

import pytest

from app.agent.nodes import retrieve as retrieve_module
from app.agent.state import AgentState, ChunkHit, FilingRef, XbrlFact


@asynccontextmanager
async def _noop_session():
    yield None


@pytest.fixture(autouse=True)
def _patch_factory(monkeypatch):
    monkeypatch.setattr(retrieve_module, "async_session_factory", _noop_session)
    async def one_query(question):
        return [question]
    monkeypatch.setattr(retrieve_module, "_decompose_question", one_query)


def _xbrl_fact() -> XbrlFact:
    return XbrlFact(
        cik="0000320193",
        concept="revenue",
        concept_key="revenue",
        unit="USD",
        value=Decimal("391035000000"),
        period_end=date(2024, 9, 28),
        fiscal_year=2024,
        fiscal_period="FY",
        accession_no="acc",
        form="10-K",
    )


def _chunk(i: int) -> ChunkHit:
    return ChunkHit(
        chunk_id=i,
        accession_no="acc",
        item="1A",
        seq=i,
        text=f"chunk {i}",
        score=0.1 * i,
        char_start=0,
        char_end=10,
    )


@pytest.mark.asyncio
async def test_retrieve_a_calls_get_xbrl_fact(monkeypatch):
    fact = _xbrl_fact()

    async def fake(**kw):
        return fact

    monkeypatch.setattr(retrieve_module, "get_xbrl_fact", fake)
    state = AgentState(
        question="q",
        request_id="r",
        question_type="A_numerical",
        ticker="AAPL",
        cik="0000320193",
        concept_key="revenue",
        fiscal_year=2024,
    )
    out = await retrieve_module.retrieve(state)
    assert out["xbrl_fact"] is fact
    assert out["trace_events"][-1]["path"] == "A"


@pytest.mark.asyncio
async def test_retrieve_a_refuses_when_no_fact(monkeypatch):
    async def fake(**kw):
        return None

    monkeypatch.setattr(retrieve_module, "get_xbrl_fact", fake)
    state = AgentState(
        question="q",
        request_id="r",
        question_type="A_numerical",
        ticker="AAPL",
        cik="0000320193",
        concept_key="rnd",
        fiscal_year=2024,
    )
    out = await retrieve_module.retrieve(state)
    assert out["question_type"] == "REFUSE"
    assert "rnd" in out["refusal_reason"]


@pytest.mark.asyncio
async def test_retrieve_b_calls_vector_search(monkeypatch):
    chunks = [_chunk(i) for i in range(5)]

    async def fake(**kw):
        return chunks

    monkeypatch.setattr(retrieve_module, "vector_search", fake)
    state = AgentState(
        question="q",
        request_id="r",
        question_type="B_qualitative",
        ticker="AAPL",
        cik="0000320193",
        item="1A",
    )
    out = await retrieve_module.retrieve(state)
    # The current validator receives a bounded set of four merged hits.
    assert out["chunks"] == chunks[:4]
    assert out["trace_events"][-1]["k"] == 4


@pytest.mark.asyncio
async def test_retrieve_b_refuses_when_no_hits(monkeypatch):
    async def fake(**kw):
        return []

    monkeypatch.setattr(retrieve_module, "vector_search", fake)
    state = AgentState(
        question="q",
        request_id="r",
        question_type="B_qualitative",
        ticker="AAPL",
        cik="0000320193",
        item="1A",
    )
    out = await retrieve_module.retrieve(state)
    assert out["question_type"] == "REFUSE"


@pytest.mark.asyncio
async def test_retrieve_passes_through_refuse():
    state = AgentState(question="q", request_id="r", question_type="REFUSE")
    assert await retrieve_module.retrieve(state) == {}


@pytest.mark.asyncio
async def test_retrieve_numerical_uses_requested_company_concept_and_year(monkeypatch):
    calls = []
    async def fact_lookup(**kwargs):
        calls.append(kwargs)
        return _xbrl_fact()
    monkeypatch.setattr(retrieve_module, "get_xbrl_fact", fact_lookup)
    state = AgentState(question="and 2024?", request_id="r", question_type="A_numerical",
                       ticker="AAPL", cik="0000320193", concept_key="revenue", fiscal_year=2024)
    await retrieve_module.retrieve(state)
    assert len(calls) == 1
    assert calls[0]["cik"] == "0000320193"
    assert calls[0]["concept_key"] == "revenue"
    assert calls[0]["fiscal_year"] == 2024


@pytest.mark.asyncio
@pytest.mark.parametrize("year,expected_accession", [(2024, "ACC-2024"), (None, None)])
async def test_retrieve_uses_followup_rewrite_and_filing_filter(monkeypatch, year, expected_accession):
    calls = []
    async def search(**kwargs):
        calls.append(kwargs)
        return [_chunk(1)]
    monkeypatch.setattr(retrieve_module, "vector_search", search)
    state = AgentState(question="and the risks?", request_id="r", question_type="B_qualitative",
                       ticker="AAPL", cik="0000320193", item="1A", fiscal_year=year,
                       rewritten_question="What were AAPL supply-chain risks in 2024?",
                       filing=FilingRef(accession_no="ACC-2024", fiscal_year=2024,
                                        period_of_report=date(2024, 9, 28), filing_date=date(2024, 11, 1)))
    await retrieve_module.retrieve(state)
    assert len(calls) == 1
    assert calls[0]["question"] == state.rewritten_question
    assert calls[0]["accession_no"] == expected_accession
    assert calls[0]["cik"] == state.cik
    assert calls[0]["item"] == "1A"


@pytest.mark.asyncio
async def test_retrieve_merges_queries_without_duplicate_chunks(monkeypatch):
    async def queries(question):
        return [question, "supply chain", "liquidity"]
    async def search(**kwargs):
        return {"risks": [_chunk(1), _chunk(2)],
                "supply chain": [_chunk(1), _chunk(3)],
                "liquidity": [_chunk(1), _chunk(4)]}[kwargs["question"]]
    monkeypatch.setattr(retrieve_module, "_decompose_question", queries)
    monkeypatch.setattr(retrieve_module, "vector_search", search)
    state = AgentState(question="risks", request_id="r", question_type="B_qualitative",
                       ticker="AAPL", cik="0000320193", item="1A")
    result = await retrieve_module.retrieve(state)
    ids = [chunk.chunk_id for chunk in result["chunks"]]
    assert ids[0] == 1
    assert len(ids) == len(set(ids)) == 4
