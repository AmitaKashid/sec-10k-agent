"""Locate node — DB mocked."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date

import pytest

from app.agent.nodes import locate as locate_module
from app.agent.state import AgentState, FilingRef


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _FakeSession:
    def __init__(self, value):
        self._value = value

    async def execute(self, stmt):
        return _FakeResult(self._value)


def _patch(monkeypatch, value):
    @asynccontextmanager
    async def factory():
        yield _FakeSession(value)

    monkeypatch.setattr(locate_module, "async_session_factory", factory)


def _filing() -> FilingRef:
    return FilingRef(
        accession_no="acc",
        fiscal_year=2024,
        period_of_report=date(2024, 9, 28),
        filing_date=date(2024, 11, 1),
    )


@pytest.mark.asyncio
async def test_locate_type_a_skipped():
    state = AgentState(
        question="q",
        request_id="r",
        question_type="A_numerical",
        ticker="AAPL",
        cik="0000320193",
        filing=_filing(),
    )
    out = await locate_module.locate(state)
    assert out["located"] is True
    assert out["trace_events"][-1].get("skipped_for_type_a") is True


@pytest.mark.asyncio
async def test_locate_type_b_passes_when_chunks_present(monkeypatch):
    _patch(monkeypatch, 42)  # any chunk id
    state = AgentState(
        question="q",
        request_id="r",
        question_type="B_qualitative",
        ticker="AAPL",
        cik="0000320193",
        item="1A",
        filing=_filing(),
    )
    out = await locate_module.locate(state)
    assert out["located"] is True


@pytest.mark.asyncio
async def test_locate_type_b_refuses_when_section_missing(monkeypatch):
    _patch(monkeypatch, None)
    state = AgentState(
        question="q",
        request_id="r",
        question_type="B_qualitative",
        ticker="AAPL",
        cik="0000320193",
        item="7",
        fiscal_year=2024,
        filing=_filing(),
    )
    out = await locate_module.locate(state)
    assert out["question_type"] == "REFUSE"
    assert "Item 7" in out["refusal_reason"]


@pytest.mark.asyncio
async def test_locate_passes_through_refuse():
    state = AgentState(question="q", request_id="r", question_type="REFUSE")
    assert await locate_module.locate(state) == {}
