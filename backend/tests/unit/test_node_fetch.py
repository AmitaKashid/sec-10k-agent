"""Fetch node — DB mocked. Verifies ticker→cik resolve, year filter,
and refusal triggers."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
from types import SimpleNamespace

import pytest

from app.agent.nodes import fetch as fetch_module
from app.agent.state import AgentState


class _FakeScalars:
    def __init__(self, values):
        self._values = values

    def all(self):
        return list(self._values)


class _FakeResult:
    """Test result supporting `.scalar_one_or_none()` (single-row queries)
    and `.scalars().all()` (multi-row column queries)."""

    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        # For multi-row results, mimic SQLAlchemy by raising if used wrongly;
        # tests only use this method on single-row queries.
        if isinstance(self._value, list):
            return self._value[0] if self._value else None
        return self._value

    def scalars(self):
        # Multi-row queries pass a list as the canned value.
        if isinstance(self._value, list):
            return _FakeScalars(self._value)
        return _FakeScalars([] if self._value is None else [self._value])


class _FakeSession:
    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    async def execute(self, stmt):
        self.calls += 1
        return _FakeResult(self._results.pop(0) if self._results else None)


def _patch_factory(monkeypatch, results):
    @asynccontextmanager
    async def factory():
        yield _FakeSession(results)

    monkeypatch.setattr(fetch_module, "async_session_factory", factory)


@pytest.mark.asyncio
async def test_fetch_unknown_ticker_refuses(monkeypatch):
    _patch_factory(monkeypatch, [None])  # cik lookup misses
    state = AgentState(
        question="q",
        request_id="r",
        question_type="A_numerical",
        ticker="XYZ",
    )
    out = await fetch_module.fetch(state)
    assert out["question_type"] == "REFUSE"
    assert "XYZ" in out["refusal_reason"]


@pytest.mark.asyncio
async def test_fetch_missing_year_refuses(monkeypatch):
    # Three queries: cik lookup, filing-by-year (miss), available-years.
    _patch_factory(monkeypatch, ["0000320193", None, [2024, 2023, 2022]])
    state = AgentState(
        question="q",
        request_id="r",
        question_type="A_numerical",
        ticker="AAPL",
        fiscal_year=2010,
    )
    out = await fetch_module.fetch(state)
    assert out["question_type"] == "REFUSE"
    assert out["refusal_code"] == "MISSING_FILING"
    assert "2010" in out["refusal_reason"]
    # Composer-facing context: the years we DO have, so it can suggest one.
    assert out["available_fiscal_years"] == [2024, 2023, 2022]


@pytest.mark.asyncio
async def test_fetch_happy_path(monkeypatch):
    fake_filing = SimpleNamespace(
        accession_no="0000320193-24-000123",
        fiscal_year=2024,
        period_of_report=date(2024, 9, 28),
        filing_date=date(2024, 11, 1),
    )
    _patch_factory(monkeypatch, ["0000320193", fake_filing])
    state = AgentState(
        question="q",
        request_id="r",
        question_type="A_numerical",
        ticker="AAPL",
        fiscal_year=2024,
    )
    out = await fetch_module.fetch(state)
    assert out["cik"] == "0000320193"
    assert out["filing"].fiscal_year == 2024
    assert out["fiscal_year"] == 2024
    assert "filing" in out


@pytest.mark.asyncio
async def test_fetch_backfills_fiscal_year_when_null(monkeypatch):
    """When plan didn't specify a year, fetch picks the latest and writes it back."""
    fake_filing = SimpleNamespace(
        accession_no="acc",
        fiscal_year=2025,
        period_of_report=date(2025, 9, 28),
        filing_date=date(2025, 11, 1),
    )
    _patch_factory(monkeypatch, ["0000320193", fake_filing])
    state = AgentState(
        question="q",
        request_id="r",
        question_type="A_numerical",
        ticker="AAPL",
        fiscal_year=None,
    )
    out = await fetch_module.fetch(state)
    assert out["fiscal_year"] == 2025  # backfilled


@pytest.mark.asyncio
async def test_fetch_passes_through_on_refuse(monkeypatch):
    state = AgentState(
        question="q",
        request_id="r",
        question_type="REFUSE",
    )
    out = await fetch_module.fetch(state)
    assert out == {}
