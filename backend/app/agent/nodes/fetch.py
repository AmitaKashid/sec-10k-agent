"""Fetch node — resolve ticker→cik and locate the right Filing row.

No LLM. Refuses if ticker not in our 50-company universe or the
requested fiscal year wasn't ingested.
"""

from __future__ import annotations

from sqlalchemy import select

from app.agent.state import AgentState, FilingRef, RefusalCode
from app.data.db import async_session_factory
from app.data.models import Company, Filing


async def fetch(state: AgentState) -> dict:
    if state.question_type == "REFUSE":
        return {}  # plan already refused; pass through

    if not state.ticker:
        return _refuse(state, "Plan failed to identify a ticker.", "INTERNAL_ERROR")

    available_years: list[int] = []
    async with async_session_factory() as session:
        cik = (
            await session.execute(select(Company.cik).where(Company.ticker == state.ticker))
        ).scalar_one_or_none()

        if cik is None:
            return _refuse(
                state,
                f"{state.ticker} isn't in our 50-company universe. "
                "I only cover the tickers in config/tickers.csv.",
                "UNKNOWN_TICKER",
            )

        stmt = (
            select(Filing)
            .where(Filing.cik == cik, Filing.ingestion_status == "embedded")
            .order_by(Filing.fiscal_year.desc())
        )
        if state.fiscal_year is not None:
            stmt = stmt.where(Filing.fiscal_year == state.fiscal_year)
        stmt = stmt.limit(1)

        filing_row = (await session.execute(stmt)).scalar_one_or_none()

        if filing_row is None:
            # Look up the years we DO have for this ticker so the composer
            # can suggest alternatives ("we have 2022/2023/2024 — try one
            # of those"). Cheap indexed query; reuses the open session.
            years_result = await session.execute(
                select(Filing.fiscal_year)
                .where(Filing.cik == cik, Filing.ingestion_status == "embedded")
                .order_by(Filing.fiscal_year.desc())
            )
            available_years = [int(y) for y in years_result.scalars().all()]

    if filing_row is None:
        if state.fiscal_year is not None:
            reason = (
                f"No 10-K filing for {state.ticker} fiscal year {state.fiscal_year} "
                f"is loaded. We have the 3 most recent fiscal years per company."
            )
        else:
            reason = f"No ingested 10-K filings for {state.ticker}."
        return _refuse(state, reason, "MISSING_FILING", available_years=available_years)

    filing = FilingRef(
        accession_no=filing_row.accession_no,
        fiscal_year=filing_row.fiscal_year,
        period_of_report=filing_row.period_of_report,
        filing_date=filing_row.filing_date,
    )
    event = {
        "node": "fetch",
        "cik": cik,
        "fiscal_year": filing.fiscal_year,
        "accession_no": filing.accession_no,
    }
    return {
        "cik": cik,
        "fiscal_year": filing.fiscal_year,  # backfill if plan left it null
        "filing": filing,
        "trace_events": state.trace_events + [event],
    }


def _refuse(
    state: AgentState,
    reason: str,
    code: RefusalCode,
    available_years: list[int] | None = None,
) -> dict:
    delta: dict = {
        "question_type": "REFUSE",
        "refusal_reason": reason,
        "refusal_code": code,
        "trace_events": state.trace_events
        + [{"node": "fetch", "refused": True, "reason": reason, "refusal_code": code}],
    }
    if available_years is not None:
        delta["available_fiscal_years"] = available_years
    return delta
