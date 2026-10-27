"""SEC XBRL company-facts → whitelisted, FY-only structured facts.

See docs/lld.md §LLD 4. Numbers come from this module, never from prose —
this is the structural anti-hallucination layer.

`extract_facts` is the ingestion-time entrypoint. `get_fact` is the
query-time DB lookup; its body lands with LLD 5 (db.py + models).
"""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)

FiscalPeriod = Literal["FY"]

# Whitelist: user-facing key → ordered list of us-gaap tags. First tag with
# any matching rows wins; later tags are fallbacks only.
CONCEPT_WHITELIST: dict[str, list[str]] = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "net_income": ["NetIncomeLoss"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    "total_assets": ["Assets"],
    "total_debt": ["LongTermDebt"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "shares_out": ["CommonStockSharesOutstanding"],
    "opex": ["OperatingExpenses"],
    "gross_profit": ["GrossProfit"],
}


class XbrlFact(BaseModel):
    cik: str
    concept: str
    concept_key: str
    unit: str
    value: Decimal
    period_end: date
    fiscal_year: int
    fiscal_period: FiscalPeriod
    accession_no: str
    form: str


def extract_facts(
    company_facts: dict[str, Any],
    cik: str,
    years_back: int = 3,
) -> list[XbrlFact]:
    """Flatten EDGAR company-facts JSON into whitelisted FY 10-K rows."""
    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        return []

    # Determine the cutoff year from the maximum FY across the whole document
    # so the filter is stable regardless of which concept is read first.
    max_fy = _max_fiscal_year(us_gaap)
    if max_fy is None:
        return []
    min_fy = max_fy - years_back + 1

    out: list[XbrlFact] = []
    for concept_key, tag_order in CONCEPT_WHITELIST.items():
        for tag in tag_order:
            tag_block = us_gaap.get(tag)
            if not tag_block:
                continue
            rows = _extract_tag(
                tag_block=tag_block,
                tag=tag,
                concept_key=concept_key,
                cik=cik,
                min_fy=min_fy,
            )
            if rows:
                out.extend(rows)
                break  # first-in-order wins; ignore later tags
    return out


def _max_fiscal_year(us_gaap: dict[str, Any]) -> int | None:
    best: int | None = None
    for tag_block in us_gaap.values():
        for unit_rows in tag_block.get("units", {}).values():
            for row in unit_rows:
                if row.get("fp") != "FY" or row.get("form") != "10-K":
                    continue
                fy = row.get("fy")
                if isinstance(fy, int) and (best is None or fy > best):
                    best = fy
    return best


def _extract_tag(
    *,
    tag_block: dict[str, Any],
    tag: str,
    concept_key: str,
    cik: str,
    min_fy: int,
) -> list[XbrlFact]:
    rows: list[XbrlFact] = []
    units: dict[str, list[dict[str, Any]]] = tag_block.get("units", {})
    for unit, unit_rows in units.items():
        for row in unit_rows:
            if row.get("fp") != "FY" or row.get("form") != "10-K":
                continue
            fy = row.get("fy")
            if not isinstance(fy, int) or fy < min_fy:
                continue
            try:
                fact = XbrlFact(
                    cik=cik,
                    concept=tag,
                    concept_key=concept_key,
                    unit=unit,
                    value=Decimal(str(row["val"])),
                    period_end=date.fromisoformat(row["end"]),
                    fiscal_year=fy,
                    fiscal_period="FY",
                    accession_no=row["accn"],
                    form=row["form"],
                )
            except (KeyError, ValueError) as exc:
                logger.warning("skipping malformed XBRL row for %s/%s: %s", cik, tag, exc)
                continue
            rows.append(fact)
    return rows


async def get_fact(
    cik: str,
    concept_key: str,
    fiscal_year: int | None,
) -> XbrlFact | None:
    """Query-time DB lookup. Filled in with LLD 5 (db.py + models)."""
    raise NotImplementedError("get_fact lands with LLD 5 — see docs/lld.md")
