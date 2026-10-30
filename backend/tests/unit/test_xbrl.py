"""Unit tests for app.data.xbrl (LLD 4 test plan).

Synthetic EDGAR-shaped JSON fixtures — deterministic, no live calls.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from app.data.xbrl import CONCEPT_WHITELIST, XbrlFact, extract_facts, get_fact


def _row(
    *,
    val: float | int | str,
    end: str,
    fy: int,
    fp: str = "FY",
    form: str = "10-K",
    accn: str = "0000320193-24-000123",
) -> dict[str, Any]:
    return {"val": val, "end": end, "fy": fy, "fp": fp, "form": form, "accn": accn}


def _facts_doc(us_gaap: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"facts": {"us-gaap": us_gaap}}


def _all_concepts_fixture() -> dict[str, Any]:
    """Apple-shaped: every whitelist concept populated with 3 FY rows."""
    us_gaap: dict[str, Any] = {}
    for concept_key, tags in CONCEPT_WHITELIST.items():
        tag = tags[0]  # populate the preferred tag only
        unit = _unit_for(concept_key)
        rows = [_row(val=100 + i, end=f"202{i + 2}-09-30", fy=2022 + i) for i in range(3)]
        us_gaap[tag] = {"units": {unit: rows}}
    return _facts_doc(us_gaap)


def _unit_for(concept_key: str) -> str:
    if concept_key == "eps_diluted":
        return "USD/shares"
    if concept_key == "shares_out":
        return "shares"
    return "USD"


def test_apple_extracts_all_10_concepts():
    facts = extract_facts(_all_concepts_fixture(), cik="0000320193", years_back=3)
    found = {f.concept_key for f in facts}
    assert found == set(CONCEPT_WHITELIST.keys())
    for ck in CONCEPT_WHITELIST:
        assert sum(1 for f in facts if f.concept_key == ck) >= 1


def test_only_fy_periods_kept():
    doc = _facts_doc(
        {
            "Revenues": {
                "units": {
                    "USD": [
                        _row(val=1, end="2024-09-30", fy=2024, fp="FY"),
                        _row(val=2, end="2024-06-30", fy=2024, fp="Q3"),
                        _row(val=3, end="2024-03-31", fy=2024, fp="Q2"),
                    ]
                }
            }
        }
    )
    facts = extract_facts(doc, cik="x", years_back=3)
    assert len(facts) == 1
    assert facts[0].fiscal_period == "FY"


def test_only_10k_forms_kept():
    doc = _facts_doc(
        {
            "Revenues": {
                "units": {
                    "USD": [
                        _row(val=1, end="2024-09-30", fy=2024, form="10-K"),
                        _row(val=2, end="2024-09-30", fy=2024, form="10-K/A"),
                        _row(val=3, end="2024-09-30", fy=2024, form="20-F"),
                    ]
                }
            }
        }
    )
    facts = extract_facts(doc, cik="x", years_back=3)
    assert len(facts) == 1
    assert facts[0].form == "10-K"


def test_years_back_filter():
    rows = [_row(val=i, end=f"20{20 + i}-09-30", fy=2020 + i) for i in range(5)]
    doc = _facts_doc({"Revenues": {"units": {"USD": rows}}})
    facts = extract_facts(doc, cik="x", years_back=3)
    assert len(facts) == 3
    years = sorted(f.fiscal_year for f in facts)
    assert years == [2022, 2023, 2024]


def test_multi_tag_first_in_order_wins():
    doc = _facts_doc(
        {
            "Revenues": {"units": {"USD": [_row(val=100, end="2024-09-30", fy=2024)]}},
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "units": {"USD": [_row(val=999, end="2024-09-30", fy=2024)]}
            },
        }
    )
    facts = [f for f in extract_facts(doc, cik="x", years_back=3) if f.concept_key == "revenue"]
    assert len(facts) == 1
    assert facts[0].concept == "Revenues"
    assert facts[0].value == Decimal("100")


def test_multi_tag_falls_back():
    doc = _facts_doc(
        {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "units": {"USD": [_row(val=42, end="2024-09-30", fy=2024)]}
            }
        }
    )
    facts = [f for f in extract_facts(doc, cik="x", years_back=3) if f.concept_key == "revenue"]
    assert len(facts) == 1
    assert facts[0].concept == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert facts[0].value == Decimal("42")


def test_concept_absent_no_rows_no_exception():
    # Berkshire-style: no R&D tag at all
    doc = _facts_doc(
        {
            "Revenues": {"units": {"USD": [_row(val=1, end="2024-12-31", fy=2024)]}},
        }
    )
    facts = extract_facts(doc, cik="0001067983", years_back=3)
    rnd = [f for f in facts if f.concept_key == "rnd"]
    assert rnd == []


def test_decimal_precision_preserved():
    big = "383285000000"
    doc = _facts_doc({"Revenues": {"units": {"USD": [_row(val=big, end="2024-09-30", fy=2024)]}}})
    facts = extract_facts(doc, cik="x", years_back=3)
    assert facts[0].value == Decimal(big)


def test_negative_value_passed_through():
    doc = _facts_doc(
        {"NetIncomeLoss": {"units": {"USD": [_row(val=-2_000_000_000, end="2024-12-31", fy=2024)]}}}
    )
    facts = extract_facts(doc, cik="x", years_back=3)
    assert len(facts) == 1
    assert facts[0].value == Decimal("-2000000000")


def test_ifrs_namespace_ignored():
    doc = {
        "facts": {
            "us-gaap": {"Revenues": {"units": {"USD": [_row(val=100, end="2024-12-31", fy=2024)]}}},
            "ifrs-full": {
                "Revenue": {"units": {"USD": [_row(val=999, end="2024-12-31", fy=2024)]}}
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {"shares": [_row(val=1_000_000, end="2024-12-31", fy=2024)]}
                }
            },
        }
    }
    facts = extract_facts(doc, cik="x", years_back=3)
    # only the us-gaap Revenues row should appear
    assert len(facts) == 1
    assert facts[0].concept == "Revenues"
    assert facts[0].value == Decimal("100")


@pytest.mark.asyncio
async def test_get_fact_is_stub():
    with pytest.raises(NotImplementedError):
        await get_fact("0000320193", "revenue", 2024)


def test_xbrlfact_model_shape():
    f = XbrlFact(
        cik="0000320193",
        concept="Revenues",
        concept_key="revenue",
        unit="USD",
        value=Decimal("383285000000"),
        period_end="2024-09-28",  # type: ignore[arg-type]
        fiscal_year=2024,
        fiscal_period="FY",
        accession_no="0000320193-24-000123",
        form="10-K",
    )
    assert f.value == Decimal("383285000000")
    assert f.fiscal_period == "FY"
