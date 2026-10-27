"""Generate the 30-row golden eval dataset from live SEC EDGAR data.

Fetches real accession numbers + XBRL values for a hand-picked sector-
diverse company set, so `score_citation` and `score_numeric` line up
with what the agent actually surfaces from `app/data/xbrl.py`. Run once
when the dataset needs to be (re)built; output is committed to git.

Usage:
    uv run python evals/scripts/build_golden.py > evals/golden_dataset.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

UA = "10K-Eval-Builder dhiraj.poddar@maindtec.de"
HEADERS = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}

# (ticker, [(concept_key, gaap_tags)]) — tags mirror app/data/xbrl.py CONCEPT_WHITELIST.
CONCEPT_TAGS: dict[str, list[str]] = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "net_income": ["NetIncomeLoss"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    "total_assets": ["Assets"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "gross_profit": ["GrossProfit"],
    "opex": ["OperatingExpenses"],
    "eps_diluted": ["EarningsPerShareDiluted"],
}

# Sector-diverse subset of the 50-company universe. Each picks a few
# concepts to get to ~18 numeric rows. Banks (JPM) skipped for revenue
# because they don't tag Revenues / RevenueFromContract.
NUMERIC_PICKS: list[tuple[str, int, list[str]]] = [
    ("AAPL", 2024, ["revenue", "net_income", "rnd", "eps_diluted"]),
    ("MSFT", 2024, ["revenue", "net_income", "cash"]),
    ("GOOGL", 2024, ["revenue", "rnd", "net_income"]),
    ("NVDA", 2024, ["revenue", "gross_profit"]),
    ("META", 2024, ["revenue", "net_income"]),
    ("JPM", 2024, ["net_income", "total_assets"]),
    ("UNH", 2024, ["revenue"]),
    ("WMT", 2024, ["revenue"]),
    ("XOM", 2024, ["revenue", "net_income"]),
    ("KO", 2024, ["revenue"]),
]

# Prose rows: (ticker, fy, item_section, keywords). Accession resolved
# from submissions feed at build time.
PROSE_PICKS: list[tuple[str, int, str, list[str], str]] = [
    ("AAPL", 2024, "1A", ["supply", "manufactur"], "supply-chain risk factors"),
    ("AAPL", 2024, "7", ["services"], "services segment in management's discussion"),
    ("MSFT", 2024, "1A", ["cybersecurity", "competition"], "cybersecurity and competition risks"),
    ("MSFT", 2024, "7", ["cloud", "azure"], "cloud / Azure performance in MD&A"),
    ("GOOGL", 2024, "1A", ["regulat", "antitrust"], "regulatory and antitrust risks"),
    ("NVDA", 2024, "1A", ["china", "export"], "China export-control risks"),
    ("META", 2024, "1A", ["advertising", "user"], "advertising and user-engagement risks"),
    ("WMT", 2024, "7", ["ecommerce", "comparable"], "e-commerce and comparable sales discussion"),
    ("XOM", 2024, "1A", ["climate", "regulat"], "climate-related risks"),
    ("KO", 2024, "1A", ["consumer", "health"], "consumer health and demand risks"),
    ("UNH", 2024, "1A", ["regulat", "medicare"], "Medicare / regulatory risks"),
]


def fetch_ticker_map(client: httpx.Client) -> dict[str, str]:
    r = client.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS, timeout=30)
    r.raise_for_status()
    raw = r.json()
    out: dict[str, str] = {}
    for v in raw.values():
        out[v["ticker"].upper()] = str(v["cik_str"]).zfill(10)
    return out


def fetch_facts(client: httpx.Client, cik: str) -> dict[str, Any]:
    r = client.get(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
        headers=HEADERS,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def fetch_submissions(client: httpx.Client, cik: str) -> dict[str, Any]:
    r = client.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def find_fy_fact(
    facts: dict[str, Any], gaap_tags: list[str], fy: int
) -> tuple[str, str, str] | None:
    """Return (value_str, accession, period_end) for the FY-fp 10-K fact, or None."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in gaap_tags:
        block = us_gaap.get(tag)
        if not block:
            continue
        for unit_rows in block.get("units", {}).values():
            for row in unit_rows:
                if (
                    row.get("fp") == "FY"
                    and row.get("form") == "10-K"
                    and row.get("fy") == fy
                ):
                    return str(row["val"]), row["accn"], row["end"]
    return None


def find_fy_accession(subs: dict[str, Any], fy: int) -> str | None:
    """Return accession of the 10-K whose period_of_report year == fy."""
    recent = subs.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accns = recent.get("accessionNumber", [])
    periods = recent.get("reportDate", [])
    for form, accn, per in zip(forms, accns, periods, strict=False):
        if form != "10-K":
            continue
        if per and per.startswith(str(fy)):
            return accn
    return None


def _q_for_concept(ticker: str, concept_key: str, fy: int) -> str:
    label = {
        "revenue": "total revenue",
        "net_income": "net income",
        "rnd": "research and development expense",
        "total_assets": "total assets",
        "cash": "cash and cash equivalents",
        "gross_profit": "gross profit",
        "opex": "total operating expenses",
        "eps_diluted": "diluted earnings per share",
    }[concept_key]
    return f"What was {ticker}'s {label} in fiscal year {fy}?"


def main() -> int:
    client = httpx.Client()
    tmap = fetch_ticker_map(client)

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    # --- Numeric / XBRL rows ---
    for ticker, fy, concepts in NUMERIC_PICKS:
        cik = tmap.get(ticker)
        if not cik:
            skipped.append(f"{ticker}: no CIK")
            continue
        try:
            facts = fetch_facts(client, cik)
        except httpx.HTTPError as e:
            skipped.append(f"{ticker}: facts fetch failed: {e}")
            continue
        time.sleep(0.2)  # courtesy below 5 req/s
        for ck in concepts:
            tags = CONCEPT_TAGS[ck]
            hit = find_fy_fact(facts, tags, fy)
            if not hit:
                skipped.append(f"{ticker} {ck} FY{fy}: no FY fact")
                continue
            value, accn, _period = hit
            rows.append(
                {
                    "id": f"g_{ticker.lower()}_{ck}_fy{fy}",
                    "question": _q_for_concept(ticker, ck, fy),
                    "pipeline": "xbrl",
                    "ticker": ticker,
                    "fiscal_year": fy,
                    "accession": accn,
                    "concept_key": ck,
                    "value": value,
                    "tolerance_pct": 0.5,
                    "tags": ["numeric", "single-company"],
                }
            )

    # --- Prose rows ---
    submissions_cache: dict[str, dict[str, Any]] = {}
    for ticker, fy, item, kws, descr in PROSE_PICKS:
        cik = tmap.get(ticker)
        if not cik:
            skipped.append(f"{ticker}: no CIK")
            continue
        if cik not in submissions_cache:
            try:
                submissions_cache[cik] = fetch_submissions(client, cik)
            except httpx.HTTPError as e:
                skipped.append(f"{ticker}: submissions fetch failed: {e}")
                continue
            time.sleep(0.2)
        accn = find_fy_accession(submissions_cache[cik], fy)
        if not accn:
            skipped.append(f"{ticker} prose FY{fy}: no 10-K accession")
            continue
        slug = descr.split()[0].replace("/", "-").lower().rstrip(",.")
        rows.append(
            {
                "id": f"g_{ticker.lower()}_{item.lower()}_{slug}_fy{fy}",
                "question": (
                    f"What did {ticker} disclose about {descr} in fiscal year {fy}?"
                ),
                "pipeline": "prose",
                "ticker": ticker,
                "fiscal_year": fy,
                "accession": accn,
                "item_section": item,
                "keywords": kws,
                "tags": ["prose", f"item-{item}"],
            }
        )

    print(json.dumps(rows, indent=2), flush=True)
    if skipped:
        print(f"\n# skipped {len(skipped)} rows:", file=sys.stderr)
        for s in skipped:
            print(f"#   {s}", file=sys.stderr)
    print(f"\n# wrote {len(rows)} rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
