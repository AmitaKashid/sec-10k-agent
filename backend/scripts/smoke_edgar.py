"""Day-1 milestone: fetch real data from SEC EDGAR with one command.

Usage:
    cd backend && uv run python scripts/smoke_edgar.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data import edgar_client  # noqa: E402


async def main() -> None:
    print("→ resolving AAPL...")
    cik = await edgar_client.get_cik_for_ticker("AAPL")
    print(f"  CIK = {cik}")
    assert cik == "0000320193", f"expected 0000320193, got {cik}"

    print("→ fetching submissions...")
    filings = await edgar_client.get_submissions(cik, form_type="10-K")
    print(f"  found {len(filings)} 10-K filings")
    latest = filings[0]
    print(
        f"  latest: {latest.accession_no}  filed={latest.filing_date}  period={latest.period_of_report}"
    )

    print("→ fetching latest 10-K HTML...")
    html = await edgar_client.get_filing_html(cik, latest.accession_no, latest.primary_doc)
    print(f"  downloaded {len(html):,} bytes")
    assert b"<html" in html.lower() or b"<!doctype" in html.lower()

    print("→ fetching XBRL company facts...")
    facts = await edgar_client.get_company_facts(cik)
    rev = facts["facts"]["us-gaap"].get("Revenues") or facts["facts"]["us-gaap"].get(
        "RevenueFromContractWithCustomerExcludingAssessedTax"
    )
    rnd = facts["facts"]["us-gaap"]["ResearchAndDevelopmentExpense"]
    print(f"  Revenues: {len(rev['units']['USD'])} historical rows")
    print(f"  R&D:      {len(rnd['units']['USD'])} historical rows")

    fy_revs = [r for r in rev["units"]["USD"] if r.get("fp") == "FY" and r.get("form") == "10-K"]
    fy_revs.sort(key=lambda r: r["fy"], reverse=True)
    if fy_revs:
        latest_rev = fy_revs[0]
        print(f"  latest FY revenue: ${latest_rev['val']:,} (FY{latest_rev['fy']})")

    print("\n✓ Day 1 milestone complete: SEC EDGAR connection works.")


if __name__ == "__main__":
    asyncio.run(main())
