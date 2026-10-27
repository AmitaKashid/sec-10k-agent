"""Audit and rewrite golden_dataset.json against the actual ingested DB.

For each row:
  - If `ticker` is not in `companies` → DROP (out-of-universe).
  - Else, look up the filing for (cik, fiscal_year) and replace
    `accession` with the real one.
  - If `value` is set, look up the matching xbrl fact and replace
    `value` with the actual ingested value.
  - If a match cannot be found, the row is left untouched and reported
    for human review.

Run from repo root:
    cd backend && uv run python ../evals/audit_dataset.py [--apply]

Without `--apply` it prints a diff and exits 0; with `--apply` it
overwrites golden_dataset.json (after backing it up to .bak).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from decimal import Decimal
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "backend"))

from sqlalchemy import select  # noqa: E402

from app.data.db import async_session_factory  # noqa: E402
from app.data.models import Company, Filing, XbrlFact  # noqa: E402

DATASET = _REPO / "evals" / "golden_dataset.json"


async def _resolve_company(session, ticker: str) -> str | None:
    return (
        await session.execute(select(Company.cik).where(Company.ticker == ticker))
    ).scalar_one_or_none()


async def _resolve_filing(session, cik: str, fiscal_year: int) -> str | None:
    return (
        await session.execute(
            select(Filing.accession_no)
            .where(Filing.cik == cik, Filing.fiscal_year == fiscal_year)
            .order_by(Filing.fiscal_year.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


async def _resolve_value(
    session, *, cik: str, concept_key: str, fiscal_year: int
) -> tuple[str, str] | None:
    """Return (value, accession_no) the same way the agent's runtime lookup does.

    The DB stores the friendly key ("revenue", "net_income", …) in
    `concept`. Comparative rows from prior years are tagged with the
    filing's fiscal_year — order by period_end DESC and take the first
    row to pick the period that corresponds to the requested fiscal year
    (mirrors `app.agent.tools.get_xbrl_fact`).
    """
    row = (
        await session.execute(
            select(XbrlFact.value, XbrlFact.accession_no)
            .where(
                XbrlFact.cik == cik,
                XbrlFact.concept == concept_key,
                XbrlFact.fiscal_year == fiscal_year,
                XbrlFact.fiscal_period == "FY",
            )
            .order_by(XbrlFact.period_end.desc())
            .limit(1)
        )
    ).first()
    if row is None:
        return None
    return str(row[0]), row[1]


async def main(apply: bool) -> int:
    rows = json.loads(DATASET.read_text())
    new_rows: list[dict] = []
    notes: list[str] = []

    async with async_session_factory() as session:
        for row in rows:
            ticker = row.get("ticker")
            fy = row.get("fiscal_year")

            # Adversarial-style refusal rows: keep as-is.
            if not ticker or row.get("pipeline") == "refuse":
                new_rows.append(row)
                continue

            cik = await _resolve_company(session, ticker)
            if cik is None:
                notes.append(f"DROP {row['id']}: ticker {ticker} not in DB")
                continue

            updated = dict(row)

            if fy is not None:
                accn = await _resolve_filing(session, cik, fy)
                if accn is None:
                    notes.append(f"DROP {row['id']}: no filing for {ticker} FY{fy}")
                    continue
                if updated.get("accession") != accn:
                    notes.append(
                        f"FIX {row['id']}: accession {updated.get('accession')!r} → {accn!r}"
                    )
                    updated["accession"] = accn

            if "value" in updated and updated.get("concept_key"):
                resolved = await _resolve_value(
                    session,
                    cik=cik,
                    concept_key=updated["concept_key"],
                    fiscal_year=fy,
                )
                if resolved is None:
                    notes.append(
                        f"DROP {row['id']}: no XBRL fact for "
                        f"{ticker}/{updated['concept_key']}/FY{fy} "
                        f"(likely EDGAR fiscal-year offset; track separately)"
                    )
                    continue
                else:
                    new_val, new_accn = resolved
                    if Decimal(updated["value"]) != Decimal(new_val):
                        notes.append(f"FIX {row['id']}: value {updated['value']} → {new_val}")
                    updated["value"] = new_val
                    if updated.get("accession") != new_accn:
                        notes.append(
                            f"FIX {row['id']}: xbrl-accession "
                            f"{updated.get('accession')!r} → {new_accn!r}"
                        )
                        updated["accession"] = new_accn

            new_rows.append(updated)

    print("\n".join(notes) if notes else "(no changes)")
    print(f"\n→ rows: {len(rows)} → {len(new_rows)}")

    if apply:
        backup = DATASET.with_suffix(".json.bak")
        shutil.copy2(DATASET, backup)
        DATASET.write_text(json.dumps(new_rows, indent=2))
        print(f"wrote {DATASET} (backup: {backup})")

    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    raise SystemExit(asyncio.run(main(args.apply)))
