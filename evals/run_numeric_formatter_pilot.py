"""Compare answer formatting on shared XBRL inputs without agent or cache calls."""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from evals.metrics import Expected, score_numeric


async def run(model_name, output):
    from app.agent.llm import get_chat_model
    from app.agent.nodes.validate import format_numeric_fact
    from app.agent.tools import get_xbrl_fact
    from app.data.db import async_session_factory

    rows = json.loads((ROOT / "evals/numeric_formatter_pilot.json").read_text(encoding="utf-8"))
    model = get_chat_model(model_name, streaming=False)
    results = []
    async with async_session_factory() as session:
        for raw in rows:
            expected = Expected.model_validate(raw)
            # Accession prefixes can identify a filing agent, not the issuer.
            cik = {"AAPL": "0000320193", "MSFT": "0000789019"}[expected.ticker]
            fact = await get_xbrl_fact(cik, expected.concept_key, expected.fiscal_year, session)
            row = {"id": expected.id, "expected": raw, "fact": None, "variants": {}}
            results.append(row)
            if fact is None:
                row["input_issue"] = "missing_fact"
                continue
            row["fact"] = fact.model_dump(mode="json")
            row["input_matches_label"] = (
                fact.value == expected.value
                and fact.unit == raw["reference"]["unit"]
                and fact.accession_no.replace("-", "") == expected.accession.replace("-", "")
            )
            start = perf_counter()
            answer = format_numeric_fact(expected.ticker, fact)
            elapsed = (perf_counter() - start) * 1000
            ok, note = score_numeric(expected, {"final_answer": answer})
            row["variants"]["python"] = {
                "answer": answer, "amount_correct": ok, "note": note,
                "ms": elapsed, "generation_invocations": 0,
            }
            prompt = (
                "Answer in one sentence using only this supplied fact. "
                "Display the exact unrounded amount with its unit. "
                "Do not add comparisons or other amounts. "
                f"Question: {expected.question}\nFact: {fact.model_dump_json()}"
            )
            start = perf_counter()
            try:
                response = await model.ainvoke(prompt)
                elapsed = (perf_counter() - start) * 1000
                answer = response.content
                ok, note = score_numeric(expected, {"final_answer": answer})
                result = {"answer": answer, "amount_correct": ok, "note": note, "ms": elapsed}
            except Exception as exc:
                result = {"error": type(exc).__name__, "amount_correct": False,
                          "ms": (perf_counter() - start) * 1000}
            row["variants"]["llm"] = {**result, "generation_invocations": 1}
    summary = {}
    for name in ("python", "llm"):
        runs = [r["variants"][name] for r in results if name in r["variants"]]
        completed = [r for r in runs if "error" not in r]
        summary[name] = {
            "correct_amounts": sum(r["amount_correct"] for r in runs),
            "cases": len(rows),
            "attempted": len(runs),
            "errors": len(runs) - len(completed),
            "generation_invocations": sum(r["generation_invocations"] for r in runs),
            "median_formatting_ms": median(r["ms"] for r in completed) if completed else None,
        }
    report = {"model": model_name, "summary": summary, "rows": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    asyncio.run(run(args.model, args.output))
