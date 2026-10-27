"""Eval runner — invokes the LangGraph in-process and scores rows.

Usage (from repo root):
    cd backend && uv run python ../evals/run_evals.py \
        --golden ../evals/golden_dataset.json \
        --adversarial ../evals/adversarial.json \
        --out-dir ../evals/results

The runner imports `app.agent.graph.GRAPH` directly — same compiled
graph the FastAPI route uses — so we test the actual product, not an
HTTP envelope around it. Postgres + ingestion must already be up; this
runner does not seed data.

Exits non-zero if any threshold in `metrics.Thresholds` is violated, so
CI can use it as a gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allow running as `python evals/run_evals.py` from the repo root by
# adding `backend/` to sys.path. When invoked from inside `backend/`
# (the documented form) this is a no-op.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "backend"))
sys.path.insert(0, str(_REPO))

from evals.metrics import (  # noqa: E402  — sys.path tweak above is intentional
    Expected,
    RowOutcome,
    Thresholds,
    aggregate,
    format_report,
    score_row,
)
from evals.judge import DEFAULT_JUDGE_MODEL, judge_hallucination  # noqa: E402


def _load_dataset(path: Path) -> list[Expected]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON array of rows")
    return [Expected.model_validate(r) for r in raw]


async def _run_one(graph, expected: Expected) -> tuple[RowOutcome, dict[str, Any]]:
    """Invoke the graph for one row and score it.

    Returns (outcome, terminal_state) so the caller can persist raw
    transcripts alongside scores.
    """
    # Imported lazily so `--help` works without a configured DB / API key.
    from app.agent.state import AgentState

    initial = AgentState(question=expected.question, request_id=f"eval-{expected.id}")
    t0 = time.perf_counter()
    try:
        final = await graph.ainvoke(initial)
    except Exception as e:  # noqa: BLE001 — turn graph crashes into a row failure
        outcome = RowOutcome(
            id=expected.id,
            pipeline=expected.pipeline,
            refused=False,
            latency_ms=(time.perf_counter() - t0) * 1000,
            notes=[f"graph crashed: {type(e).__name__}: {str(e)[:160]}"],
        )
        if not expected.should_refuse:
            outcome.citation_correct = False
            outcome.false_refusal = False
        return outcome, {}

    latency_ms = (time.perf_counter() - t0) * 1000
    state = final.model_dump(mode="python") if hasattr(final, "model_dump") else dict(final)
    return score_row(expected, state, latency_ms=latency_ms), state


def _sources_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in state.get("claims") or []:
        ref = c.get("source_ref") if isinstance(c, dict) else getattr(c, "source_ref", None)
        if ref is None:
            continue
        out.append(ref.model_dump(mode="python") if hasattr(ref, "model_dump") else ref)
    return out


async def _judge_outcome(
    expected: Expected,
    outcome: RowOutcome,
    state: dict[str, Any],
    *,
    judge_model: str,
) -> None:
    """Run the LLM-as-judge for one row and merge the verdict in-place.

    Skipped for refusals (nothing to ground), false refusals (no answer
    to judge), and rows that already failed the structural numeric
    check (already counted as hallucinated)."""
    if expected.should_refuse or outcome.refused:
        return
    if outcome.numeric_correct is False:
        return  # already a hallucination by the structural floor
    answer = state.get("final_answer") or state.get("draft_answer")
    if not answer:
        return
    sources = _sources_from_state(state)
    verdict = await judge_hallucination(
        question=expected.question,
        answer=answer,
        sources=sources,
        model=judge_model,
    )
    outcome.judge_supported = verdict.supported
    outcome.judge_unsupported_claims = verdict.unsupported_claims
    outcome.judge_total_claims = verdict.total_claims
    outcome.judge_unsupported_count = verdict.unsupported_count
    if not verdict.supported:
        outcome.notes.append(
            "judge: " + (verdict.reasoning or "; ".join(verdict.unsupported_claims) or "unsupported")
        )


async def _run_all(
    rows: list[Expected],
    *,
    use_judge: bool,
    judge_model: str,
) -> tuple[list[RowOutcome], list[dict[str, Any]]]:
    from app.agent.graph import GRAPH

    outcomes: list[RowOutcome] = []
    transcripts: list[dict[str, Any]] = []
    for row in rows:
        print(f"  · {row.id} ({row.pipeline}) …", flush=True)
        outcome, state = await _run_one(GRAPH, row)
        if use_judge:
            try:
                await _judge_outcome(row, outcome, state, judge_model=judge_model)
            except Exception as e:  # noqa: BLE001
                outcome.notes.append(f"judge crashed: {type(e).__name__}: {str(e)[:120]}")
        outcomes.append(outcome)
        transcripts.append(
            {
                "id": row.id,
                "question": row.question,
                "answer": state.get("final_answer") or state.get("draft_answer"),
                "verified": state.get("verified"),
                "refused": outcome.refused,
                "refusal_reason": state.get("refusal_reason_final")
                or state.get("refusal_reason"),
                "trace_events": state.get("trace_events") or [],
                "chunks_retrieved": len(state.get("chunks") or []),
                "outcome": outcome.__dict__,
            }
        )
    return outcomes, transcripts


def _outcome_to_jsonable(o: RowOutcome) -> dict[str, Any]:
    return {k: v for k, v in o.__dict__.items()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--golden", type=Path, default=_REPO / "evals" / "golden_dataset.json")
    p.add_argument("--adversarial", type=Path, default=_REPO / "evals" / "adversarial.json")
    p.add_argument("--out-dir", type=Path, default=_REPO / "evals" / "results")
    p.add_argument("--threshold-citation", type=float, default=0.85)
    p.add_argument(
        "--threshold-numeric-errors", type=int, default=0,
        help="max numeric tripwire failures (default 0 — any wrong number fails CI)",
    )
    p.add_argument(
        "--threshold-unsupported-claims", type=float, default=0.05,
        help="max claim-level unsupported rate (default 0.05)",
    )
    p.add_argument("--threshold-refusal", type=float, default=1.0)
    p.add_argument("--threshold-false-refusal", type=float, default=0.05)
    p.add_argument(
        "--limit", type=int, default=None,
        help="run only the first N rows from each dataset (smoke mode)",
    )
    p.add_argument(
        "--no-judge", action="store_true",
        help="skip LLM-as-judge semantic hallucination scoring",
    )
    p.add_argument(
        "--judge-model", default=DEFAULT_JUDGE_MODEL,
        help=f"chat model used as the hallucination judge (default: {DEFAULT_JUDGE_MODEL})",
    )
    args = p.parse_args(argv)

    golden = _load_dataset(args.golden)
    adversarial = _load_dataset(args.adversarial)
    if args.limit is not None:
        golden = golden[: args.limit]
        adversarial = adversarial[: args.limit]
    rows = golden + adversarial
    if not rows:
        print("no rows loaded — nothing to evaluate", file=sys.stderr)
        return 2

    judge_note = "judge=off" if args.no_judge else f"judge={args.judge_model}"
    print(
        f"running {len(rows)} rows ({len(golden)} golden, "
        f"{len(adversarial)} adversarial; {judge_note})"
    )
    outcomes, transcripts = asyncio.run(
        _run_all(rows, use_judge=not args.no_judge, judge_model=args.judge_model)
    )

    metrics = aggregate(outcomes)
    thresholds = Thresholds(
        citation_accuracy=args.threshold_citation,
        numeric_errors_max=args.threshold_numeric_errors,
        unsupported_claim_rate=args.threshold_unsupported_claims,
        refusal_accuracy=args.threshold_refusal,
        false_refusal_rate=args.threshold_false_refusal,
    )
    report = format_report(metrics, outcomes, thresholds)
    print()
    print(report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (args.out_dir / f"{stamp}.json").write_text(
        json.dumps(
            {
                "metrics": metrics.__dict__,
                "thresholds": thresholds.__dict__,
                "outcomes": [_outcome_to_jsonable(o) for o in outcomes],
                "transcripts": transcripts,
            },
            indent=2,
            default=str,
        )
    )
    (args.out_dir / f"{stamp}.md").write_text(report, encoding="utf-8")
    print(f"wrote {args.out_dir / f'{stamp}.json'}")

    return 0 if metrics.passes(thresholds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
