"""Pure scoring functions for the SEC 10-K agent eval harness.

Score one row at a time, then aggregate. No LLM calls live here — the
LLM-as-judge for hallucination semantics lands in `judge.py`. This module
operates on the terminal `AgentState` (as a dict) projected through
`app.api.sse.terminal_payload`-compatible fields, plus the raw `claims`
list with their `source_ref` objects.

Metrics computed (per CLAUDE.md DoD):
  - citation_accuracy   target ≥ 0.85
  - hallucination_rate  target < 0.05  (structural only here; semantic
                                        layer added by the judge)
  - refusal_accuracy    target = 1.00 on adversarial refusal rows
  - false_refusal_rate  target ≈ 0.00 on the golden set
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field

Pipeline = Literal["xbrl", "prose", "cross", "refuse"]
ItemKey = Literal["1A", "7", "8"]


# --- Dataset schema ------------------------------------------------------


class Expected(BaseModel):
    """One row of the golden / adversarial dataset.

    `pipeline` decides which scorer paths apply:
      xbrl   → numeric_correct + citation (accession)
      prose  → keyword_coverage + citation (accession + item)
      cross  → expected.accession may be a list (any-of match)
      refuse → should_refuse must be True; everything else ignored
    """

    id: str
    question: str
    pipeline: Pipeline
    ticker: str | None = None
    fiscal_year: int | None = None
    accession: str | list[str] | None = None
    item_section: ItemKey | None = None
    concept_key: str | None = None
    value: Decimal | None = None
    tolerance_pct: float = 0.5
    should_refuse: bool = False
    refusal_kinds: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


# --- Per-row outcome -----------------------------------------------------


@dataclass
class RowOutcome:
    """What actually happened for one row. Fields are None when not
    applicable to the row's pipeline (so aggregation can ignore them)."""

    id: str
    pipeline: Pipeline
    refused: bool
    citation_correct: bool | None = None
    numeric_correct: bool | None = None
    keyword_coverage: float | None = None  # 0.0–1.0
    refused_correctly: bool | None = None  # only set when should_refuse
    false_refusal: bool | None = None  # only set on golden rows
    judge_supported: bool | None = None  # set by run_evals after LLM-as-judge pass
    judge_unsupported_claims: list[str] = field(default_factory=list)
    # Per-claim counts (None when judge didn't run on this row).
    judge_total_claims: int | None = None
    judge_unsupported_count: int | None = None
    latency_ms: float | None = None
    notes: list[str] = field(default_factory=list)
    # Retrieval quality — None when not applicable (xbrl, refusal rows)
    recall_at_5: bool | None = None
    recall_at_8: bool | None = None


# --- Helpers -------------------------------------------------------------


def _normalize_accn(s: str | None) -> str | None:
    if not s:
        return None
    return s.replace("-", "").strip()


def _accn_set(expected: str | list[str] | None) -> set[str]:
    if expected is None:
        return set()
    if isinstance(expected, str):
        return {_normalize_accn(expected) or ""}
    return {_normalize_accn(s) or "" for s in expected}


def _claim_sources(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten claim source_refs (XbrlFact or ChunkHit) into plain dicts.

    Accepts both Pydantic models and dicts so callers can hand us either
    a raw AgentState dump or a JSON-roundtripped payload.
    """
    out: list[dict[str, Any]] = []
    for c in state.get("claims") or []:
        ref = c.get("source_ref") if isinstance(c, dict) else getattr(c, "source_ref", None)
        if ref is None:
            continue
        if hasattr(ref, "model_dump"):
            out.append(ref.model_dump(mode="python"))
        elif isinstance(ref, dict):
            out.append(ref)
    return out


def _filing_accn(state: dict[str, Any]) -> str | None:
    f = state.get("filing")
    if f is None:
        return None
    if hasattr(f, "accession_no"):
        return f.accession_no
    if isinstance(f, dict):
        return f.get("accession_no")
    return None


# --- Scorers -------------------------------------------------------------


def score_citation(expected: Expected, state: dict[str, Any]) -> tuple[bool, str]:
    """Did the agent cite a filing matching expected.accession (and item)?

    Looks at every claim's source_ref; falls back to state.filing if no
    claims carry a source_ref (shouldn't happen on a verified answer, but
    be tolerant).
    """
    wanted = _accn_set(expected.accession)
    if not wanted:
        return True, "no expected accession"

    sources = _claim_sources(state)
    seen_accns = {_normalize_accn(s.get("accession_no")) for s in sources if s.get("accession_no")}
    seen_accns.discard(None)

    if not seen_accns:
        fallback = _normalize_accn(_filing_accn(state))
        if fallback:
            seen_accns = {fallback}

    if not seen_accns & wanted:
        return False, f"cited {seen_accns or '∅'}, expected one of {wanted}"

    if expected.pipeline == "prose" and expected.item_section is not None:
        items = {s.get("item") for s in sources if s.get("item")}
        if expected.item_section not in items:
            return False, f"item mismatch: cited {items}, expected {expected.item_section}"

    return True, "ok"


_ANSWER_AMOUNT = re.compile(
    r"(?<![\w.,])([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"(?:\s*(thousand|million|billion|trillion)\b)?(?![\w]|[.,]\d)",
    re.IGNORECASE,
)
_AMOUNT_SCALE = {"": 1, "thousand": 10**3, "million": 10**6,
                 "billion": 10**9, "trillion": 10**12}


def score_numeric(expected: Expected, state: dict[str, Any]) -> tuple[bool, str]:
    """Score the single displayed amount against the independent label.

    Explicit fiscal-year labels and numbered citations are not amounts.
    Missing or ambiguous amounts fail rather than falling back to retrieval.
    """
    if expected.value is None:
        return True, "no expected value"

    answer = state.get("final_answer")
    if not isinstance(answer, str) or not answer.strip():
        return False, "no final numerical answer"

    text = re.sub(r"\[\d+\]|\bFY\s*\d{4}\b|\bfiscal year\s+\d{4}\b",
                  " ", answer, flags=re.IGNORECASE)
    amounts = _ANSWER_AMOUNT.findall(text.replace("−", "-").replace("$", ""))
    if len(amounts) != 1 or "%" in text:
        return False, "expected one unambiguous amount in the final answer"

    number, scale = amounts[0]
    displayed = Decimal(number.replace(",", "")) * _AMOUNT_SCALE[scale.lower()]
    target = expected.value
    allowed_error = abs(target) * Decimal(str(expected.tolerance_pct)) / 100
    return (
        abs(displayed - target) <= allowed_error,
        f"displayed {displayed}, expected {target} ±{expected.tolerance_pct}%",
    )


def score_recall_at_k(
    expected: Expected,
    chunks: list[dict[str, Any]],
    k: int,
) -> bool | None:
    """Recall@K: did any of the top-K retrieved chunks contain a keyword?

    Returns None (not applicable) for rows without keywords or for
    pipeline types where retrieval isn't prose-based (xbrl, refusal).
    Returns False when k chunks were requested but retrieval returned
    nothing — that itself is a retrieval failure worth surfacing.
    """
    if not expected.keywords:
        return None
    if expected.pipeline not in ("prose", "cross"):
        return None

    top_k = chunks[:k]
    if not top_k:
        return False

    combined = " ".join((c.get("text") or "").lower() for c in top_k)
    return any(kw.lower() in combined for kw in expected.keywords)


def score_keywords(expected: Expected, state: dict[str, Any]) -> float:
    """Fraction of expected keywords appearing in the answer text."""
    if not expected.keywords:
        return 1.0
    answer = (state.get("final_answer") or state.get("draft_answer") or "").lower()
    if not answer:
        return 0.0
    hits = sum(1 for kw in expected.keywords if kw.lower() in answer)
    return hits / len(expected.keywords)


def is_refused(state: dict[str, Any]) -> bool:
    """Mirror app.api.sse.terminal_payload's refused logic."""
    return state.get("question_type") == "REFUSE" or bool(state.get("refusal_reason_final"))


# --- Top-level row scorer ------------------------------------------------


def score_row(
    expected: Expected,
    state: dict[str, Any],
    *,
    latency_ms: float | None = None,
) -> RowOutcome:
    refused = is_refused(state)
    out = RowOutcome(
        id=expected.id,
        pipeline=expected.pipeline,
        refused=refused,
        latency_ms=latency_ms,
    )

    # Retrieval quality — computed regardless of refusal so we can diagnose
    # false refusals caused by retrieval failures (recall=0 → wrong chunks).
    chunks = state.get("chunks") or []
    if hasattr(chunks[0], "model_dump") if chunks else False:
        chunks = [c.model_dump(mode="python") for c in chunks]
    out.recall_at_5 = score_recall_at_k(expected, chunks, k=5)
    out.recall_at_8 = score_recall_at_k(expected, chunks, k=8)

    if expected.should_refuse:
        out.refused_correctly = refused
        if not refused:
            out.notes.append("expected refusal, agent answered")
        return out

    out.false_refusal = refused
    if refused:
        reason = state.get("refusal_reason_final") or state.get("refusal_reason")
        out.notes.append(f"unexpected refusal: {reason}")
        return out  # everything else moot for a refused row

    cite_ok, cite_note = score_citation(expected, state)
    out.citation_correct = cite_ok
    if not cite_ok:
        out.notes.append(f"citation: {cite_note}")

    if expected.pipeline in ("xbrl", "cross") and expected.value is not None:
        num_ok, num_note = score_numeric(expected, state)
        out.numeric_correct = num_ok
        if not num_ok:
            out.notes.append(f"numeric: {num_note}")

    if expected.pipeline in ("prose", "cross") and expected.keywords:
        out.keyword_coverage = score_keywords(expected, state)

    return out


# --- Aggregation ---------------------------------------------------------


@dataclass
class Thresholds:
    citation_accuracy: float = 0.85
    # Numeric is a tripwire, not a rate: the architecture's whole pitch is
    # that hallucinated numbers are structurally impossible (XBRL pipeline).
    # Any numeric failure means that guarantee broke — fail loud.
    numeric_errors_max: int = 0
    # Semantic hallucination is scored at the *claim* level, not the row
    # level: 1 unsupported claim in a 5-claim answer is 20%, not 100%.
    unsupported_claim_rate: float = 0.05
    refusal_accuracy: float = 1.0
    false_refusal_rate: float = 0.05


@dataclass
class Metrics:
    n: int
    n_golden: int
    n_adversarial: int
    citation_accuracy: float | None
    numeric_error_count: int
    numeric_eligible: int
    unsupported_claim_rate: float | None
    total_claims_judged: int
    unsupported_claim_count: int
    hallucination_rate: float | None
    refusal_accuracy: float | None
    false_refusal_rate: float | None
    median_latency_ms: float | None
    recall_at_5: float | None = None
    recall_at_8: float | None = None
    citation_eligible: int = 0
    hallucination_eligible: int = 0

    def coverage_gaps(self) -> list[str]:
        """Required metrics with no observations; subsets remain diagnostic."""
        gaps = []
        for name, observed in (
            ("citation_accuracy", self.citation_accuracy is not None),
            ("numeric_errors", self.numeric_eligible > 0),
            ("unsupported_claim_rate", self.unsupported_claim_rate is not None),
            ("refusal_accuracy", self.refusal_accuracy is not None),
            ("false_refusal_rate", self.false_refusal_rate is not None),
        ):
            if not observed:
                gaps.append(name)
        return gaps

    def passes(self, t: Thresholds) -> bool:
        # No observations cannot satisfy a release-quality threshold.
        return (
            self.citation_accuracy is not None
            and self.citation_accuracy >= t.citation_accuracy
            and self.numeric_eligible > 0
            and self.numeric_error_count <= t.numeric_errors_max
            and self.unsupported_claim_rate is not None
            and self.unsupported_claim_rate < t.unsupported_claim_rate
            and self.refusal_accuracy is not None
            and self.refusal_accuracy >= t.refusal_accuracy
            and self.false_refusal_rate is not None
            and self.false_refusal_rate <= t.false_refusal_rate
        )


def _safe_div(num: int, den: int) -> float:
    return num / den if den else 0.0


def _accuracy(num: int, den: int) -> float | None:
    """Absent observations are unknown, not perfect (or zero) performance."""
    return num / den if den else None


def aggregate(outcomes: list[RowOutcome]) -> Metrics:
    n = len(outcomes)

    # Citation accuracy: among rows that were *expected to be answered*
    # (false_refusal in {True, False}) and were not falsely refused.
    cite_eligible = [o for o in outcomes if o.citation_correct is not None]
    cite_correct = sum(1 for o in cite_eligible if o.citation_correct)

    # Numeric tripwire: count how many rows had a numeric check that failed.
    numeric_eligible_rows = [o for o in outcomes if o.numeric_correct is not None]
    numeric_error_count = sum(1 for o in numeric_eligible_rows if o.numeric_correct is False)

    # Semantic hallucination: claim-level, summed across all judged rows.
    total_claims = sum(o.judge_total_claims or 0 for o in outcomes)
    unsupported_claims = sum(o.judge_unsupported_count or 0 for o in outcomes)
    unsupported_rate = _accuracy(unsupported_claims, total_claims)

    # Headline hallucination_rate (derived, row-level OR — kept for DoD
    # continuity). Eligible = any answered row with at least one signal.
    halluc_eligible = [
        o for o in outcomes
        if o.numeric_correct is not None or (o.judge_total_claims or 0) > 0
    ]
    halluc_wrong = sum(
        1 for o in halluc_eligible
        if (o.numeric_correct is False) or ((o.judge_unsupported_count or 0) > 0)
    )

    # Refusal accuracy (adversarial side).
    refusal_eligible = [o for o in outcomes if o.refused_correctly is not None]
    refused_ok = sum(1 for o in refusal_eligible if o.refused_correctly)

    # False refusal (golden side).
    false_eligible = [o for o in outcomes if o.false_refusal is not None]
    falsely_refused = sum(1 for o in false_eligible if o.false_refusal)

    latencies = sorted(o.latency_ms for o in outcomes if o.latency_ms is not None)
    median = latencies[len(latencies) // 2] if latencies else None

    # Recall@K — only prose/cross rows where the metric is applicable.
    r5_rows = [o for o in outcomes if o.recall_at_5 is not None]
    r8_rows = [o for o in outcomes if o.recall_at_8 is not None]
    recall_5 = _safe_div(sum(1 for o in r5_rows if o.recall_at_5), len(r5_rows)) if r5_rows else None
    recall_8 = _safe_div(sum(1 for o in r8_rows if o.recall_at_8), len(r8_rows)) if r8_rows else None

    return Metrics(
        n=n,
        n_golden=len(false_eligible),
        n_adversarial=len(refusal_eligible),
        citation_accuracy=_accuracy(cite_correct, len(cite_eligible)),
        numeric_error_count=numeric_error_count,
        numeric_eligible=len(numeric_eligible_rows),
        unsupported_claim_rate=unsupported_rate,
        total_claims_judged=total_claims,
        unsupported_claim_count=unsupported_claims,
        hallucination_rate=_accuracy(halluc_wrong, len(halluc_eligible)),
        refusal_accuracy=_accuracy(refused_ok, len(refusal_eligible)),
        false_refusal_rate=_accuracy(falsely_refused, len(false_eligible)),
        median_latency_ms=median,
        recall_at_5=recall_5,
        recall_at_8=recall_8,
        citation_eligible=len(cite_eligible),
        hallucination_eligible=len(halluc_eligible),
    )


# --- Reporting -----------------------------------------------------------


def format_report(metrics: Metrics, outcomes: list[RowOutcome], t: Thresholds) -> str:
    def pct(value: float | None) -> str:
        return "NOT EVALUATED" if value is None else f"{value * 100:.1f}%"

    def result(value: float | None, threshold: float, comparison: str) -> str:
        if value is None:
            return "NOT EVALUATED"
        passed = (
            value >= threshold if comparison == ">=" else
            value <= threshold if comparison == "<=" else
            value < threshold
        )
        return f"{pct(value)} [target {comparison} {pct(threshold)}] " + (
            "PASS" if passed else "FAIL"
        )

    if metrics.numeric_eligible:
        numeric = (
            f"{metrics.numeric_error_count} / {metrics.numeric_eligible} "
            f"[tripwire: <= {t.numeric_errors_max}] "
            + ("PASS" if metrics.numeric_error_count <= t.numeric_errors_max else "FAIL")
        )
    else:
        numeric = "NOT EVALUATED (0 eligible rows)"

    gaps = metrics.coverage_gaps()
    gate = "PASS" if metrics.passes(t) else "FAIL"
    if gaps:
        gate += " — incomplete coverage; not evaluated: " + ", ".join(gaps)
    head = [
        "# Eval Report",
        "",
        f"- rows: {metrics.n} ({metrics.n_golden} golden, {metrics.n_adversarial} adversarial)",
        f"- quality_gate: {gate}",
        f"- citation_accuracy: {result(metrics.citation_accuracy, t.citation_accuracy, '>=')} "
        f"({metrics.citation_eligible} evaluated rows)",
        f"- numeric_errors: {numeric}",
        f"- unsupported_claim_rate: "
        f"{result(metrics.unsupported_claim_rate, t.unsupported_claim_rate, '<')} "
        f"({metrics.unsupported_claim_count}/{metrics.total_claims_judged} claims)",
        f"- hallucination_rate: {pct(metrics.hallucination_rate)} "
        f"[derived row-level diagnostic; {metrics.hallucination_eligible} evaluated rows]",
        f"- refusal_accuracy: {result(metrics.refusal_accuracy, t.refusal_accuracy, '>=')} "
        f"({metrics.n_adversarial} evaluated rows)",
        f"- false_refusal_rate: "
        f"{result(metrics.false_refusal_rate, t.false_refusal_rate, '<=')} "
        f"({metrics.n_golden} evaluated rows)",
    ]
    if metrics.median_latency_ms is not None:
        head.append(f"- median_latency_ms: {metrics.median_latency_ms:.0f}")
    if metrics.recall_at_5 is not None:
        head.append(f"- recall_at_5: {pct(metrics.recall_at_5)} [retrieval diagnostic]")
    if metrics.recall_at_8 is not None:
        head.append(f"- recall_at_8: {pct(metrics.recall_at_8)} [retrieval diagnostic]")

    failed = [o for o in outcomes if o.notes]
    if failed:
        head += ["", "## Failures"]
        for o in failed:
            head.append(f"- **{o.id}** ({o.pipeline}): {'; '.join(o.notes)}")
    return "\n".join(head) + "\n"


__all__ = [
    "Expected",
    "Metrics",
    "RowOutcome",
    "Thresholds",
    "aggregate",
    "format_report",
    "is_refused",
    "score_citation",
    "score_keywords",
    "score_numeric",
    "score_row",
]
