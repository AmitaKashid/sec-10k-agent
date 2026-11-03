"""Metric coverage regressions: no database, provider keys, or model calls."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.metrics import RowOutcome, Thresholds, aggregate, format_report


def complete_rows():
    return [
        RowOutcome(id="number", pipeline="xbrl", refused=False,
                   citation_correct=True, numeric_correct=True, false_refusal=False),
        RowOutcome(id="prose", pipeline="prose", refused=False,
                   citation_correct=True, false_refusal=False,
                   judge_total_claims=3, judge_unsupported_count=0, judge_supported=True),
        RowOutcome(id="refusal", pipeline="refuse", refused=True, refused_correctly=True),
    ]


def test_empty_run_is_unknown_in_json_and_cannot_pass():
    metrics = aggregate([])
    decoded = json.loads(json.dumps(asdict(metrics)))
    for name in ("citation_accuracy", "unsupported_claim_rate", "hallucination_rate",
                 "refusal_accuracy", "false_refusal_rate"):
        assert decoded[name] is None
    assert metrics.numeric_eligible == 0
    assert not metrics.passes(Thresholds())
    report = format_report(metrics, [], Thresholds())
    assert "quality_gate: FAIL" in report
    assert "PASS" not in report
    assert "100.0%" not in report


@pytest.mark.parametrize("index", [0, 1, 2])
def test_single_category_reports_missing_metrics_without_passing(index):
    rows = [complete_rows()[index]]
    metrics = aggregate(rows)
    assert not metrics.passes(Thresholds())
    report = format_report(metrics, rows, Thresholds())
    assert "quality_gate: FAIL" in report
    assert "incomplete coverage" in report
    for metric in metrics.coverage_gaps():
        line = next(line for line in report.splitlines() if line.startswith(f"- {metric}:"))
        assert "NOT EVALUATED" in line
        assert "PASS" not in line


def test_no_judge_does_not_become_zero_unsupported_claims():
    rows = complete_rows()
    rows[1] = replace(rows[1], judge_total_claims=None,
                      judge_unsupported_count=None, judge_supported=None)
    metrics = aggregate(rows)
    assert metrics.unsupported_claim_rate is None
    assert metrics.coverage_gaps() == ["unsupported_claim_rate"]
    assert not metrics.passes(Thresholds())


def test_zero_claim_judge_result_is_not_an_observation():
    row = replace(complete_rows()[1], judge_total_claims=0)
    metrics = aggregate([row])
    assert metrics.unsupported_claim_rate is None
    assert metrics.hallucination_rate is None
    assert metrics.hallucination_eligible == 0


def test_all_false_refusals_do_not_claim_citation_accuracy():
    rows = [RowOutcome(id="answerable", pipeline="xbrl", refused=True, false_refusal=True)]
    metrics = aggregate(rows)
    assert metrics.false_refusal_rate == 1.0
    assert metrics.citation_accuracy is None
    assert metrics.numeric_eligible == 0
    assert not metrics.passes(Thresholds())
    report = format_report(metrics, rows, Thresholds())
    assert "false_refusal_rate: 100.0% [target <= 5.0%] FAIL" in report


def test_observed_zero_errors_remains_zero_and_complete_run_passes():
    rows = complete_rows()
    metrics = aggregate(rows)
    assert metrics.coverage_gaps() == []
    assert metrics.unsupported_claim_rate == 0.0
    assert metrics.hallucination_rate == 0.0
    assert metrics.citation_eligible == 2
    assert metrics.hallucination_eligible == 2
    assert metrics.passes(Thresholds())
    report = format_report(metrics, rows, Thresholds())
    assert "quality_gate: PASS" in report
    assert "NOT EVALUATED" not in report
    assert "numeric_errors: 0 / 1" in report


@pytest.mark.parametrize("changes, expected_metric", [
    ({"citation_accuracy": 0.5}, "citation_accuracy"),
    ({"numeric_error_count": 1}, "numeric_errors"),
    ({"unsupported_claim_rate": 0.05, "unsupported_claim_count": 1,
      "total_claims_judged": 20}, "unsupported_claim_rate"),
    ({"refusal_accuracy": 0.0}, "refusal_accuracy"),
    ({"false_refusal_rate": 0.1}, "false_refusal_rate"),
])
def test_complete_coverage_does_not_hide_threshold_failure(changes, expected_metric):
    rows = complete_rows()
    metrics = replace(aggregate(rows), **changes)
    assert metrics.coverage_gaps() == []
    assert not metrics.passes(Thresholds())
    report = format_report(metrics, rows, Thresholds())
    line = next(line for line in report.splitlines()
                if line.startswith(f"- {expected_metric}:"))
    assert "FAIL" in line
    assert "NOT EVALUATED" not in line


def test_coverage_and_threshold_failures_are_both_visible():
    row = replace(complete_rows()[0], numeric_correct=False)
    report = format_report(aggregate([row]), [row], Thresholds())
    assert "quality_gate: FAIL" in report
    assert "numeric_errors: 1 / 1 [tripwire: <= 0] FAIL" in report
    assert "unsupported_claim_rate: NOT EVALUATED" in report
