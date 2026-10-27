"""Unit tests for evals.metrics — synthetic states, no DB, no LLM."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.metrics import (  # noqa: E402
    Expected,
    Thresholds,
    aggregate,
    score_row,
)


def _state_xbrl(*, accn: str, value: str) -> dict:
    return {
        "verified": True,
        "final_answer": f"Revenue was {value}.",
        "filing": {"accession_no": accn},
        "xbrl_fact": {"accession_no": accn, "value": value, "concept_key": "revenue"},
        "claims": [
            {
                "text": value,
                "kind": "number",
                "source_ref": {"accession_no": accn, "value": value, "concept_key": "revenue"},
            }
        ],
    }


def _state_prose(*, accn: str, item: str, answer: str) -> dict:
    return {
        "verified": True,
        "final_answer": answer,
        "filing": {"accession_no": accn},
        "claims": [
            {
                "text": answer,
                "kind": "prose",
                "source_ref": {"accession_no": accn, "item": item, "text": answer},
            }
        ],
    }


def _state_refused(reason: str = "out of scope") -> dict:
    return {"refusal_reason_final": reason, "question_type": "REFUSE", "claims": []}


def test_xbrl_correct_passes_citation_and_numeric():
    exp = Expected(
        id="x1",
        question="q",
        pipeline="xbrl",
        accession="0000320193-23-000106",
        value="383285000000",
        tolerance_pct=0.5,
    )
    state = _state_xbrl(accn="0000320193-23-000106", value="383285000000")
    o = score_row(exp, state, latency_ms=12.0)
    assert o.citation_correct is True
    assert o.numeric_correct is True
    assert o.false_refusal is False


def test_xbrl_dashed_vs_undashed_accession_normalized():
    exp = Expected(id="x2", question="q", pipeline="xbrl", accession="0000320193-23-000106",
                   value="100", tolerance_pct=0)
    # Source uses the un-dashed form.
    state = _state_xbrl(accn="000032019323000106", value="100")
    o = score_row(exp, state)
    assert o.citation_correct is True


def test_xbrl_value_outside_tolerance_flags_hallucination():
    exp = Expected(id="x3", question="q", pipeline="xbrl",
                   accession="0000320193-23-000106", value="100", tolerance_pct=0.5)
    state = _state_xbrl(accn="0000320193-23-000106", value="200")
    o = score_row(exp, state)
    assert o.numeric_correct is False
    metrics = aggregate([o])
    assert metrics.hallucination_rate == 1.0


def test_prose_item_mismatch_fails_citation():
    exp = Expected(id="p1", question="q", pipeline="prose",
                   accession="0000320193-23-000106", item_section="1A",
                   keywords=["supply"])
    state = _state_prose(accn="0000320193-23-000106", item="7",
                         answer="Apple discussed services and supply lines.")
    o = score_row(exp, state)
    assert o.citation_correct is False
    assert o.keyword_coverage == 1.0  # keyword still present in answer


def test_refusal_correct():
    exp = Expected(id="r1", question="q", pipeline="refuse", should_refuse=True)
    o = score_row(exp, _state_refused())
    assert o.refused_correctly is True


def test_false_refusal_on_golden():
    exp = Expected(id="g1", question="q", pipeline="xbrl",
                   accession="0000320193-23-000106", value="100")
    o = score_row(exp, _state_refused("graph error"))
    assert o.false_refusal is True
    assert o.citation_correct is None  # not evaluated


def test_numeric_tripwire_fails_threshold():
    """One numeric error must fail Thresholds(), regardless of other metrics."""
    exp_ok = Expected(id="ok", question="q", pipeline="xbrl",
                      accession="X", value="100", tolerance_pct=0.5)
    exp_bad = Expected(id="bad", question="q", pipeline="xbrl",
                       accession="X", value="100", tolerance_pct=0.5)
    rows = [
        score_row(exp_ok, _state_xbrl(accn="X", value="100")),
        score_row(exp_bad, _state_xbrl(accn="X", value="200")),
    ]
    m = aggregate(rows)
    assert m.numeric_error_count == 1
    assert m.numeric_eligible == 2
    # 100% citation, but 1 numeric error → tripwire fails.
    assert m.citation_accuracy == 1.0
    assert not m.passes(Thresholds())


def test_claim_level_rate_not_row_level():
    """1 unsupported claim in a 5-claim row must score 20%, not 100%."""
    exp = Expected(id="p", question="q", pipeline="prose",
                   accession="X", item_section="1A")
    state = _state_prose(accn="X", item="1A", answer="some answer")
    o = score_row(exp, state)
    o.judge_supported = False
    o.judge_unsupported_claims = ["one bad claim"]
    o.judge_total_claims = 5
    o.judge_unsupported_count = 1

    m = aggregate([o])
    assert m.total_claims_judged == 5
    assert m.unsupported_claim_count == 1
    assert abs(m.unsupported_claim_rate - 0.2) < 1e-6
    # Headline row-level still 100% — preserved for DoD continuity.
    assert m.hallucination_rate == 1.0
    # 20% claim rate fails the 5% threshold.
    assert not m.passes(Thresholds())


def test_zero_numeric_errors_and_low_claim_rate_passes():
    rows = []
    # 4 cite-correct numeric rows, all numerics right.
    for i in range(4):
        rows.append(score_row(
            Expected(id=f"x{i}", question="q", pipeline="xbrl",
                     accession="X", value="100", tolerance_pct=0.5),
            _state_xbrl(accn="X", value="100"),
        ))
    # 20 judged claims, 0 unsupported.
    rows[0].judge_supported = True
    rows[0].judge_total_claims = 20
    rows[0].judge_unsupported_count = 0
    m = aggregate(rows)
    assert m.numeric_error_count == 0
    assert m.unsupported_claim_rate == 0.0
    assert m.passes(Thresholds())


def test_aggregate_mixed():
    rows = [
        # 2 cite-correct, 1 cite-wrong → 66.7% citation accuracy
        score_row(Expected(id="a", question="q", pipeline="xbrl",
                           accession="X", value="1"),
                  _state_xbrl(accn="X", value="1")),
        score_row(Expected(id="b", question="q", pipeline="xbrl",
                           accession="X", value="1"),
                  _state_xbrl(accn="X", value="1")),
        score_row(Expected(id="c", question="q", pipeline="xbrl",
                           accession="X", value="1"),
                  _state_xbrl(accn="Y", value="1")),
        # adversarial refusal
        score_row(Expected(id="r", question="q", pipeline="refuse", should_refuse=True),
                  _state_refused()),
    ]
    m = aggregate(rows)
    assert m.n == 4
    assert m.n_golden == 3
    assert m.n_adversarial == 1
    assert abs(m.citation_accuracy - 2 / 3) < 1e-6
    assert m.refusal_accuracy == 1.0
    assert m.false_refusal_rate == 0.0
    assert not m.passes(Thresholds())  # citation < 0.85
