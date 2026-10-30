"""Schema-level checks on AgentState. The contract these enforce is what
the nodes will rely on; if you change the shape, this is the failure mode."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.agent.state import (
    AgentState,
    ChunkHit,
    Claim,
    FilingRef,
    PlanResult,
    ValidateResult,
    XbrlFact,
)


def _xbrl_fact() -> XbrlFact:
    return XbrlFact(
        cik="0000320193",
        concept="revenue",
        concept_key="revenue",
        unit="USD",
        value=Decimal("391035000000"),
        period_end=date(2024, 9, 28),
        fiscal_year=2024,
        fiscal_period="FY",
        accession_no="0000320193-24-000123",
        form="10-K",
    )


def _chunk_hit() -> ChunkHit:
    return ChunkHit(
        chunk_id=1,
        accession_no="0000320193-24-000123",
        item="1A",
        seq=0,
        text="risk factor body",
        score=0.12,
        char_start=0,
        char_end=16,
    )


def test_agent_state_minimum_constructible():
    s = AgentState(question="hi", request_id="req-1")
    assert s.question == "hi"
    assert s.question_type is None
    assert s.cache_hit is False
    assert s.chunks == []
    assert s.claims == []
    assert s.trace_events == []


def test_filing_ref_required_fields():
    f = FilingRef(
        accession_no="0000320193-24-000123",
        fiscal_year=2024,
        period_of_report=date(2024, 9, 28),
        filing_date=date(2024, 11, 1),
    )
    assert f.fiscal_year == 2024


def test_chunk_hit_item_constrained():
    with pytest.raises(ValidationError):
        ChunkHit(
            chunk_id=1,
            accession_no="x",
            item="99",
            seq=0,
            text="x",
            score=0,
            char_start=0,
            char_end=1,
        )


def test_claim_with_xbrl_source_ref():
    c = Claim(text="$391,035M", kind="number", source_ref=_xbrl_fact())
    assert c.kind == "number"
    assert isinstance(c.source_ref, XbrlFact)


def test_claim_with_chunk_source_ref():
    c = Claim(text="risk factor", kind="prose", source_ref=_chunk_hit())
    assert c.kind == "prose"
    assert isinstance(c.source_ref, ChunkHit)


def test_claim_kind_constrained():
    with pytest.raises(ValidationError):
        Claim(text="x", kind="other", source_ref=None)


def test_plan_result_question_type_constrained():
    PlanResult(question_type="A_numerical")
    PlanResult(question_type="REFUSE", refusal_reason="off-topic")
    with pytest.raises(ValidationError):
        PlanResult(question_type="C_other")


def test_validate_result_default_empty_claims():
    v = ValidateResult(draft_answer="")
    assert v.claims == []


def test_agent_state_accumulates_chunks_and_trace_events():
    s = AgentState(question="q", request_id="r")
    # Default factories must produce independent lists per instance —
    # easy to break by using mutable default values.
    s.trace_events.append({"node": "plan"})
    s2 = AgentState(question="q2", request_id="r2")
    assert s2.trace_events == []
