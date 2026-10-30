"""Graph integration tests against an inline AAPL fixture.

Real Postgres + pgvector + MOCKED LLMs. The `agent_session` fixture
seeds the company/filing/XBRL fact inline, so these tests are
self-contained — no external `seed_dev` step required.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import GRAPH
from app.agent.nodes import plan as plan_module
from app.agent.nodes import retrieve as retrieve_module
from app.agent.nodes import validate as validate_module
from app.agent.nodes.validate import _ClaimByIdx, _ValidateLLMResponse
from app.agent.state import AgentState, ChunkHit, PlanResult
from app.data.models import AnswerCache
from tests.integration.conftest import SKIP_REASON, db_reachable

pytestmark = pytest.mark.skipif(not db_reachable(), reason=SKIP_REASON)


# --- Helpers --------------------------------------------------------------


class _StructuredFake:
    def __init__(self, response, include_raw: bool = False):
        self._response = response
        self._include_raw = include_raw

    async def ainvoke(self, prompt):
        if self._include_raw:
            return {"raw": None, "parsed": self._response, "parsing_error": None}
        return self._response


class _ChatFake:
    def __init__(self, response):
        self._response = response

    def bind(self, **kwargs):
        return self

    async def ainvoke(self, prompt):
        if isinstance(self._response, PlanResult):
            content = self._response.model_dump_json()
        else:
            content = json.dumps({"selected_ids": sorted({c.source_idx for c in self._response.claims})})
        return AIMessage(content=content, response_metadata={"finish_reason": "stop"})

    def with_structured_output(self, schema, *, include_raw: bool = False, **_):
        return _StructuredFake(self._response, include_raw=include_raw)


def _patch_plan(monkeypatch, plan_response: PlanResult):
    monkeypatch.setattr(plan_module, "get_chat_model", lambda *a, **kw: _ChatFake(plan_response))


def _patch_validate(monkeypatch, validate_response: _ValidateLLMResponse):
    monkeypatch.setattr(
        validate_module, "get_chat_model", lambda *a, **kw: _ChatFake(validate_response)
    )


# --- Tests ---------------------------------------------------------------


async def test_aapl_revenue_numerical_path(monkeypatch, agent_session: AsyncSession):
    """Full graph: plan → fetch → locate → retrieve → validate → cite_check.
    Real DB lookups; mocked LLMs emit deterministic structured output."""
    _patch_plan(
        monkeypatch,
        PlanResult(
            question_type="A_numerical",
            ticker="AAPL",
            fiscal_year=2024,
            concept_key="revenue",
        ),
    )
    # Seeded AAPL FY2024 revenue is $391,035M USD. Validate must emit the
    # exact value to pass cite_check.
    _patch_validate(
        monkeypatch,
        _ValidateLLMResponse(
            draft_answer="Apple's FY2024 revenue was $391,035 million.",
            claims=[
                _ClaimByIdx(
                    text="Apple's FY2024 revenue was $391,035 million.",
                    kind="number",
                    source_idx=0,
                )
            ],
        ),
    )

    out = await GRAPH.ainvoke(
        AgentState(
            question="What was Apple's revenue in FY2024?",
            request_id="t-rev",
        )
    )
    final = AgentState.model_validate(out)

    assert final.question_type == "A_numerical"
    assert final.cik == "0000320193"
    assert final.fiscal_year == 2024
    assert final.xbrl_fact is not None
    assert final.xbrl_fact.value == Decimal("391035000000")
    assert final.verified is True
    assert final.final_answer is not None
    assert "391,035,000,000" in final.final_answer


async def test_aapl_supply_chain_qualitative_path(monkeypatch, agent_session: AsyncSession):
    """Type B: real vector_search returns 5 chunks; we stub it to a fixed
    chunk so we can mock validate deterministically."""
    fixed_chunk = ChunkHit(
        chunk_id=999,
        accession_no="0000320193-24-000123",
        item="1A",
        seq=0,
        text=(
            "We depend on a limited number of suppliers, primarily in Asia, "
            "for components and assembly. A disruption in supply could harm "
            "our business."
        ),
        score=0.05,
        char_start=0,
        char_end=160,
    )

    async def fake_vector_search(*, cik, item, question, session, k=5, accession_no=None):
        return [fixed_chunk]

    monkeypatch.setattr(retrieve_module, "vector_search", fake_vector_search)

    _patch_plan(
        monkeypatch,
        PlanResult(
            question_type="B_qualitative",
            ticker="AAPL",
            item="1A",
        ),
    )
    # Use a substring that appears in fixed_chunk.text → cite_check passes.
    _patch_validate(
        monkeypatch,
        _ValidateLLMResponse(
            draft_answer="Apple depends on a limited number of suppliers.",
            claims=[
                _ClaimByIdx(
                    text="depend on a limited number of suppliers",
                    kind="prose",
                    source_idx=0,
                )
            ],
        ),
    )

    out = await GRAPH.ainvoke(
        AgentState(
            question="What does Apple say about supply chain risk?",
            request_id="t-supply",
        )
    )
    final = AgentState.model_validate(out)

    assert final.question_type == "B_qualitative"
    assert final.item == "1A"
    assert len(final.chunks) == 1
    assert final.verified is True
    assert "suppliers" in (final.final_answer or "")


async def test_unknown_ticker_refuses_at_fetch(monkeypatch, agent_session: AsyncSession):
    """Plan classifies fine; fetch refuses because FAKE isn't in our seeded DB."""
    _patch_plan(
        monkeypatch,
        PlanResult(
            question_type="A_numerical",
            ticker="FAKE",
            fiscal_year=2024,
            concept_key="rnd",
        ),
    )

    def boom(*a, **kw):
        raise AssertionError("validate should not be reached")

    monkeypatch.setattr(validate_module, "get_chat_model", boom)

    out = await GRAPH.ainvoke(
        AgentState(
            question="What was FakeCo's R&D in 2024?",
            request_id="t-fake",
        )
    )
    final = AgentState.model_validate(out)

    assert final.question_type == "REFUSE"
    assert "universe" in (final.refusal_reason or "")
    assert final.verified is False
    assert final.final_answer is None


async def test_compare_aapl_msft_refuses_at_plan(monkeypatch, agent_session: AsyncSession):
    """Plan node refuses cross-company comparisons. Nothing else runs."""
    _patch_plan(
        monkeypatch,
        PlanResult(
            question_type="REFUSE",
            refusal_reason="I answer questions about one company at a time.",
        ),
    )

    out = await GRAPH.ainvoke(
        AgentState(
            question="Compare AAPL and MSFT revenue",
            request_id="t-cmp",
        )
    )
    final = AgentState.model_validate(out)

    assert final.question_type == "REFUSE"
    assert "one company" in (final.refusal_reason or "")
    assert final.cik is None  # fetch never ran


async def test_cache_hit_short_circuits(monkeypatch, agent_session: AsyncSession):
    """Two identical questions: second one hits the cache and skips the LLM."""
    _patch_plan(
        monkeypatch,
        PlanResult(
            question_type="A_numerical",
            ticker="AAPL",
            fiscal_year=2024,
            concept_key="revenue",
        ),
    )
    _patch_validate(
        monkeypatch,
        _ValidateLLMResponse(
            draft_answer="Apple's FY2024 revenue was $391,035 million.",
            claims=[
                _ClaimByIdx(
                    text="Apple's FY2024 revenue was $391,035 million.",
                    kind="number",
                    source_idx=0,
                )
            ],
        ),
    )

    first = await GRAPH.ainvoke(
        AgentState(
            question="What was AAPL revenue in 2024?",
            request_id="t-c1",
        )
    )
    first_state = AgentState.model_validate(first)
    assert first_state.verified is True

    row = (await agent_session.execute(select(AnswerCache))).scalar_one_or_none()
    assert row is not None
    cached_answer = row.final_answer

    # Second call: validate's LLM must NOT be invoked.
    def boom(*a, **kw):
        raise AssertionError("LLM should not be called on cache hit")

    monkeypatch.setattr(validate_module, "get_chat_model", boom)

    second = await GRAPH.ainvoke(
        AgentState(
            question="What was AAPL revenue in 2024?",
            request_id="t-c2",
        )
    )
    second_state = AgentState.model_validate(second)
    assert second_state.cache_hit is True
    assert second_state.final_answer == cached_answer
