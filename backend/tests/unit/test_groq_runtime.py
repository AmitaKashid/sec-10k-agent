from contextlib import asynccontextmanager
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import groq
import httpx
import pytest
from langchain_core.messages import AIMessage

from app.agent import groq_runtime as runtime
from app.agent.nodes import cite_check, plan, refusal_composer, validate
from app.agent.state import AgentState, ChunkHit, PlanResult


def chunk(text, idx=1):
    return ChunkHit(
        chunk_id=idx,
        accession_no="acc",
        item="1A",
        seq=idx,
        text=text,
        score=0.1,
        char_start=0,
        char_end=len(text),
    )


def state():
    return AgentState(
        question="What are the risks?",
        request_id="test",
        question_type="B_qualitative",
        ticker="AAPL",
        validate_model_override="openai/gpt-oss-120b",
        chunks=[
            chunk(
                "Changing suppliers can be expensive, time-consuming and disruptive to operations."
            )
        ],
    )


class Chat:
    def __init__(self, content='{"selected_ids":[0]}', finish="stop", error=None):
        self.content, self.finish, self.error = content, finish, error
        self.params = None

    def bind(self, **kwargs):
        self.params = kwargs
        return self

    async def ainvoke(self, prompt):
        if self.error:
            raise self.error
        return AIMessage(content=self.content, response_metadata={"finish_reason": self.finish})


@asynccontextmanager
async def no_cache():
    yield SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: None))
    )


def test_strict_schema_requires_nullable_fields_and_nested_objects():
    schema = runtime.strict_schema(PlanResult)
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False
    assert "default" not in schema["properties"]["ticker"]


@pytest.mark.asyncio
async def test_selection_copies_exact_source_and_passes_citation():
    s = state()
    claims = await runtime.select_evidence(Chat(), s)
    assert claims[0].text == s.chunks[0].text
    assert claims[0].evidence_quote == s.chunks[0].text
    assert "time-consistent" not in claims[0].text
    s.claims = claims
    result = await cite_check.cite_check(s)
    assert result["verified"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content,finish",
    [
        ('{"selected_ids":[0]}', "length"),
        ('{"selected_ids":[0]}', "content_filter"),
        ("not json", "stop"),
        ('{"selected_ids":[99]}', "stop"),
        ('{"selected_ids":[-1]}', "stop"),
        ('{"selected_ids":[0,0]}', "stop"),
        ('{"selected_ids":[0,1,2,3]}', "stop"),
    ],
)
async def test_bad_generation_becomes_refusal(monkeypatch, content, finish):
    monkeypatch.setattr(validate, "async_session_factory", no_cache)
    monkeypatch.setattr(validate, "get_chat_model", lambda *a, **kw: Chat(content, finish))
    result = await validate.validate(state())
    assert result["question_type"] == "REFUSE"
    assert result["refusal_code"] == "INTERNAL_ERROR"
    assert result["claims"] == []
    assert result["verified"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 413, 429, 503])
async def test_provider_errors_do_not_escape_validator(monkeypatch, status):
    error = groq.APIStatusError(
        "provider failure",
        response=httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com/test")),
        body={},
    )
    monkeypatch.setattr(validate, "async_session_factory", no_cache)
    monkeypatch.setattr(validate, "get_chat_model", lambda *a, **kw: Chat(error=error))
    result = await validate.validate(state())
    assert result["refusal_code"] == "INTERNAL_ERROR"


@pytest.mark.asyncio
async def test_plan_failure_resets_state_and_skips_refusal_model(monkeypatch):
    s = state()
    s.plan_model_override = "openai/gpt-oss-20b"
    monkeypatch.setattr(plan, "get_chat_model", lambda *a, **kw: Chat(finish="length"))
    delta = await plan.plan(s)
    s = s.model_copy(update=delta)
    monkeypatch.setattr(
        refusal_composer, "get_chat_model", lambda *a, **kw: pytest.fail("extra API call")
    )
    out = await refusal_composer.refusal_composer(s)
    assert out["refusal_message"]
    assert s.claims == []


def test_candidate_input_is_bounded_and_exact():
    text = " ".join(
        f"The supplier faces serious operational disruption in region {i}." for i in range(100)
    )
    candidates = runtime.evidence_candidates([chunk(text, i) for i in range(20)])
    assert len(candidates) <= 16
    assert sum(len(t.encode("utf-8")) for _, t in candidates) <= 9000
    assert all(t in c.text for c, t in candidates)


@pytest.mark.asyncio
async def test_empty_selection_is_not_verified(monkeypatch):
    monkeypatch.setattr(validate, "async_session_factory", no_cache)
    monkeypatch.setattr(validate, "get_chat_model", lambda *a, **kw: Chat('{"selected_ids":[]}'))
    s = state()
    s = s.model_copy(update=await validate.validate(s))
    result = await cite_check.cite_check(s)
    assert result["verified"] is False


@pytest.mark.parametrize("text", ["-$1,200", "$-1,200", "-1200"])
def test_negative_values_preserve_sign(text):
    assert cite_check._parse_money(text) == Decimal("-1200")


@pytest.mark.asyncio
async def test_partial_answer_drops_claims_from_response():
    s = state()
    from app.agent.state import Claim

    good = Claim(
        text=s.chunks[0].text, kind="prose", evidence_quote=s.chunks[0].text, source_ref=s.chunks[0]
    )
    bad = Claim(
        text="Made up sentence.",
        kind="prose",
        evidence_quote="A fabricated quote.",
        source_ref=s.chunks[0],
    )
    s.claims = [good, good.model_copy(), bad]
    result = await cite_check.cite_check(s)
    assert result["verified"]
    assert len(result["claims"]) == 2
    assert bad not in result["claims"]


@pytest.mark.asyncio
async def test_real_graph_returns_refusal_for_provider_failure(monkeypatch):
    from app.agent.graph import build_graph

    monkeypatch.setattr(plan, "get_chat_model", lambda *a, **kw: Chat(finish="length"))
    out = await build_graph().ainvoke(
        AgentState(
            question="What are Apple risks?",
            request_id="graph-error",
            plan_model_override="openai/gpt-oss-20b",
        )
    )
    assert out["question_type"] == "REFUSE"
    assert out["refusal_message"]
    assert out["verified"] is False


@pytest.mark.asyncio
async def test_external_error_in_retrieval_is_handled():
    from app.agent.graph import _instrument

    async def broken(state):
        raise httpx.ReadTimeout("timeout")

    out = await _instrument("retrieve", broken)(state())
    assert out["refusal_code"] == "INTERNAL_ERROR"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unit,value", [("USD", "-1200"), ("USD/shares", "6.11"), ("shares", "12"), ("pure", "0.5")]
)
async def test_structured_fact_needs_no_llm_and_passes_citation(monkeypatch, unit, value):
    from datetime import date

    from app.agent.state import XbrlFact

    s = state()
    s.question_type = "A_numerical"
    s.xbrl_fact = XbrlFact(
        cik="0000320193",
        concept="example",
        concept_key="example",
        unit=unit,
        value=Decimal(value),
        period_end=date(2025, 9, 30),
        fiscal_year=2025,
        fiscal_period="FY",
        accession_no="acc",
        form="10-K",
    )
    monkeypatch.setattr(validate, "async_session_factory", no_cache)
    monkeypatch.setattr(
        validate, "get_chat_model", lambda *a, **kw: pytest.fail("unnecessary model call")
    )
    monkeypatch.setattr(cite_check, "write_to_cache", AsyncMock())
    s = s.model_copy(update=await validate.validate(s))
    out = await cite_check.cite_check(s)
    assert out["verified"] is True
