"""Validate node — LLM and DB mocked. Exercises cache hit/miss + claim mapping."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.agent.nodes import validate as validate_module
from app.agent.nodes import cite_check as cite_module
from app.agent.nodes.validate import _ClaimByIdx, _ValidateLLMResponse
from app.agent.state import AgentState, ChunkHit, XbrlFact


@pytest.fixture(autouse=True)
def structured_model_branch(monkeypatch):
    # Restored tests below cover the generic structured-output branch;
    # the Groq branch is explicitly selected in the new tests below.
    monkeypatch.setattr(validate_module.settings, "VALIDATE_MODEL", "test-structured-model")


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
        accession_no="acc",
        form="10-K",
    )


def _chunk(i: int) -> ChunkHit:
    return ChunkHit(
        chunk_id=i,
        accession_no="acc",
        item="1A",
        seq=i,
        text=f"chunk {i}",
        score=0.1,
        char_start=0,
        char_end=8,
    )


class _FakeStructured:
    def __init__(self, response, include_raw: bool = False):
        self._response = response
        self._include_raw = include_raw

    async def ainvoke(self, prompt):
        if self._include_raw:
            return {"raw": None, "parsed": self._response, "parsing_error": None}
        return self._response


class _FakeChat:
    def __init__(self, response):
        self._response = response

    def with_structured_output(self, schema, *, include_raw: bool = False, **_):
        return _FakeStructured(self._response, include_raw=include_raw)


def _patch_llm(monkeypatch, response):
    monkeypatch.setattr(validate_module, "get_chat_model", lambda *a, **kw: _FakeChat(response))


@asynccontextmanager
async def _no_cache_session():
    """Session that always misses the cache."""

    class _S:
        async def execute(self, stmt):
            return SimpleNamespace(scalar_one_or_none=lambda: None)

        async def commit(self):
            pass

    yield _S()


def _patch_no_cache(monkeypatch):
    monkeypatch.setattr(validate_module, "async_session_factory", _no_cache_session)


@pytest.mark.asyncio
async def test_validate_a_path_maps_claim_to_xbrl_fact(monkeypatch):
    _patch_no_cache(monkeypatch)
    fact = _xbrl_fact()
    _patch_llm(
        monkeypatch,
        _ValidateLLMResponse(
            draft_answer="Apple's revenue was $391,035 million.",
            claims=[
                _ClaimByIdx(
                    text="Apple's revenue was $391,035 million.", kind="number", source_idx=0
                )
            ],
        ),
    )
    state = AgentState(
        question="What was Apple's revenue?",
        request_id="r",
        question_type="A_numerical",
        cik="0000320193",
        fiscal_year=2024,
        xbrl_fact=fact,
    )
    out = await validate_module.validate(state)
    assert out["draft_answer"].startswith("Apple's revenue")
    assert len(out["claims"]) == 1
    assert out["claims"][0].source_ref is fact


@pytest.mark.asyncio
async def test_validate_b_path_maps_claims_to_chunks(monkeypatch):
    _patch_no_cache(monkeypatch)
    chunks = [_chunk(i) for i in range(3)]
    _patch_llm(
        monkeypatch,
        _ValidateLLMResponse(
            draft_answer="Apple cites supply chain risk.",
            claims=[_ClaimByIdx(text="Apple cites supply chain risk.", kind="prose", source_idx=2)],
        ),
    )
    state = AgentState(
        question="risk?",
        request_id="r",
        question_type="B_qualitative",
        cik="0000320193",
        item="1A",
        chunks=chunks,
    )
    out = await validate_module.validate(state)
    assert out["claims"][0].source_ref is chunks[2]


@pytest.mark.asyncio
async def test_validate_empty_claims_signals_refusal(monkeypatch):
    _patch_no_cache(monkeypatch)
    _patch_llm(monkeypatch, _ValidateLLMResponse(draft_answer="", claims=[]))
    state = AgentState(
        question="q",
        request_id="r",
        question_type="A_numerical",
        cik="0000320193",
        fiscal_year=2024,
        xbrl_fact=_xbrl_fact(),
    )
    out = await validate_module.validate(state)
    assert out["draft_answer"] == ""
    assert out["claims"] == []


@pytest.mark.asyncio
async def test_validate_cache_hit_short_circuits(monkeypatch):
    """If the cache has a row, validate must return it without calling the LLM."""
    cached_row = SimpleNamespace(
        cache_key="k",
        final_answer="cached answer",
        claims_json=[
            {"text": "cached", "kind": "prose", "source_ref": None},
        ],
        hit_count=0,
    )

    @asynccontextmanager
    async def session_with_hit():
        class _S:
            async def execute(self, stmt):
                return SimpleNamespace(scalar_one_or_none=lambda: cached_row)

            async def commit(self):
                pass

        yield _S()

    monkeypatch.setattr(validate_module, "async_session_factory", session_with_hit)

    # If the LLM is called it'll explode — proves cache short-circuits.
    def _no_llm(*a, **kw):
        raise AssertionError("LLM should not be called on cache hit")

    monkeypatch.setattr(validate_module, "get_chat_model", _no_llm)

    state = AgentState(
        question="q",
        request_id="r",
        question_type="B_qualitative",
        cik="0000320193",
        item="1A",
        chunks=[_chunk(0)],
    )
    out = await validate_module.validate(state)
    assert out["cache_hit"] is True
    assert out["final_answer"] == "cached answer"
    assert out["verified"] is True


@pytest.mark.asyncio
async def test_validate_passes_through_refuse():
    state = AgentState(question="q", request_id="r", question_type="REFUSE")
    assert await validate_module.validate(state) == {}


def test_cache_key_changes_with_model(monkeypatch):
    """Cache key must include model names — overrides shouldn't collide."""
    state = AgentState(
        question="q",
        request_id="r",
        question_type="A_numerical",
        cik="c",
        fiscal_year=2024,
        xbrl_fact=_xbrl_fact(),
    )
    monkeypatch.setattr(validate_module.settings, "VALIDATE_MODEL", "gpt-4o")
    k1 = validate_module._compute_cache_key(state)
    monkeypatch.setattr(validate_module.settings, "VALIDATE_MODEL", "gpt-4o-mini")
    k2 = validate_module._compute_cache_key(state)
    assert k1 != k2


@pytest.mark.asyncio
@pytest.mark.parametrize("value,unit,concept,expected", [
    ("391035000000.0000", "USD", "revenue", "AAPL's revenue was $391,035,000,000."),
    ("-2000000000", "USD", "net_income", "AAPL's net income was $-2,000,000,000."),
    ("6.13", "USD/shares", "eps_diluted", "AAPL's eps diluted was $6.13 per share."),
    ("15000000000", "shares", "shares_out", "AAPL's shares out was 15,000,000,000 shares."),
])
async def test_groq_numeric_path_uses_exact_fact_without_llm(monkeypatch, value, unit, concept, expected):
    _patch_no_cache(monkeypatch)
    def no_llm(*args, **kwargs):
        pytest.fail("A structured numerical fact must not trigger an LLM call")
    monkeypatch.setattr(validate_module, "get_chat_model", no_llm)
    fact = _xbrl_fact().model_copy(update={"value": Decimal(value), "unit": unit, "concept_key": concept})
    state = AgentState(question="What was the value?", request_id="r", ticker="AAPL",
                       question_type="A_numerical", cik="0000320193", fiscal_year=2024,
                       xbrl_fact=fact, validate_model_override="openai/gpt-oss-120b")
    out = await validate_module.validate(state)
    assert out["draft_answer"] == expected
    assert len(out["claims"]) == 1
    assert out["claims"][0].source_ref is fact
    assert out["claims"][0].kind == "number"
    async def no_cache_write(*args, **kwargs):
        pass
    monkeypatch.setattr(cite_module, "write_to_cache", no_cache_write)
    verified = await cite_module.cite_check(state.model_copy(update=out))
    assert verified["verified"] is True
    assert "[1]" in verified["final_answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("index", [-1, 99])
async def test_validate_invalid_source_index_never_points_to_another_chunk(monkeypatch, index):
    _patch_no_cache(monkeypatch)
    _patch_llm(monkeypatch, _ValidateLLMResponse(draft_answer="Unsupported sentence.",
               claims=[_ClaimByIdx(text="Unsupported sentence.", kind="prose", source_idx=index)]))
    state = AgentState(question="risks?", request_id="r", question_type="B_qualitative",
                       cik="0000320193", item="1A", chunks=[_chunk(0)])
    out = await validate_module.validate(state)
    assert out["claims"][0].source_ref is None


def test_cache_key_changes_for_different_filings():
    first = AgentState(question="revenue?", request_id="r", question_type="A_numerical",
                       cik="0000320193", fiscal_year=2024, xbrl_fact=_xbrl_fact())
    second = first.model_copy(update={"fiscal_year": 2025,
        "xbrl_fact": _xbrl_fact().model_copy(update={"accession_no": "ACC-2025"})})
    assert validate_module._compute_cache_key(first) != validate_module._compute_cache_key(second)


def test_cache_key_uses_rewritten_followup_question():
    first = AgentState(question="AAPL revenue in 2024?", request_id="r", question_type="A_numerical",
                       cik="0000320193", fiscal_year=2024, xbrl_fact=_xbrl_fact())
    followup = first.model_copy(update={"question": "and 2024?",
                                       "rewritten_question": first.question})
    assert validate_module._compute_cache_key(first) == validate_module._compute_cache_key(followup)
