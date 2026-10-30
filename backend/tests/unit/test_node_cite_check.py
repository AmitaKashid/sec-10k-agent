"""Cite-check — the structural anti-hallucination gate.

Number rule: parse decimal from claim.text → exact equality with
source_ref.value (no tolerance).
Prose rule: substring (case-insensitive) OR cosine ≥ 0.85.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.agent.nodes import cite_check as cite_module
from app.agent.nodes.cite_check import _parse_money
from app.agent.state import AgentState, ChunkHit, Claim, XbrlFact


def _xbrl_fact(value: str = "391035000000") -> XbrlFact:
    return XbrlFact(
        cik="0000320193",
        concept="revenue",
        concept_key="revenue",
        unit="USD",
        value=Decimal(value),
        period_end=date(2024, 9, 28),
        fiscal_year=2024,
        fiscal_period="FY",
        accession_no="acc",
        form="10-K",
    )


def _chunk(text: str, chunk_id: int = 1) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        accession_no="acc",
        item="1A",
        seq=0,
        text=text,
        score=0.1,
        char_start=0,
        char_end=len(text),
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("$391,035 million", Decimal("391035000000")),
        ("$391.035 billion", Decimal("391035000000")),
        ("391035000000", Decimal("391035000000")),
        ("391,035,000,000", Decimal("391035000000")),
        ("$6.13", Decimal("6.13")),
        ("approximately $391 billion", Decimal("391000000000")),  # rounded → fails check
        ("no number here", None),
    ],
)
def test_parse_money_handles_common_forms(text, expected):
    assert _parse_money(text) == expected


@pytest.mark.asyncio
async def test_number_exact_match_passes(monkeypatch):
    """LLM emitted the exact value — pass."""
    monkeypatch.setattr(cite_module, "write_to_cache", lambda *a, **kw: _async_noop())
    state = AgentState(
        question="q",
        request_id="r",
        question_type="A_numerical",
        draft_answer="Apple's revenue was $391,035 million.",
        cache_key="k",
        claims=[
            Claim(
                text="Apple's revenue was $391,035 million.",
                kind="number",
                source_ref=_xbrl_fact(),
            )
        ],
    )
    out = await cite_module.cite_check(state)
    assert out["verified"] is True
    # cite_check now appends 1-based [N] markers per verified claim
    assert out["final_answer"] == "Apple's revenue was $391,035 million [1]."


@pytest.mark.asyncio
async def test_number_rounded_value_fails():
    """The whole point: rounding must be caught."""
    state = AgentState(
        question="q",
        request_id="r",
        question_type="A_numerical",
        draft_answer="Apple's revenue was approximately $391 billion.",
        cache_key="k",
        claims=[
            Claim(
                text="Apple's revenue was approximately $391 billion.",
                kind="number",
                source_ref=_xbrl_fact("391035000000"),
            )
        ],
    )
    out = await cite_module.cite_check(state)
    assert out["verified"] is False
    assert "couldn't verify" in out["refusal_reason_final"]
    assert len(out["failed_claims"]) == 1


@pytest.mark.asyncio
async def test_number_with_no_source_fails():
    state = AgentState(
        question="q",
        request_id="r",
        question_type="A_numerical",
        draft_answer="x",
        cache_key="k",
        claims=[Claim(text="$1M", kind="number", source_ref=None)],
    )
    out = await cite_module.cite_check(state)
    assert out["verified"] is False


@pytest.mark.asyncio
async def test_prose_evidence_quote_passes_without_calling_embed(monkeypatch):
    """Primary path: verbatim quote substring match short-circuits embedding."""

    async def boom(_):
        raise AssertionError("embed should not be called when evidence_quote matches")

    monkeypatch.setattr(cite_module, "embed_texts", boom)
    monkeypatch.setattr(cite_module, "write_to_cache", lambda *a, **kw: _async_noop())
    chunk_text = (
        "The Company relies on a limited number of suppliers based in Asia. "
        "Component shortages could materially affect operations."
    )
    state = AgentState(
        question="q",
        request_id="r",
        question_type="B_qualitative",
        draft_answer="Apple depends on Asian suppliers.",
        cache_key="k",
        claims=[
            Claim(
                text="Apple depends on Asian suppliers.",  # paraphrase
                kind="prose",
                source_ref=_chunk(chunk_text),
                evidence_quote="limited number of suppliers based in Asia",
            )
        ],
    )
    out = await cite_module.cite_check(state)
    assert out["verified"] is True


@pytest.mark.asyncio
async def test_prose_evidence_quote_whitespace_normalized(monkeypatch):
    """Whitespace drift in the quote is tolerated; other characters must match."""

    async def boom(_):
        raise AssertionError("embed should not be called")

    monkeypatch.setattr(cite_module, "embed_texts", boom)
    monkeypatch.setattr(cite_module, "write_to_cache", lambda *a, **kw: _async_noop())
    chunk_text = "Risk factors include\nsupply chain   disruption and demand swings."
    state = AgentState(
        question="q",
        request_id="r",
        question_type="B_qualitative",
        draft_answer="A.",
        cache_key="k",
        claims=[
            Claim(
                text="A.",
                kind="prose",
                source_ref=_chunk(chunk_text),
                # Same letters, different whitespace runs.
                evidence_quote="supply chain disruption and demand swings",
            )
        ],
    )
    out = await cite_module.cite_check(state)
    assert out["verified"] is True


@pytest.mark.asyncio
async def test_prose_evidence_quote_not_in_chunk_fails(monkeypatch):
    """Fabricated quote that is not in the chunk → fail closed, no embed call."""

    async def boom(_):
        raise AssertionError("embed should not be called when evidence_quote is checked")

    monkeypatch.setattr(cite_module, "embed_texts", boom)
    state = AgentState(
        question="q",
        request_id="r",
        question_type="B_qualitative",
        draft_answer="A.",
        cache_key="k",
        claims=[
            Claim(
                text="A.",
                kind="prose",
                source_ref=_chunk("Some real source text about supply chains."),
                evidence_quote="invented sentence the source never said",
            )
        ],
    )
    out = await cite_module.cite_check(state)
    assert out["verified"] is False
    assert any("evidence_quote" in f for f in out["trace_events"][-1]["failures"])


@pytest.mark.asyncio
async def test_prose_ellipsis_split_same_chunk_passes(monkeypatch):
    """LLM stitches two contiguous spans from the SAME chunk with '...'.
    Fallback splits, requires each fragment ≥30 chars verbatim in one chunk.
    Should pass.
    """

    async def boom(_):
        raise AssertionError("embed should not be called when fallback hits")

    monkeypatch.setattr(cite_module, "embed_texts", boom)
    monkeypatch.setattr(cite_module, "write_to_cache", lambda *a, **kw: _async_noop())
    chunk_text = (
        "Restrictions on international trade can be announced with little or no advance notice. "
        "The Company may not be able to mitigate all adverse impacts. "
        "Restrictive measures can require the Company to take various actions, "
        "including changing suppliers and restructuring business relationships."
    )
    state = AgentState(
        question="q",
        request_id="r",
        question_type="B_qualitative",
        draft_answer="A.",
        cache_key="k",
        chunks=[_chunk(chunk_text, chunk_id=1)],
        claims=[
            Claim(
                text="A.",
                kind="prose",
                source_ref=_chunk(chunk_text, chunk_id=1),
                evidence_quote=(
                    "Restrictions on international trade can be announced with little or no advance notice"
                    " ... "
                    "Restrictive measures can require the Company to take various actions"
                ),
            )
        ],
    )
    out = await cite_module.cite_check(state)
    assert out["verified"] is True


@pytest.mark.asyncio
async def test_prose_ellipsis_split_cross_chunk_fails(monkeypatch):
    """Fragments span TWO different chunks. Must FAIL — fallback requires
    all fragments to come from the same chunk to preserve provenance."""

    async def boom(_):
        raise AssertionError("embed should not be called when fallback runs")

    monkeypatch.setattr(cite_module, "embed_texts", boom)
    chunk_a = "The Company depends on component manufacturing partners located outside the U.S."
    chunk_b = (
        "Restrictions on international trade can be announced with little or no advance notice."
    )
    state = AgentState(
        question="q",
        request_id="r",
        question_type="B_qualitative",
        draft_answer="A.",
        cache_key="k",
        chunks=[
            _chunk(chunk_a, chunk_id=1),
            _chunk(chunk_b, chunk_id=2),
        ],
        claims=[
            Claim(
                text="A.",
                kind="prose",
                source_ref=_chunk(chunk_a, chunk_id=1),
                evidence_quote=(
                    "The Company depends on component manufacturing partners located outside the U.S."
                    " ... "
                    "Restrictions on international trade can be announced with little or no advance notice"
                ),
            )
        ],
    )
    out = await cite_module.cite_check(state)
    assert out["verified"] is False
    assert any("not verbatim" in f for f in out["trace_events"][-1]["failures"])


@pytest.mark.asyncio
async def test_prose_ellipsis_split_short_fragments_ignored(monkeypatch):
    """All fragments below _MIN_FRAGMENT_LEN should be filtered out, leaving
    fewer than 2 usable fragments → fallback skipped → fail."""

    async def boom(_):
        raise AssertionError("embed should not be called")

    monkeypatch.setattr(cite_module, "embed_texts", boom)
    chunk_text = (
        "The Company is large. It is global. It has many partners. It supplies many products."
    )
    state = AgentState(
        question="q",
        request_id="r",
        question_type="B_qualitative",
        draft_answer="A.",
        cache_key="k",
        chunks=[_chunk(chunk_text, chunk_id=1)],
        claims=[
            Claim(
                text="A.",
                kind="prose",
                source_ref=_chunk(chunk_text, chunk_id=1),
                # Each fragment is < 30 chars after whitespace collapse — fallback rejects.
                evidence_quote="It is global ... It has many",
            )
        ],
    )
    out = await cite_module.cite_check(state)
    assert out["verified"] is False


@pytest.mark.asyncio
async def test_prose_substring_passes(monkeypatch):
    """Substring match (case-insensitive) avoids the embedding call entirely."""

    async def boom(_):  # if reached, embedder was wrongly invoked
        raise AssertionError("substring should have short-circuited")

    monkeypatch.setattr(cite_module, "embed_texts", boom)
    monkeypatch.setattr(cite_module, "write_to_cache", lambda *a, **kw: _async_noop())
    chunk_text = "We depend on a limited number of suppliers in Asia."
    state = AgentState(
        question="q",
        request_id="r",
        question_type="B_qualitative",
        draft_answer="Apple depends on suppliers in Asia.",
        cache_key="k",
        claims=[
            Claim(
                text="suppliers in Asia",
                kind="prose",
                source_ref=_chunk(chunk_text),
            )
        ],
    )
    out = await cite_module.cite_check(state)
    assert out["verified"] is True


@pytest.mark.asyncio
async def test_prose_cosine_passes(monkeypatch):
    """No substring; cosine ≥ 0.85 saves the claim."""

    async def fake_embed(texts):
        # Return identical-direction vectors → cosine = 1.0
        return [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

    monkeypatch.setattr(cite_module, "embed_texts", fake_embed)
    monkeypatch.setattr(cite_module, "write_to_cache", lambda *a, **kw: _async_noop())
    state = AgentState(
        question="q",
        request_id="r",
        question_type="B_qualitative",
        draft_answer="A.",
        cache_key="k",
        claims=[
            Claim(
                text="paraphrase",
                kind="prose",
                source_ref=_chunk("totally different surface text"),
            )
        ],
    )
    out = await cite_module.cite_check(state)
    assert out["verified"] is True


@pytest.mark.asyncio
async def test_prose_low_cosine_fails(monkeypatch):
    async def fake_embed(texts):
        # Orthogonal → cosine = 0
        return [[1.0, 0.0], [0.0, 1.0]]

    monkeypatch.setattr(cite_module, "embed_texts", fake_embed)
    state = AgentState(
        question="q",
        request_id="r",
        question_type="B_qualitative",
        draft_answer="A.",
        cache_key="k",
        claims=[
            Claim(
                text="invented prose",
                kind="prose",
                source_ref=_chunk("totally unrelated source"),
            )
        ],
    )
    out = await cite_module.cite_check(state)
    assert out["verified"] is False


@pytest.mark.asyncio
async def test_no_claims_refuses():
    state = AgentState(
        question="q",
        request_id="r",
        question_type="A_numerical",
        cache_key="k",
        claims=[],
    )
    out = await cite_module.cite_check(state)
    assert out["verified"] is False
    assert "couldn't ground" in out["refusal_reason_final"]


@pytest.mark.asyncio
async def test_cache_hit_skips_check():
    state = AgentState(
        question="q",
        request_id="r",
        question_type="A_numerical",
        cache_hit=True,
    )
    out = await cite_module.cite_check(state)
    assert out["trace_events"][-1].get("skipped_cache_hit") is True


@pytest.mark.asyncio
async def test_passes_through_refuse():
    state = AgentState(question="q", request_id="r", question_type="REFUSE")
    assert await cite_module.cite_check(state) == {}


@pytest.mark.asyncio
async def test_prose_empty_chunk_text_fails_without_calling_embed(monkeypatch):
    """Empty chunk text must not be sent to the embedding provider.

    `char_start`/`char_end` are document-level offsets used for UI
    citation highlighting and must NOT be used to re-slice `src.text`
    (doing so produced empty slices for every chunk past seq=0). The
    empty case we still need to guard against is a chunk whose own
    `text` is blank.
    """

    async def boom(_):
        raise AssertionError("embedder should not be called for empty inputs")

    monkeypatch.setattr(cite_module, "embed_texts", boom)
    chunk = ChunkHit(
        chunk_id=1,
        accession_no="acc",
        item="1A",
        seq=0,
        text="   ",  # whitespace-only chunk body
        score=0.1,
        char_start=0,
        char_end=3,
    )
    state = AgentState(
        question="q",
        request_id="r",
        question_type="B_qualitative",
        draft_answer="A.",
        cache_key="k",
        claims=[Claim(text="claim text", kind="prose", source_ref=chunk)],
    )
    out = await cite_module.cite_check(state)
    assert out["verified"] is False
    assert any("empty" in f for f in out["trace_events"][-1]["failures"])


@pytest.mark.asyncio
async def test_prose_empty_claim_fails_without_calling_embed(monkeypatch):
    async def boom(_):
        raise AssertionError("embedder should not be called for empty inputs")

    monkeypatch.setattr(cite_module, "embed_texts", boom)
    state = AgentState(
        question="q",
        request_id="r",
        question_type="B_qualitative",
        draft_answer="A.",
        cache_key="k",
        claims=[Claim(text="   ", kind="prose", source_ref=_chunk("real chunk text"))],
    )
    out = await cite_module.cite_check(state)
    assert out["verified"] is False
    assert any("empty" in f for f in out["trace_events"][-1]["failures"])


# helper
async def _async_noop(*a, **kw):
    return None
