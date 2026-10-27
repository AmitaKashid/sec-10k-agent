"""Typed agent clipboard. See docs/lld.md §LLD 7.

Every node reads/writes this. Pydantic models also serve as the
structured-output schemas for `plan` and `validate` LLM calls.
"""

from __future__ import annotations

import operator
import re
from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.data.xbrl import XbrlFact  # re-export so nodes have one import

ItemKey = Literal["1A", "7", "8"]
QuestionType = Literal["A_numerical", "B_qualitative", "META", "REFUSE"]
# "META" is a non-grounded, non-refusal flow: greetings, capability
# questions, identity questions. Routed to the concierge node which
# describes the agent without answering any domain question. Distinct
# from REFUSE so evals don't conflate "user said hi" with "we couldn't
# ground an answer".

# Structured classification of *why* a refusal happened. Set alongside the
# free-form `refusal_reason` so evals/logs can branch deterministically and
# the refusal_composer node can shape user-facing prose by code.
RefusalCode = Literal[
    # plan-stage
    "OFF_TOPIC",  # not a 10-K-shaped question (weather, greetings, code, …)
    "NO_TICKER",  # no US-public-company ticker mentioned
    "ANON_TICKER_NOT_ALLOWED",  # anon tier: ticker outside the 5-ticker preview allowlist
    "YOY_NOT_SUPPORTED",  # YoY / growth / change-over-time question
    "CROSS_COMPANY",  # comparing two or more companies
    "UNSUPPORTED_CONCEPT",  # concept outside whitelist or Item outside {1A, 7, 8}
    "NON_10K_SOURCE",  # earnings call, 10-Q, 8-K, proxy, press release, etc.
    # fetch-stage
    "UNKNOWN_TICKER",  # ticker not in our 50-company universe
    "MISSING_FILING",  # ticker known, requested fiscal year not ingested
    # locate-stage
    "LOCATE_FAILED",  # requested Item section not parsed for the filing
    # retrieve-stage
    "NO_RETRIEVAL_MATCH",  # no XBRL fact / no relevant chunks
    # validate / cite_check
    "VALIDATE_REFUSED",  # validator returned no draft / empty claims
    "NUMERIC_TRIPWIRE",  # cite_check: claim number didn't match XBRL value
    "CLAIM_UNSUPPORTED",  # cite_check: claim has no citation or quote not verbatim
    # internal
    "INTERNAL_ERROR",  # missing required state, schema mismatch, etc.
]


class Turn(BaseModel):
    """Compact summary of one Q/A round. Accumulated in `AgentState.history`
    across turns of a single conversation thread (see
    `docs/conversation_memory_plan.md`). Deliberately small: no chunks, no
    XBRL facts, no claims — only the text answer and the resolved entity
    fields the next turn's `plan` node needs to disambiguate follow-ups.

    History is *never* a source of grounding. The presence of a `Turn` in
    state lets `plan` rewrite "and 2024?" into a self-contained question;
    every downstream node still retrieves and verifies citations fresh
    for the current turn.
    """

    question: str
    rewritten_question: str | None = None
    ticker: str | None = None
    fiscal_year: int | None = None
    concept_key: str | None = None
    item: ItemKey | None = None
    final_answer: str | None = None
    refusal_code: RefusalCode | None = None


def build_turn_from_state(state, final_answer: str | None) -> Turn:
    """Build a compact Turn record from the terminal state of a run.
    Called by `cite_check` / `refusal_composer` / `concierge` / the
    `validate` cache-hit path so the next turn's `plan` can see what
    was resolved here. Deliberately captures no chunks, claims, or
    XBRL facts — history is never a source of grounding."""
    return Turn(
        question=state.question,
        rewritten_question=state.rewritten_question,
        ticker=state.ticker,
        fiscal_year=state.fiscal_year,
        concept_key=state.concept_key,
        item=state.item,
        final_answer=final_answer,
        refusal_code=state.refusal_code,
    )


class FilingRef(BaseModel):
    accession_no: str
    fiscal_year: int
    period_of_report: date
    filing_date: date


class ChunkHit(BaseModel):
    chunk_id: int
    accession_no: str
    item: ItemKey
    seq: int
    text: str
    score: float  # cosine distance (lower = closer)
    char_start: int
    char_end: int
    # Denormalized metadata — populated from ingestion, None for legacy rows
    ticker: str | None = None
    company_name: str | None = None
    cik: str | None = None
    fiscal_year: int | None = None
    filing_date: date | None = None
    section_title: str | None = None


class Claim(BaseModel):
    """One sentence of draft_answer, paired with the source it claims to come from.

    `kind="number"`: source_ref must be an XbrlFact; cite_check parses a
    decimal from `text` and exact-equals it against `source_ref.value`.
    `kind="prose"`: source_ref must be a ChunkHit. The LLM emits a
    paraphrased `text` (the answer sentence) AND an `evidence_quote`
    that is a verbatim substring of the cited chunk; cite_check
    substring-matches `evidence_quote` against the chunk text. Cosine
    fallback remains as defense-in-depth on `text` when the quote
    cannot be located (e.g. light reformatting of whitespace).
    """

    text: str
    kind: Literal["number", "prose"]
    source_ref: ChunkHit | XbrlFact | None = None
    # Verbatim substring of the cited chunk that grounds the claim.
    # Required for prose claims (validate fills it from the LLM); not
    # used for number claims, where the XBRL value is the structural
    # ground truth.
    evidence_quote: str | None = None


class PlanResult(BaseModel):
    """Structured output of the plan LLM call."""

    question_type: QuestionType
    ticker: str | None = None
    fiscal_year: int | None = None  # null = "latest"
    concept_key: str | None = None
    item: ItemKey | None = None
    refusal_reason: str | None = None  # set iff question_type == "REFUSE"
    refusal_code: RefusalCode | None = None  # set iff question_type == "REFUSE"
    # Multi-turn follow-up resolution. When True, `rewritten_question`
    # holds a self-contained restatement of `state.question` that carries
    # forward the prior turn's ticker / fiscal year / concept as needed.
    # Every downstream node reads `state.rewritten_question or state.question`
    # so retrieval and validation run against the rewritten form.
    # See docs/conversation_memory_plan.md §8.
    is_followup: bool = False
    rewritten_question: str | None = None

    @field_validator("ticker")
    @classmethod
    def _validate_ticker(cls, v: str | None) -> str | None:
        if v is not None and not re.fullmatch(r"[A-Z]{1,5}", v.upper()):
            return None
        return v.upper() if v else v

    @model_validator(mode="after")
    def _followup_must_be_self_contained(self) -> PlanResult:
        # A follow-up rewrite is only valid if it contains a resolvable
        # ticker and a non-empty rewritten_question. Otherwise downgrade
        # to standalone — downstream nodes will treat it as fresh, which
        # is the correct failure mode (likely a NO_TICKER refusal).
        if self.is_followup and (not self.rewritten_question or not self.ticker):
            self.is_followup = False
            self.rewritten_question = None
        return self


class ValidateResult(BaseModel):
    """Structured output of the validate LLM call."""

    draft_answer: str
    claims: list[Claim] = Field(default_factory=list)


class AgentState(BaseModel):
    """The clipboard. Each node returns a dict that gets merged into this."""

    # input
    question: str
    request_id: str
    # Caller tier — "user" (signed in via Clerk) or "anon" (IP-keyed).
    # Defaulted to "user" so internal callers and existing tests that
    # construct AgentState directly bypass anon-only gates. The /ask
    # route handler always sets this explicitly from Identity.tier.
    tier: Literal["anon", "user"] = "user"
    # per-request overrides — fall back to settings.PLAN_MODEL / VALIDATE_MODEL
    plan_model_override: str | None = None
    validate_model_override: str | None = None

    # plan
    question_type: QuestionType | None = None
    ticker: str | None = None
    cik: str | None = None
    fiscal_year: int | None = None
    concept_key: str | None = None
    item: ItemKey | None = None
    refusal_reason: str | None = None
    refusal_code: RefusalCode | None = None
    # Follow-up resolution: when plan detects a multi-turn follow-up,
    # `rewritten_question` is the self-contained restatement of
    # `question`. Downstream readers should prefer
    # `state.rewritten_question or state.question`.
    is_followup: bool = False
    rewritten_question: str | None = None

    # fetch / locate
    filing: FilingRef | None = None
    located: bool = False
    # Years we DO have ingested for the resolved CIK. Populated by fetch
    # on MISSING_FILING so the refusal_composer can suggest alternatives.
    available_fiscal_years: list[int] = Field(default_factory=list)

    # retrieve
    xbrl_fact: XbrlFact | None = None
    chunks: list[ChunkHit] = Field(default_factory=list)

    # cache
    cache_hit: bool = False
    cache_key: str | None = None

    # validate
    draft_answer: str | None = None
    claims: list[Claim] = Field(default_factory=list)

    # cite_check
    verified: bool = False
    failed_claims: list[Claim] = Field(default_factory=list)
    final_answer: str | None = None
    refusal_reason_final: str | None = None
    # Note: `refusal_code` is declared once above (under plan); cite_check
    # writes into the same field — Pydantic merges the delta on top.

    # refusal_composer (last node before END on the refuse path)
    # User-facing prose. `refusal_reason` stays as the deterministic short
    # hint for evals and logs; `refusal_message` is what the UI displays.
    refusal_message: str | None = None

    # concierge (META path: greetings, capability/identity questions).
    # User-facing prose for non-refusal, non-grounded conversation about
    # the agent itself. Streams tokens like refusal_message.
    meta_message: str | None = None

    # observability — accumulated SSE events, forwarded by the FastAPI route
    trace_events: list[dict] = Field(default_factory=list)

    # Multi-turn conversation history. Concatenated across turns via the
    # `operator.add` reducer when a LangGraph checkpointer is wired
    # (see docs/conversation_memory_plan.md §5–§6). Without a
    # checkpointer this is always an empty list — every request is a
    # fresh thread and no Turn is ever observed by `plan`.
    history: Annotated[list[Turn], operator.add] = Field(default_factory=list)


# Custom (non-stdlib) types that can appear inside a checkpointed
# AgentState. LangGraph's msgpack serde refuses to deserialize
# unregistered types (warns now, hard-blocks in a future version), so
# the AsyncPostgresSaver in app.main must be constructed with a serde
# that explicitly allows these. Listed as concrete classes (not a
# wildcard module) so the allowlist stays tight. XbrlFact is included
# because it rides on AgentState.xbrl_fact mid-graph before
# RESET_DEFAULTS clears it.
CHECKPOINTED_TYPES: tuple[type, ...] = (
    AgentState,
    Turn,
    FilingRef,
    ChunkHit,
    Claim,
    XbrlFact,
)


# Per-turn working-memory fields. When the checkpointer carries state
# across turns, `plan` returns these as part of its delta to wipe stale
# values from the previous turn (chunks, claims, final_answer, etc.).
# `question` / `request_id` are set fresh by the caller each turn;
# `history` is intentionally absent — it accumulates via reducer.
# Not applied yet — gated behind LANGGRAPH_CHECKPOINTER_ENABLED. See
# docs/conversation_memory_plan.md §7.
RESET_DEFAULTS: dict = {
    "question_type": None,
    "ticker": None,
    "cik": None,
    "fiscal_year": None,
    "concept_key": None,
    "item": None,
    "refusal_reason": None,
    "refusal_code": None,
    "is_followup": False,
    "rewritten_question": None,
    "filing": None,
    "located": False,
    "available_fiscal_years": [],
    "xbrl_fact": None,
    "chunks": [],
    "cache_hit": False,
    "cache_key": None,
    "draft_answer": None,
    "claims": [],
    "verified": False,
    "failed_claims": [],
    "final_answer": None,
    "refusal_reason_final": None,
    "refusal_message": None,
    "meta_message": None,
    "trace_events": [],
}


__all__ = [
    "CHECKPOINTED_TYPES",
    "RESET_DEFAULTS",
    "AgentState",
    "ChunkHit",
    "Claim",
    "FilingRef",
    "ItemKey",
    "PlanResult",
    "QuestionType",
    "RefusalCode",
    "Turn",
    "ValidateResult",
    "XbrlFact",
    "build_turn_from_state",
]
