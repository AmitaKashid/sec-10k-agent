"""HTTP-layer schemas. Reuses agent.state models per CLAUDE.md.

Only request/response envelopes live here; domain types (Claim, ChunkHit,
FilingRef) come from `app.agent.state` and are not duplicated.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.agent.state import Claim, QuestionType
from app.settings import settings


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # Conversation thread id. When the server's checkpointer is enabled
    # and the caller supplies a `thread_id`, prior turns in that thread
    # are available to the `plan` node for follow-up resolution
    # ("and 2024?"). Omitted → fresh thread, no history. See
    # docs/conversation_memory_plan.md §9.
    # Constrained charset: thread_id becomes a LangGraph checkpointer
    # partition key. Reject control chars / newlines / unbounded
    # cardinality input — accept only UUID-ish/slug tokens.
    thread_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,128}$")
    plan_model_override: str | None = None
    validate_model_override: str | None = None

    @field_validator("plan_model_override", "validate_model_override")
    @classmethod
    def _check_model_allowlist(cls, v: str | None) -> str | None:
        # Caller-supplied model names must be on the operator allowlist —
        # otherwise an attacker with a valid key can bill arbitrary models.
        if v is not None and v not in settings.ALLOWED_MODELS:
            raise ValueError(f"model not in ALLOWED_MODELS: {v}")
        return v


class AskResponse(BaseModel):
    """Sync (`Accept: application/json`) response — terminal payload only.

    Mirrors the SSE `done` event so callers can use the same parsing path
    in eval/curl mode and in streaming mode.
    """

    request_id: str
    # Echoed back so the client can pin the thread id chosen server-side
    # (e.g. when a request omitted `thread_id` and the server generated one).
    thread_id: str | None = None
    question_type: QuestionType | None = None
    answer: str | None = None
    claims: list[Claim] = Field(default_factory=list)
    verified: bool = False
    refused: bool = False
    refusal_reason: str | None = None
    refusal_code: str | None = None
    refusal_message: str | None = None
    meta_message: str | None = None
    ticker: str | None = None
    fiscal_year: int | None = None
    # Langfuse trace deep-link. Populated only when EXPOSE_TRACE_URLS is
    # true (dev/demo) and Langfuse is configured; null otherwise.
    trace_url: str | None = None


class CompanyOut(BaseModel):
    ticker: str
    name: str
    sector: str | None = None
    latest_10k_period: int | None = None
    latest_10k_filed_at: str | None = None
    filings_count: int | None = None
    coverage_status: str | None = None


class CompanyStatsOut(BaseModel):
    issuers: int
    filings_indexed: int
    xbrl_facts: int
    coverage_complete: int
    coverage_total: int
    last_refresh: str | None = None


class HealthOut(BaseModel):
    status: str
    db: str
    graph: str


class SseEvent(BaseModel):
    """Documentation-only schema for SSE event payloads. Not directly served."""

    event: str  # "node" | "done" | "error"
    data: dict[str, Any]


__all__ = [
    "AskRequest",
    "AskResponse",
    "CompanyOut",
    "CompanyStatsOut",
    "HealthOut",
    "SseEvent",
]
