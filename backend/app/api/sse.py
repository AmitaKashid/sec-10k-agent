"""SSE event shaping for `/ask`.

Streaming policy: only `event: status` (compact `{step}`) and a terminal
`event: done` carrying the verified answer. Per-node state is never put
on the wire — cite_check has not run by validate, so any intermediate
answer would be unverified, which contradicts the product's anti-
hallucination guarantee.
"""

from __future__ import annotations

from typing import Any


def terminal_payload(
    state: dict[str, Any],
    request_id: str,
    thread_id: str | None = None,
    trace_url: str | None = None,
) -> dict[str, Any]:
    """Build the `done` event payload from a terminal AgentState dict."""
    question_type = state.get("question_type")
    refused = question_type == "REFUSE" or bool(state.get("refusal_reason_final"))
    refusal_reason = state.get("refusal_reason_final") or state.get("refusal_reason")
    refusal_code = state.get("refusal_code")
    refusal_message = state.get("refusal_message")
    meta_message = state.get("meta_message")
    answer = state.get("final_answer") or state.get("draft_answer")
    claims = state.get("claims") or []
    claims_out = [c.model_dump(mode="json") if hasattr(c, "model_dump") else c for c in claims]
    return {
        "request_id": request_id,
        "thread_id": thread_id,
        "question_type": question_type,
        "answer": None if refused else answer,
        "claims": claims_out,
        "verified": bool(state.get("verified")),
        "refused": refused,
        "refusal_reason": refusal_reason,
        "refusal_code": refusal_code,
        "refusal_message": refusal_message,
        "meta_message": meta_message,
        "ticker": state.get("ticker"),
        "fiscal_year": state.get("fiscal_year"),
        "trace_url": trace_url,
    }


__all__ = ["terminal_payload"]
