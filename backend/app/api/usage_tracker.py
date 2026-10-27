"""Per-request LLM usage tracker.

Keyed by `request_id` (the per-/ask UUID minted in the route). Nodes call
`record(request_id, ...)` after each LLM call; the route calls
`collect(request_id)` at the end to reconcile the token-budget
reservation with actual cost.

We deliberately do NOT put this on AgentState. The conversation
checkpointer makes any list-with-reducer on state accumulate across
turns, which would charge each /ask for the entire thread's prior LLM
cost. Keying by request_id (which is fresh per request) keeps the
accounting per-call regardless of checkpointer mode.

In-process only; cross-replica accounting is out of scope along with the
rest of the limiter (see docs/auth_and_rate_limit_plan.md §Out of
scope).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeUsage:
    node: str
    input_tokens: int
    output_tokens: int

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


# Module-level dict, mutated only from inside the asyncio loop. Dict ops
# in CPython are atomic per call — and our writers always append to a
# bucketed list — so we skip the explicit lock. Tests can clear it.
_USAGE: dict[str, list[NodeUsage]] = {}


def record(request_id: str, node: str, input_tokens: int, output_tokens: int) -> None:
    """Append one usage entry for the request. No-op when `request_id`
    is falsy — e.g. when a node runs outside a request context (tests)."""
    if not request_id:
        return
    _USAGE.setdefault(request_id, []).append(
        NodeUsage(node=node, input_tokens=int(input_tokens), output_tokens=int(output_tokens))
    )


def collect(request_id: str) -> list[NodeUsage]:
    """Pop and return all usage entries for the request. Safe to call
    even if nothing was recorded (returns empty list)."""
    return _USAGE.pop(request_id, [])


def total_tokens(entries: list[NodeUsage]) -> int:
    return sum(e.total for e in entries)


def extract_openai_usage(response) -> tuple[int, int] | None:
    """Best-effort (input, output) extraction from an OpenAI ChatCompletion
    response. Returns None if usage is missing or malformed.

    The OpenAI Python SDK exposes `response.usage.prompt_tokens` /
    `response.usage.completion_tokens`. Langchain's wrappers re-expose
    via `response_metadata.token_usage` or `usage_metadata` depending
    on version — caller may need to dispatch."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    try:
        return int(usage.prompt_tokens), int(usage.completion_tokens)
    except (AttributeError, TypeError, ValueError):
        return None


def extract_langchain_usage(message) -> tuple[int, int] | None:
    """Best-effort (input, output) extraction from a LangChain AIMessage.

    LangChain normalizes provider usage into either:
      * `message.usage_metadata = {"input_tokens": N, "output_tokens": M, ...}` (newer)
      * `message.response_metadata["token_usage"] = {"prompt_tokens": N, ...}` (older)

    Returns None when neither is populated.
    """
    meta = getattr(message, "usage_metadata", None)
    if isinstance(meta, dict) and ("input_tokens" in meta or "output_tokens" in meta):
        try:
            return int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))
        except (TypeError, ValueError):
            pass
    rm = getattr(message, "response_metadata", None) or {}
    tu = rm.get("token_usage") if isinstance(rm, dict) else None
    if isinstance(tu, dict):
        try:
            return int(tu.get("prompt_tokens", 0)), int(tu.get("completion_tokens", 0))
        except (TypeError, ValueError):
            pass
    return None


__all__ = [
    "NodeUsage",
    "collect",
    "extract_langchain_usage",
    "extract_openai_usage",
    "record",
    "total_tokens",
]
