"""Langfuse v2 observability — client factory + safe no-op guard.

Stage A: primitives only. Graph/route wiring is Stage B.

Why no LangChain CallbackHandler: langfuse v2's
`langfuse.callback.CallbackHandler` hard-imports `langchain.schema.*`
paths removed in langchain 1.x (this repo runs langchain 1.2.x). So we
use the v2 client + `@observe()`/`langfuse_context` only and capture LLM
calls as manual generation spans in Stage B. See
`docs/langfuse_observability_plan.md` §3a.

Hard contract: when `LANGFUSE_PUBLIC_KEY`/`SECRET_KEY` are unset (dev,
CI, tests — the default), every function here is a cheap no-op with
zero network calls and no client construction. Importing this module
has no side effects (no client built at import time).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import TYPE_CHECKING, Any

# Safe without langchain (verified): the decorators module does not pull
# the broken v2 langchain callback. Kept at module top because it has no
# import-time side effects (no client is constructed here).
from langfuse.decorators import langfuse_context, observe

from app.settings import settings

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse.client import StatefulSpanClient, StatefulTraceClient

logger = logging.getLogger(__name__)

# Active per-request trace handle, mirroring the request_id/node
# contextvars in observability/logging.py. Set by `request_trace()` in
# the route; read by `node_span()` (graph._instrument) and
# `current_trace_url()`. ContextVars propagate across the asyncio tasks
# LangGraph spawns within one request, so node spans nest under the
# request's trace without threading a handle through AgentState (which
# would also collide with the checkpointer serde).
_trace_var: ContextVar[StatefulTraceClient | None] = ContextVar("langfuse_trace", default=None)

__all__ = [
    "langfuse_enabled",
    "get_client",
    "request_trace",
    "node_span",
    "current_trace_url",
    "update_trace_output",
    "flush",
    "observe",
    "langfuse_context",
]


def langfuse_enabled() -> bool:
    """True only when both keys are configured. Single source of truth
    for every no-op branch in this module and (Stage B) the graph/route
    instrumentation."""
    return bool(settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY)


@lru_cache(maxsize=1)
def get_client() -> Langfuse | None:
    """Lazily build the v2 client, or return None when disabled.

    Returning None (rather than a `Langfuse(enabled=False)`) avoids the
    SDK's "client is disabled" stderr notice and makes the no-op branch
    explicit at every call site. Cached so we build at most one client
    per process; tests clear the cache via `get_client.cache_clear()`.
    """
    if not langfuse_enabled():
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
    except Exception as exc:  # noqa: BLE001 — observability must never break /ask
        logger.warning(
            "langfuse.client.init_failed",
            extra={"error_type": type(exc).__name__},
        )
        return None


@contextmanager
def request_trace(
    request_id: str,
    *,
    input: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[StatefulTraceClient | None]:
    """Open one Langfuse trace for a `/ask` request, keyed by
    `request_id` (so the Langfuse trace id == our request id, which also
    satisfies the deferred query_log correlation). Sets `_trace_var` for
    the duration so `node_span()` nests under it. No-op + yields None
    when disabled or on any SDK error — the request must never break.

    `input` populates the trace-level Input panel in the dashboard
    (the user's question — single biggest UX win for a Langfuse user).
    `session_id`, when set, groups this trace under a Langfuse session
    so all turns of one conversation cluster under the Sessions view.
    We use `thread_id` as the session id when the multi-turn
    checkpointer is active; otherwise None ⇒ no session grouping.
    """
    client = get_client()
    if client is None:
        yield None
        return
    try:
        kwargs: dict[str, Any] = {
            "id": request_id,
            "name": "ask",
            "metadata": metadata or {},
        }
        # User content (the question) only leaves the process when the
        # operator opts in. Trace + spans + session_id are metadata and
        # always flow when Langfuse is enabled — only the user-content
        # fields are gated. See settings.LANGFUSE_CAPTURE_IO.
        if input is not None and settings.LANGFUSE_CAPTURE_IO:
            kwargs["input"] = input
        if session_id:
            # Pass session_id only when truthy — Langfuse treats an
            # explicit None as a valid value and would create
            # "sessionless" groupings cluttering the dashboard.
            kwargs["session_id"] = session_id
        trace = client.trace(**kwargs)
    except Exception as exc:  # noqa: BLE001 — observability must not break /ask
        logger.warning(
            "langfuse.trace_start_failed",
            extra={"error_type": type(exc).__name__},
        )
        yield None
        return
    token = _trace_var.set(trace)
    try:
        yield trace
    finally:
        _trace_var.reset(token)


def update_trace_output(output: str | None) -> None:
    """Attach the final answer text as the active trace's `output` so
    the Langfuse dashboard's Output panel renders it. Call once after
    the graph run, inside the `request_trace` context. No-op when
    disabled / no active trace / empty output / `LANGFUSE_CAPTURE_IO`
    is False (the operator kill-switch for user-content egress). All
    SDK errors swallowed."""
    if not output:
        return
    if not settings.LANGFUSE_CAPTURE_IO:
        return
    trace = _trace_var.get()
    if trace is None:
        return
    try:
        trace.update(output=output)
    except Exception as exc:  # noqa: BLE001 — never break /ask on an update
        logger.warning(
            "langfuse.trace_update_failed",
            extra={"error_type": type(exc).__name__},
        )


@contextmanager
def node_span(name: str, **attrs: Any) -> Iterator[StatefulSpanClient | None]:
    """Child span on the active request trace. No-op when there is no
    active trace (disabled, or outside `request_trace`). All SDK errors
    are swallowed so instrumentation can never fail a graph node."""
    trace = _trace_var.get()
    if trace is None:
        yield None
        return
    try:
        span = trace.span(name=name, metadata=attrs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("langfuse.span_start_failed", extra={"error_type": type(exc).__name__})
        yield None
        return
    try:
        yield span
    except Exception as exc:
        # Mark the span failed, then re-raise — the node's own error
        # handling/logging is unchanged; we only annotate the trace.
        try:
            span.update(level="ERROR", status_message=type(exc).__name__)
        except Exception:  # noqa: BLE001
            pass
        raise
    finally:
        try:
            span.end()
        except Exception:  # noqa: BLE001
            pass


def current_trace_url() -> str | None:
    """URL of the active request trace, or None when disabled / outside
    a trace / on any SDK error. Read after a run and surfaced in the
    `done` event only when `EXPOSE_TRACE_URLS`."""
    trace = _trace_var.get()
    if trace is None:
        return None
    try:
        return trace.get_trace_url()
    except Exception as exc:  # noqa: BLE001 — never propagate into the request
        logger.warning(
            "langfuse.trace_url_failed",
            extra={"error_type": type(exc).__name__},
        )
        return None


def flush() -> None:
    """Best-effort flush of batched events. Call per-request and on app
    shutdown (Stage B) — Langfuse batches in a background thread, so an
    idle worker can otherwise drop the last trace. No-op when disabled."""
    client = get_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:  # noqa: BLE001 — flush failure must not 500 a request
        logger.warning(
            "langfuse.flush_failed",
            extra={"error_type": type(exc).__name__},
        )
