"""Langfuse v4 request traces and node spans; tracing errors never fail requests."""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class _Trace:
    client: Any
    root: Any
    trace_id: str


_trace_var: ContextVar[_Trace | None] = ContextVar("langfuse_trace", default=None)


def _warn(event: str, exc: Exception) -> None:
    logger.warning(event, extra={"error_type": type(exc).__name__})


def langfuse_enabled() -> bool:
    return bool(settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY)


@lru_cache(maxsize=1)
def get_client():
    if not langfuse_enabled():
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            base_url=settings.LANGFUSE_HOST,
        )
    except Exception as exc:
        _warn("langfuse.client.init_failed", exc)
        return None


def _update(span, **kwargs):
    try:
        span.update(**kwargs)
    except Exception as exc:
        _warn("langfuse.span_update_failed", exc)


def _end(span):
    try:
        span.end()
    except Exception as exc:
        _warn("langfuse.span_end_failed", exc)


@contextmanager
def request_trace(request_id: str, *, input: str | None = None,
                  session_id: str | None = None,
                  metadata: dict[str, Any] | None = None) -> Iterator[Any]:
    client = get_client()
    if client is None:
        # Also isolate a disabled nested request from any surrounding trace.
        token = _trace_var.set(None)
        try:
            yield None
        finally:
            _trace_var.reset(token)
        return
    root = None
    attributes = None
    entered = False
    try:
        from langfuse import propagate_attributes

        attributes = propagate_attributes(**({"session_id": session_id} if session_id else {}))
        attributes.__enter__()
        entered = True
        trace_id = client.create_trace_id(seed=request_id)
        root = client.start_observation(
            name="ask", as_type="span", trace_context={"trace_id": trace_id},
            metadata={**(metadata or {}), "request_id": request_id},
            **({"input": input} if input is not None and settings.LANGFUSE_CAPTURE_IO else {}),
        )
        active = _Trace(client, root, trace_id)
    except Exception as exc:
        _warn("langfuse.trace_start_failed", exc)
        active = None
    token = _trace_var.set(active)
    try:
        yield root if active is not None else None
    except BaseException as exc:
        if root is not None:
            _update(root, level="ERROR", status_message=type(exc).__name__)
        raise
    finally:
        _trace_var.reset(token)
        if root is not None:
            _end(root)
        if entered:
            try:
                attributes.__exit__(None, None, None)
            except Exception as exc:
                _warn("langfuse.context_end_failed", exc)


def update_trace_output(output: str | None) -> None:
    active = _trace_var.get()
    if active is not None and output and settings.LANGFUSE_CAPTURE_IO:
        _update(active.root, output=output)


@contextmanager
def node_span(name: str, **attrs: Any) -> Iterator[Any]:
    active = _trace_var.get()
    span = None
    if active is not None:
        try:
            # Explicit parenting remains correct across LangGraph asyncio tasks.
            span = active.root.start_observation(name=name, as_type="span", metadata=attrs)
        except Exception as exc:
            _warn("langfuse.span_start_failed", exc)
    try:
        yield span
    except BaseException as exc:
        if span is not None:
            _update(span, level="ERROR", status_message=type(exc).__name__)
        raise
    finally:
        if span is not None:
            _end(span)


def current_trace_url() -> str | None:
    active = _trace_var.get()
    if active is None:
        return None
    try:
        return active.client.get_trace_url(trace_id=active.trace_id)
    except Exception as exc:
        _warn("langfuse.trace_url_failed", exc)
        return None


def flush() -> None:
    client = get_client()
    if client is not None:
        try:
            client.flush()
        except Exception as exc:
            _warn("langfuse.flush_failed", exc)

