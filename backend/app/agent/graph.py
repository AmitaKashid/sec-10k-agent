"""LangGraph wiring. See docs/lld.md §LLD 7 — "graph.py" section.

Single conditional edge after each gating node. Compiled once at import
time as a module-level singleton — FastAPI startup imports this and
shares the compiled graph across requests.
"""

from __future__ import annotations

import logging
import time
from functools import wraps

from langgraph.graph import END, StateGraph

from app.agent.groq_runtime import SERVICE_ERRORS, failure
from app.agent.nodes.anon_gate import anon_gate
from app.agent.nodes.cite_check import cite_check
from app.agent.nodes.concierge import concierge
from app.agent.nodes.fetch import fetch
from app.agent.nodes.locate import locate
from app.agent.nodes.plan import plan
from app.agent.nodes.refusal_composer import refusal_composer
from app.agent.nodes.retrieve import retrieve
from app.agent.nodes.validate import validate
from app.agent.state import AgentState
from app.observability.langfuse import node_span
from app.observability.logging import node_var, request_id_var

_logger = logging.getLogger("app.agent.graph")


def _instrument(node_name: str, fn):
    """Wrap a LangGraph node so it logs entry/exit/duration with the
    `node` contextvar set, and seeds `request_id` from state on the first
    node so background graph runs (no HTTP middleware) still tag logs."""

    @wraps(fn)
    async def wrapper(state: AgentState):
        node_token = node_var.set(node_name)
        rid_token = None
        if request_id_var.get() is None and getattr(state, "request_id", None):
            rid_token = request_id_var.set(state.request_id)

        _logger.info(
            "node.start",
            extra={
                "question_type": state.question_type,
                "ticker": state.ticker,
                "fiscal_year": state.fiscal_year,
                "question": state.question if node_name == "plan" else None,
            },
        )
        start = time.perf_counter()
        try:
            # node_span is a no-op (nullcontext-like) when Langfuse is
            # disabled or there is no active request trace — zero
            # overhead on the default path.
            with node_span(
                f"node:{node_name}",
                request_id=getattr(state, "request_id", None),
                question_type=state.question_type,
                ticker=state.ticker,
                fiscal_year=state.fiscal_year,
            ):
                result = await fn(state)
        except SERVICE_ERRORS as exc:
            result = failure(state, node_name, exc)
        except Exception:
            _logger.exception(
                "node.error",
                extra={"duration_ms": round((time.perf_counter() - start) * 1000, 1)},
            )
            raise
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)

        # Result is a state delta dict; surface a few interpretable fields.
        delta = result or {}
        refused = delta.get("question_type") == "REFUSE" or state.question_type == "REFUSE"
        _logger.info(
            "node.end",
            extra={
                "duration_ms": duration_ms,
                "refused": refused,
                "refusal_reason": delta.get("refusal_reason"),
                "refusal_code": delta.get("refusal_code"),
                "delta_keys": sorted(delta.keys()),
            },
        )
        node_var.reset(node_token)
        if rid_token is not None:
            request_id_var.reset(rid_token)
        return result

    return wrapper


def _route_if_refused(target: str):
    """Generic gate: if any prior node set REFUSE, jump to END; else continue."""

    def _route(state: AgentState) -> str:
        if state.question_type == "REFUSE":
            _logger.info(
                "route.refuse",
                extra={
                    "next": "END",
                    "skipped": target,
                    "refusal_reason": state.refusal_reason,
                    "refusal_code": state.refusal_code,
                },
            )
            return "refuse"
        _logger.info("route.continue", extra={"next": target})
        return "continue"

    _route.__name__ = f"route_to_{target}"
    return _route


def _route_after_plan(state: AgentState) -> str:
    """Plan has three exits:
    * META — greetings / capability / identity → concierge
    * REFUSE — anything plan classified as out-of-scope → refusal_composer
    * else (A_numerical, B_qualitative) → fetch
    """
    if state.question_type == "META":
        _logger.info("route.meta", extra={"next": "concierge"})
        return "meta"
    if state.question_type == "REFUSE":
        _logger.info(
            "route.refuse",
            extra={
                "next": "refusal_composer",
                "skipped": "fetch",
                "refusal_reason": state.refusal_reason,
                "refusal_code": state.refusal_code,
            },
        )
        return "refuse"
    _logger.info("route.continue", extra={"next": "fetch"})
    return "continue"


def _route_after_validate(state: AgentState) -> str:
    if state.question_type == "REFUSE":
        _logger.info(
            "route.refuse",
            extra={
                "next": "refusal_composer",
                "skipped": "cite_check",
                "refusal_reason": state.refusal_reason,
                "refusal_code": state.refusal_code,
            },
        )
        return "refuse"
    if state.cache_hit:
        # Validate already populated final_answer from cache; skip cite_check
        # so we don't re-verify already-trusted output.
        _logger.info("route.cached", extra={"next": "END", "skipped": "cite_check"})
        return "cached"
    _logger.info("route.continue", extra={"next": "cite_check"})
    return "continue"


def _route_after_cite_check(state: AgentState) -> str:
    """cite_check sets `refusal_reason_final` on failure. Route to the
    composer for prose shaping; otherwise straight to END with the
    verified answer."""
    if state.refusal_reason_final or state.question_type == "REFUSE":
        _logger.info(
            "route.refuse",
            extra={
                "next": "refusal_composer",
                "skipped": "END",
                "refusal_reason": state.refusal_reason_final,
                "refusal_code": state.refusal_code,
            },
        )
        return "refuse"
    return "pass"


def _build() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("plan", _instrument("plan", plan))
    g.add_node("anon_gate", _instrument("anon_gate", anon_gate))
    g.add_node("fetch", _instrument("fetch", fetch))
    g.add_node("locate", _instrument("locate", locate))
    g.add_node("retrieve", _instrument("retrieve", retrieve))
    g.add_node("validate", _instrument("validate", validate))
    g.add_node("cite_check", _instrument("cite_check", cite_check))
    g.add_node("refusal_composer", _instrument("refusal_composer", refusal_composer))
    g.add_node("concierge", _instrument("concierge", concierge))

    g.set_entry_point("plan")

    # Plan has three exits: META → concierge (greetings, capability),
    # REFUSE → refusal_composer (out-of-scope refusals), else anon_gate
    # (which may itself REFUSE for anon tier + non-allowlisted ticker).
    g.add_conditional_edges(
        "plan",
        _route_after_plan,
        {"continue": "anon_gate", "refuse": "refusal_composer", "meta": "concierge"},
    )
    g.add_conditional_edges(
        "anon_gate",
        _route_if_refused("fetch"),
        {"continue": "fetch", "refuse": "refusal_composer"},
    )
    g.add_conditional_edges(
        "fetch",
        _route_if_refused("locate"),
        {"continue": "locate", "refuse": "refusal_composer"},
    )
    g.add_conditional_edges(
        "locate",
        _route_if_refused("retrieve"),
        {"continue": "retrieve", "refuse": "refusal_composer"},
    )
    g.add_conditional_edges(
        "retrieve",
        _route_if_refused("validate"),
        {"continue": "validate", "refuse": "refusal_composer"},
    )
    g.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"continue": "cite_check", "cached": END, "refuse": "refusal_composer"},
    )
    g.add_conditional_edges(
        "cite_check",
        _route_after_cite_check,
        {"pass": END, "refuse": "refusal_composer"},
    )
    g.add_edge("refusal_composer", END)
    g.add_edge("concierge", END)

    return g


def build_graph(checkpointer=None):
    """Compile a fresh graph, optionally with a checkpointer for multi-turn
    memory. When `checkpointer` is None, behaves identically to the
    historical module-level GRAPH (no persistence across requests). The
    FastAPI lifespan in `app.main` calls this with an `AsyncPostgresSaver`
    when `settings.LANGGRAPH_CHECKPOINTER_ENABLED` is true; tests can pass
    an `InMemorySaver`."""
    g = _build()
    return g.compile(checkpointer=checkpointer) if checkpointer is not None else g.compile()


# Compiled-once singleton. Importing this module is enough — no init step.
# This is the no-checkpointer build; the lifespan in `app.main` may
# replace `app.state.graph` with a checkpointed build when the flag is
# on. Routes should prefer `request.app.state.graph` over importing
# `GRAPH` directly.
GRAPH = build_graph()


__all__ = ["GRAPH", "build_graph"]
