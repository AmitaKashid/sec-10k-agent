"""HTTP routes: /ask (SSE or sync), /companies, /health.

The `/ask` endpoint runs the compiled LangGraph singleton. Streaming is
the default; pass `Accept: application/json` to get a single terminal
JSON response (used by evals + curl).
"""

from __future__ import annotations

import json
import logging
import secrets
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.agent.graph import GRAPH
from app.agent.state import AgentState
from app.api.anon_quota import check_and_increment as anon_check_and_increment
from app.api.anon_quota import find_disallowed_ticker
from app.api.auth import Identity, resolve_identity
from app.api.schemas import AskRequest, AskResponse, CompanyOut, CompanyStatsOut, HealthOut
from app.api.sse import terminal_payload
from app.api.token_budget import (
    BUDGET,
    REQUESTS,
    Reservation,
    limit_response,
    tier_limits,
)
from app.api.usage_tracker import collect as collect_usage
from app.api.usage_tracker import total_tokens
from app.data.db import get_session
from app.observability.langfuse import current_trace_url, request_trace, update_trace_output
from app.observability.langfuse import flush as langfuse_flush
from app.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# tiktoken encoder cached at import. cl100k_base is OpenAI's; close enough
# for Anthropic too — we're rate-limiting, not billing. Lazy import keeps
# the rest of the app loadable when tiktoken's BPE files are unavailable.
_ENCODER = None


def _count_tokens(text: str) -> int:
    global _ENCODER
    if _ENCODER is None:
        import tiktoken

        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return len(_ENCODER.encode(text))


# --- Auth dependency -----------------------------------------------------


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    # Constant-time compare to defeat timing oracles (CWE-208).
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid X-API-Key",
        )


def _has_valid_api_key(x_api_key: str | None) -> bool:
    """Admin-tier bypass: a valid X-API-Key grants the same access as a
    Clerk user-tier JWT. Used by /companies (admin tooling, eval
    scripts) so they don't need to mint Clerk sessions."""
    if not x_api_key:
        return False
    return secrets.compare_digest(x_api_key, settings.API_KEY)


# --- /ask ----------------------------------------------------------------


@router.post("/ask")
async def ask(
    body: AskRequest,
    request: Request,
    identity: Annotated[Identity, Depends(resolve_identity)],
    session: Annotated[AsyncSession, Depends(get_session)],
    accept: str | None = Header(default=None),
):
    request_id = str(uuid.uuid4())
    logger.info(
        "ask.identity",
        extra={"tier": identity.tier, "request_id": request_id},
    )

    # --- Pre-flight rate limits --------------------------------------
    # See docs/auth_and_rate_limit_plan.md §`/ask` route changes
    # and docs/anon_hardening_plan.md §5 for the lifetime cap.

    # 0. Lifetime per-IP cap for anon. Cheapest check to evaluate and
    #    the most important to never bypass: pushes anon users into
    #    Clerk sign-in after ANON_LIFETIME_LIMIT total requests, ever.
    #    No-op for user tier or when the feature is disabled.
    anon_allowed, anon_used, anon_limit = await anon_check_and_increment(identity, session)
    if not anon_allowed:
        logger.info(
            "ask.limit.lifetime",
            extra={
                "tier": identity.tier,
                "used": anon_used,
                "budget": anon_limit,
                "request_id": request_id,
            },
        )
        return limit_response(
            status_code=429,
            detail=(
                f"Free preview limit reached ({anon_used}/{anon_limit} questions). "
                "Sign in to keep asking."
            ),
            reason="anonymous_lifetime_limit",
            identity=identity,
            used=anon_used,
            budget=anon_limit,
        )

    # 0.5 Anon ticker allowlist prefilter. Best-effort regex against the
    #     known ticker universe — matches explicit uppercase mentions like
    #     "TSLA" so the frontend LimitModal fires with a sign-in CTA.
    #     Lowercase / company-name forms fall through to the in-graph
    #     anon_gate node, which produces an SSE refusal.
    if identity.tier == "anon":
        blocked_ticker = find_disallowed_ticker(body.question, settings.anon_ticker_allowlist_upper)
        if blocked_ticker:
            logger.info(
                "ask.limit.ticker_not_allowed",
                extra={
                    "tier": identity.tier,
                    "requested_ticker": blocked_ticker,
                    "request_id": request_id,
                },
            )
            return limit_response(
                status_code=403,
                detail=(
                    f"Sign in to ask about {blocked_ticker}. The free preview "
                    "covers AAPL, MSFT, GOOGL, AMZN, and NVDA."
                ),
                reason="anon_ticker_not_allowed",
                identity=identity,
                used=anon_used,
                budget=anon_limit,
            )

    limits = tier_limits(identity.tier)

    # 1. Hard cap on question size. Returns 413 so the frontend can show
    #    "your question is too long" — distinct from 429 ("slow down" or
    #    "sign in"). Tier-aware: authed users get a higher cap.
    input_tokens = _count_tokens(body.question)
    if input_tokens > limits.max_input_tokens:
        logger.info(
            "ask.limit.input_too_large",
            extra={
                "tier": identity.tier,
                "input_tokens": input_tokens,
                "max": limits.max_input_tokens,
                "request_id": request_id,
            },
        )
        return limit_response(
            status_code=413,
            detail=f"question too long ({input_tokens} tokens, max {limits.max_input_tokens})",
            reason="input_too_large",
            identity=identity,
            used=input_tokens,
            budget=limits.max_input_tokens,
        )

    # 2. Request-count cap per tier.
    allowed, req_used, req_budget = await REQUESTS.check_and_consume(identity)
    if not allowed:
        reason = "anonymous_req_limit" if identity.tier == "anon" else "authed_req_limit"
        logger.info(
            "ask.limit.requests",
            extra={
                "tier": identity.tier,
                "used": req_used,
                "budget": req_budget,
                "request_id": request_id,
            },
        )
        return limit_response(
            status_code=429,
            detail=f"request rate limit exceeded ({req_used}/{req_budget} in last 60s)",
            reason=reason,
            identity=identity,
            used=req_used,
            budget=req_budget,
        )

    # 3. Token-budget reservation. The actual usage is reconciled at the
    #    end of the request. Estimate = input tokens + retrieved-chunk
    #    context fed to validate + output. Reconciled in `_finalize_budget`.
    est_total = input_tokens + settings.ASK_EST_RETRIEVAL_TOKENS + settings.ASK_EST_OUTPUT_TOKENS
    reservation, tok_used, tok_budget = await BUDGET.check_and_reserve(identity, est_total)
    if reservation is None:
        # Refund the request-count slot consumed in step 2 — a request
        # rejected on token budget must not also burn its request slot,
        # or the two caps stop reconciling independently (plan §Token
        # budget).
        await REQUESTS.release(identity)
        reason = "anonymous_limit" if identity.tier == "anon" else "authed_limit"
        logger.info(
            "ask.limit.tokens",
            extra={
                "tier": identity.tier,
                "used": tok_used,
                "budget": tok_budget,
                "est": est_total,
                "request_id": request_id,
            },
        )
        return limit_response(
            status_code=429,
            detail=f"token budget exceeded ({tok_used}/{tok_budget} in last 60s)",
            reason=reason,
            identity=identity,
            used=tok_used,
            budget=tok_budget,
        )

    # Pull the (possibly checkpointed) graph from app state. The lifespan
    # in app.main installs the no-checkpointer GRAPH by default and only
    # swaps in a checkpointed build when the AsyncPostgresSaver actually
    # connected (it falls back silently if Postgres is unreachable even
    # with the flag on). Gate memory on the *real* checkpointer presence,
    # NOT settings.LANGGRAPH_CHECKPOINTER_ENABLED — otherwise the client
    # gets a thread_id and believes memory works when it doesn't.
    graph = getattr(request.app.state, "graph", GRAPH)
    checkpointer_active = getattr(request.app.state, "memory_active", False)

    # Only mint/echo a thread_id when memory is genuinely active. When it
    # isn't, thread_id stays None so the response signals "stateless" and
    # we never attach a configurable thread to a non-checkpointed graph.
    thread_id: str | None = None
    if checkpointer_active:
        thread_id = body.thread_id or str(uuid.uuid4())

    initial = AgentState(
        question=body.question,
        request_id=request_id,
        tier=identity.tier,
        plan_model_override=body.plan_model_override,
        validate_model_override=body.validate_model_override,
    )

    config = {"configurable": {"thread_id": thread_id}} if thread_id else None

    wants_json = bool(accept) and "application/json" in accept and "text/event-stream" not in accept
    if wants_json:
        return await _ask_sync(initial, request_id, thread_id, graph, config, reservation)
    return EventSourceResponse(
        _ask_stream(initial, request_id, thread_id, request, graph, config, reservation)
    )


async def _ask_sync(
    initial: AgentState,
    request_id: str,
    thread_id: str | None,
    graph,
    config: dict | None,
    reservation: Reservation,
) -> JSONResponse:
    try:
        with request_trace(
            request_id,
            input=initial.question,
            # session_id groups all turns of one conversation under the
            # Langfuse Sessions view. Only present when memory is on.
            session_id=thread_id,
            metadata={"thread_id": thread_id, "mode": "sync"},
        ):
            final = (
                await graph.ainvoke(initial, config=config)
                if config
                else await graph.ainvoke(initial)
            )
            # Capture once — also reused for terminal_payload below.
            final_dict = _as_dict(final)
            # Attach the answer (or refusal/meta message — same fallback
            # order as terminal_payload) so the trace's Output panel
            # renders it. Must run inside the trace context.
            update_trace_output(
                final_dict.get("final_answer")
                or final_dict.get("refusal_message")
                or final_dict.get("meta_message")
            )
            # Read inside the trace context (current_trace_url reads the
            # active-trace contextvar). EXPOSE_TRACE_URLS gates whether
            # it reaches the client (HLD 4: dev/demo on, prod off).
            trace_url = current_trace_url() if settings.EXPOSE_TRACE_URLS else None
    finally:
        langfuse_flush()
        await _finalize_budget(request_id, reservation, initial.question)
    payload = terminal_payload(final_dict, request_id, thread_id, trace_url)
    return JSONResponse(AskResponse(**payload).model_dump(mode="json"))


async def _ask_stream(
    initial: AgentState,
    request_id: str,
    thread_id: str | None,
    request: Request,
    graph,
    config: dict | None,
    reservation: Reservation,
) -> AsyncIterator[dict]:
    """Per-node `status` events for the loading UI, real per-token `token`
    events while the refusal_composer or concierge LLM streams, then a
    terminal `done` event with the verified / refused / meta payload.

    Two stream modes are interleaved:
      * `updates` → `event: status` (one per node entered)
      * `messages` → `event: token` (one per LLM chunk inside the
        refusal_composer or concierge nodes only — we don't surface
        validate's chunks because cite_check hasn't verified them yet)

    Cite_check has not run when validate streams, so any partial answer
    would be unverified. Frontend simulates typing on the final verified
    text and consumes `event: token` for the refusal and meta paths.
    """
    last_state: dict = {}
    trace_url: str | None = None
    try:
        with request_trace(
            request_id,
            input=initial.question,
            session_id=thread_id,
            metadata={"thread_id": thread_id, "mode": "stream"},
        ):
            try:
                stream = (
                    graph.astream(initial, config=config, stream_mode=["updates", "messages"])
                    if config
                    else graph.astream(initial, stream_mode=["updates", "messages"])
                )
                async for mode, payload in stream:
                    if await request.is_disconnected():
                        logger.info("[%s] client disconnected mid-stream", request_id)
                        return

                    if mode == "updates":
                        # LangGraph yields {node_name: {state_delta_dict}} per step.
                        for node, delta in payload.items():
                            last_state = {**last_state, **(delta or {})}
                            yield {
                                "event": "status",
                                "data": json.dumps({"step": node, "request_id": request_id}),
                            }
                    elif mode == "messages":
                        # `payload` is (AIMessageChunk, metadata).
                        chunk, meta = payload
                        if meta.get("langgraph_node") not in ("refusal_composer", "concierge"):
                            continue
                        # `chunk.content` is usually a string; some providers
                        # return a list of content blocks — coerce to text.
                        text_piece = _chunk_text(chunk)
                        if not text_piece:
                            continue
                        yield {
                            "event": "token",
                            "data": json.dumps({"request_id": request_id, "text": text_piece}),
                        }
            except Exception as e:  # noqa: BLE001 — convert to a clean SSE error event
                logger.exception("[%s] graph stream failed", request_id)
                yield {
                    "event": "error",
                    "data": json.dumps(
                        {
                            "request_id": request_id,
                            "error": type(e).__name__,
                            "detail": str(e)[:200],
                        }
                    ),
                }
                return
            # Attach the answer (or refusal/meta) to the trace's Output
            # panel — same fallback order as terminal_payload. Must run
            # inside the trace context, before it closes.
            update_trace_output(
                last_state.get("final_answer")
                or last_state.get("refusal_message")
                or last_state.get("meta_message")
            )
            # Read while the trace context is still active (it is reset
            # when the `with` exits). Gated by EXPOSE_TRACE_URLS.
            trace_url = current_trace_url() if settings.EXPOSE_TRACE_URLS else None
    finally:
        langfuse_flush()
        # Runs whether the stream finished, errored, or the client
        # disconnected mid-flight — reservation is always reconciled so
        # the budget doesn't leak the over-estimate.
        await _finalize_budget(request_id, reservation, initial.question)

    yield {
        "event": "done",
        "data": json.dumps(terminal_payload(last_state, request_id, thread_id, trace_url)),
    }


async def _finalize_budget(
    request_id: str,
    reservation: Reservation,
    question: str,
) -> None:
    """Reconcile the up-front token-budget reservation with actual LLM
    cost. Logs a structured `ask.tokens` event for ops visibility.

    Actual = input_tokens (from the original question) + sum of every
    node's recorded LLM input+output. If a provider didn't surface usage
    (e.g. structured-output path on an older LC version), missing
    contributions degrade gracefully — we just under-charge."""
    usage = collect_usage(request_id)
    # Re-count the question rather than thread it through — cheap, and
    # keeps this helper independent of the route's local var lifetime.
    input_tokens = _count_tokens(question)
    actual = input_tokens + total_tokens(usage)
    await BUDGET.finalize(reservation, actual)
    logger.info(
        "ask.tokens",
        extra={
            "request_id": request_id,
            "input_tokens": input_tokens,
            "node_tokens": total_tokens(usage),
            "nodes": [u.node for u in usage],
            "total": actual,
        },
    )


def _chunk_text(chunk) -> str:
    """Best-effort extraction of plain text from an AIMessageChunk.

    Anthropic returns content as a list of blocks; OpenAI returns a string.
    We only care about text content for streaming display.
    """
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content) if content else ""


def _as_dict(state) -> dict:
    """LangGraph's ainvoke may return either an AgentState or a dict
    depending on version — normalize to dict for projection."""
    if hasattr(state, "model_dump"):
        return state.model_dump()
    return dict(state)


# --- /companies ----------------------------------------------------------


_COMPANIES_LIST_SQL = text(
    """
    WITH ten_k AS (
        SELECT cik, accession_no, fiscal_year, filing_date
        FROM filings
        WHERE form_type IN ('10-K', '10-K/A')
    ),
    latest AS (
        SELECT DISTINCT ON (cik) cik, fiscal_year, filing_date
        FROM ten_k
        ORDER BY cik, fiscal_year DESC, filing_date DESC
    ),
    counts AS (
        SELECT cik, COUNT(*) AS filings_count
        FROM ten_k
        GROUP BY cik
    )
    SELECT
        c.ticker,
        c.name,
        latest.fiscal_year         AS latest_10k_period,
        latest.filing_date         AS latest_10k_filed_at,
        COALESCE(counts.filings_count, 0) AS filings_count,
        EXISTS (SELECT 1 FROM chunks ch WHERE ch.cik = c.cik)      AS has_chunks,
        EXISTS (SELECT 1 FROM xbrl_facts xf WHERE xf.cik = c.cik)  AS has_xbrl
    FROM companies c
    LEFT JOIN latest ON latest.cik = c.cik
    LEFT JOIN counts ON counts.cik = c.cik
    ORDER BY c.ticker
    """
)


def _coverage_status(has_chunks: bool, has_xbrl: bool) -> str:
    if has_chunks and has_xbrl:
        return "complete"
    if has_chunks or has_xbrl:
        return "pending"
    return "missing"


@router.get(
    "/companies",
    response_model=list[CompanyOut],
)
async def list_companies(
    session: Annotated[AsyncSession, Depends(get_session)],
    identity: Annotated[Identity, Depends(resolve_identity)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> list[CompanyOut]:
    # Anon users can chat about a 5-ticker preview but can't browse the
    # full universe. Admin tooling can bypass via X-API-Key.
    # See docs/anon_hardening_plan.md §6.
    if identity.tier == "anon" and not _has_valid_api_key(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="auth_required")
    res = await session.execute(_COMPANIES_LIST_SQL)
    rows = res.mappings().all()
    return [
        CompanyOut(
            ticker=r["ticker"],
            name=r["name"],
            sector=None,
            latest_10k_period=r["latest_10k_period"],
            latest_10k_filed_at=(
                r["latest_10k_filed_at"].isoformat() if r["latest_10k_filed_at"] else None
            ),
            filings_count=int(r["filings_count"]),
            coverage_status=_coverage_status(bool(r["has_chunks"]), bool(r["has_xbrl"])),
        )
        for r in rows
    ]


_COMPANIES_STATS_SQL = text(
    """
    SELECT
        (SELECT COUNT(*) FROM companies)                                       AS issuers,
        (SELECT COUNT(*) FROM filings WHERE form_type IN ('10-K','10-K/A'))    AS filings_indexed,
        (SELECT COUNT(*) FROM xbrl_facts)                                      AS xbrl_facts,
        (
            SELECT COUNT(*) FROM companies c
            WHERE EXISTS (SELECT 1 FROM chunks ch WHERE ch.cik = c.cik)
              AND EXISTS (SELECT 1 FROM xbrl_facts xf WHERE xf.cik = c.cik)
        )                                                                      AS coverage_complete,
        (SELECT MAX(ingested_at) FROM filings)                                 AS last_refresh
    """
)


@router.get(
    "/companies/stats",
    response_model=CompanyStatsOut,
)
async def companies_stats(
    session: Annotated[AsyncSession, Depends(get_session)],
    identity: Annotated[Identity, Depends(resolve_identity)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> CompanyStatsOut:
    if identity.tier == "anon" and not _has_valid_api_key(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="auth_required")
    res = await session.execute(_COMPANIES_STATS_SQL)
    row = res.mappings().one()
    return CompanyStatsOut(
        issuers=int(row["issuers"]),
        filings_indexed=int(row["filings_indexed"]),
        xbrl_facts=int(row["xbrl_facts"]),
        coverage_complete=int(row["coverage_complete"]),
        coverage_total=int(row["issuers"]),
        last_refresh=row["last_refresh"].isoformat() if row["last_refresh"] else None,
    )


# --- /health -------------------------------------------------------------


@router.get("/health", response_model=HealthOut)
async def health(session: Annotated[AsyncSession, Depends(get_session)]) -> HealthOut:
    db_ok = "ok"
    try:
        await session.execute(text("SELECT 1"))
    except Exception as e:  # noqa: BLE001
        db_ok = f"error: {type(e).__name__}"
    graph_ok = "ok" if GRAPH is not None else "error: graph not compiled"
    overall = "ok" if db_ok == "ok" and graph_ok == "ok" else "degraded"
    return HealthOut(status=overall, db=db_ok, graph=graph_ok)


__all__ = ["router"]
