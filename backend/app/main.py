"""FastAPI entrypoint. See docs/lld.md §LLD 8.

App-level wiring only: middleware, lifespan, route inclusion. Endpoint
logic lives in `app.api.routes`. The compiled LangGraph singleton is
imported at startup so the first `/ask` doesn't pay compile latency.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.routes import router
from app.observability.logging import configure_logging, redact_secrets, request_id_var
from app.settings import settings

configure_logging()

logger = logging.getLogger(__name__)


def _rate_limit_key(request: Request) -> str:
    """Per-API-key when present, else per-IP. Per HLD 4: this layer is the
    per-key cap; the per-IP cap lives at the Vercel edge. Uses the same
    proxy-aware client IP resolution as anon identity keying — see
    `app.api.auth._get_client_ip`."""
    from app.api.auth import _get_client_ip

    return request.headers.get("x-api-key") or _get_client_ip(request)


limiter = Limiter(
    key_func=_rate_limit_key, default_limits=[f"{settings.RATE_LIMIT_PER_MINUTE}/minute"]
)


def _checkpointer_conn_string() -> str:
    """Convert the SQLAlchemy+asyncpg DATABASE_URL to a psycopg-compatible
    URL for AsyncPostgresSaver. psycopg does NOT understand the
    `+asyncpg` driver suffix."""
    return settings.DATABASE_URL.replace("+asyncpg", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Force graph compile on startup so the first request is fast.
    from app.agent.graph import GRAPH, build_graph

    # Default: no-checkpointer module-level singleton. Routes pull from
    # `request.app.state.graph` so the lifespan can swap in a
    # checkpointed build below without touching every route.
    app.state.graph = GRAPH
    app.state.checkpointer_cm = None
    # True only when the checkpointed graph was actually installed —
    # distinct from `checkpointer_cm is not None`, which is also set on
    # the setup()-failed path (cm stored only so shutdown can close it).
    # Routes gate memory on this flag, not on the connection object.
    app.state.memory_active = False
    # Holds the fire-and-forget startup prune task. Stashed so Python's
    # GC can't reap the Task mid-execution (the canonical fix for
    # RUF006) — the running uvicorn loop already keeps it alive in
    # practice, but holding an explicit reference makes that
    # guaranteed and lets future shutdown cancel it cleanly.
    app.state.prune_task = None

    logger.info("startup: graph compiled, ready")

    # Defense-in-depth: a deployment with Clerk JWKS configured but no
    # `azp` allowlist accepts any token signed by the same Clerk app —
    # cross-app reuse is undefended. Warn loudly at startup so prod
    # operators notice the missing CLERK_AUTHORIZED_PARTIES.
    if settings.CLERK_JWKS_URL and not settings.CLERK_AUTHORIZED_PARTIES:
        logger.warning(
            "startup.auth.azp_unconfigured",
            extra={"detail": "CLERK_JWKS_URL set but CLERK_AUTHORIZED_PARTIES is empty"},
        )

    # Multi-turn memory: when enabled, enter the AsyncPostgresSaver
    # context manager for the lifetime of the app and replace
    # `app.state.graph` with a checkpointed build. See
    # docs/conversation_memory_plan.md §5.
    if settings.LANGGRAPH_CHECKPOINTER_ENABLED:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

            from app.agent.state import CHECKPOINTED_TYPES

            # Without an explicit allowlist, the msgpack serde refuses to
            # round-trip our Pydantic state types from Postgres (warns
            # now, hard-blocks in a future langgraph). InMemorySaver
            # doesn't serialize, so this only bites the flag-on path.
            serde = JsonPlusSerializer(allowed_msgpack_modules=list(CHECKPOINTED_TYPES))
            cm = AsyncPostgresSaver.from_conn_string(_checkpointer_conn_string(), serde=serde)
            saver = await cm.__aenter__()
            # Store the cm BEFORE setup(): if setup() raises, the shutdown
            # __aexit__ must still run or we leak the opened connection.
            app.state.checkpointer_cm = cm
            await saver.setup()
            app.state.graph = build_graph(checkpointer=saver)
            app.state.memory_active = True
            logger.info("startup.checkpointer.ready")

            # Best-effort TTL sweep of stale checkpoint rows. Runs after
            # setup() so the tables exist. Fire-and-forget via
            # `asyncio.create_task` so the lifespan does NOT block on it
            # — the API binds the port immediately and the prune runs
            # in the background. Failures inside the task are caught
            # there and log a warning; they cannot 500 startup.
            import asyncio as _asyncio

            from scripts.prune_checkpoints import prune

            async def _background_prune() -> None:
                try:
                    totals = await prune(ttl_days=settings.CHECKPOINT_TTL_DAYS, dry_run=False)
                    # Namespace under `prune_totals` to avoid any
                    # collision with reserved LogRecord attributes.
                    logger.info("startup.prune.done", extra={"prune_totals": totals})
                except Exception as prune_exc:  # noqa: BLE001
                    logger.warning(
                        "startup.prune.failed",
                        extra={"error_type": type(prune_exc).__name__},
                    )

            # Stash the task ref so Python's GC can't reap it
            # mid-execution (RUF006); also makes the task observable
            # from elsewhere for future graceful-shutdown handling.
            app.state.prune_task = _asyncio.create_task(_background_prune())
        except Exception as exc:  # noqa: BLE001 — never block startup on memory
            # Log only the exception type — psycopg connection errors can
            # embed the full DSN (incl. password) in str(exc).
            logger.warning(
                "startup.checkpointer.unavailable",
                extra={"error_type": type(exc).__name__},
            )
            # Falls back to the no-checkpointer graph already on app.state.

    # Non-fatal DB probe: surface a broken DB at boot instead of failing
    # on the first /ask. We warn rather than raise so a transient DB
    # blip doesn't prevent the API from starting — `/health` continues
    # to report degraded status if the DB stays down.
    from sqlalchemy import text

    from app.data.db import engine as db_engine

    try:
        async with db_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("startup.db.ok")
    except Exception as exc:  # noqa: BLE001 — never block startup on DB
        # str(exc) from DB drivers can embed the full DSN incl. password
        # (CWE-532) — redact before logging, keep the rest for debugging.
        logger.warning(
            "startup.db.unreachable",
            extra={"error_type": type(exc).__name__, "error": redact_secrets(str(exc))},
        )

    yield

    # Flush any batched Langfuse events before the process exits — the
    # SDK sends in a background thread, so an idle worker can otherwise
    # drop the last traces. No-op when Langfuse is disabled.
    from app.observability.langfuse import flush as langfuse_flush

    langfuse_flush()

    # Tear down the checkpointer connection if we opened one.
    if app.state.checkpointer_cm is not None:
        try:
            await app.state.checkpointer_cm.__aexit__(None, None, None)
            logger.info("shutdown.checkpointer.closed")
        except Exception as exc:  # noqa: BLE001 — best effort
            logger.warning(
                "shutdown.checkpointer.error",
                extra={"error_type": type(exc).__name__, "error": redact_secrets(str(exc))},
            )

    # Engine pool teardown — avoids "another operation in progress" on reload.
    from app.data.db import engine

    await engine.dispose()
    logger.info("shutdown: db engine disposed")


def create_app() -> FastAPI:
    app = FastAPI(
        title="SEC 10-K Agent",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={"detail": f"rate limit exceeded: {exc.detail}"},
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "Accept"],
    )

    # Per HLD 4: enforce per-API-key rate limits at the FastAPI layer.
    # `default_limits` on the Limiter only fire once SlowAPIMiddleware is
    # installed — without it the limiter is a no-op.
    app.add_middleware(SlowAPIMiddleware)

    # Registered last so it sits outermost in the stack — every request,
    # including rate-limit rejections, gets a request_id and lifecycle log.
    #
    # Implemented as raw ASGI (NOT `@app.middleware("http")`) because that
    # decorator builds on Starlette's BaseHTTPMiddleware, which buffers the
    # full response body in memory before forwarding it. That breaks SSE:
    # `EventSourceResponse` events get held until the generator is fully
    # exhausted (and in some Starlette versions hang outright). Raw ASGI
    # passes `send` straight through, so each event flushes immediately.
    app.add_middleware(_RequestLogMiddleware)

    app.include_router(router)
    return app


# Allow only the shape we actually want clients to send. Anything else
# (CRLF, oversized blob, non-ASCII) gets replaced with a fresh UUID so
# PrettyFormatter can't be tricked into splitting log lines and the
# header can't be used as an unbounded log-amplification vector.
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class _RequestLogMiddleware:
    """Raw ASGI middleware: logs request lifecycle + binds `request_id_var`,
    without buffering the response body (so SSE streams flush event-by-event)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        path = scope.get("path", "")
        client = scope.get("client")
        client_host = client[0] if client else None

        # Pull existing X-Request-ID header if present, else mint one.
        # Validate against a strict charset/length so a malicious client
        # can't smuggle CRLF into pretty-mode logs or inflate every log
        # line for the request with a 1MB header.
        headers = dict(scope.get("headers") or [])
        rid_bytes = headers.get(b"x-request-id")
        rid: str | None = None
        if rid_bytes is not None:
            try:
                candidate = rid_bytes.decode("ascii")
            except UnicodeDecodeError:
                candidate = ""
            if _REQUEST_ID_RE.match(candidate):
                rid = candidate
        if rid is None:
            rid = str(uuid.uuid4())
        token = request_id_var.set(rid)

        start = time.perf_counter()
        logger.info(
            "http.request.start",
            extra={"method": method, "path": path, "client": client_host},
        )

        status_holder = {"status": 0}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["status"] = message.get("status", 0)
                msg_headers = list(message.get("headers") or [])
                msg_headers.append((b"x-request-id", rid.encode()))
                message["headers"] = msg_headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception(
                "http.request.error",
                extra={
                    "method": method,
                    "path": path,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                },
            )
            request_id_var.reset(token)
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.info(
            "http.request.end",
            extra={
                "method": method,
                "path": path,
                "status": status_holder["status"],
                "duration_ms": duration_ms,
            },
        )
        request_id_var.reset(token)


app = create_app()


__all__ = ["app", "create_app"]
