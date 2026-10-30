"""Integration test fixtures — hard-isolated from the dev database.

These tests need a running Postgres with pgvector. They run **only** on
`settings.TEST_DATABASE_URL` (auto-derived from `DATABASE_URL` by
suffixing the db name with `_test`, see `app/settings.py`). The conftest
contains a TRIPWIRE that aborts the entire test session if the resolved
test URL equals `settings.DATABASE_URL` — silently shredding the dev
DB via `alembic downgrade base` + `TRUNCATE` must be impossible.

If the configured TEST_DATABASE_URL is unreachable they auto-skip — keeps
CI green when no DB is available. To run locally:

    docker compose up -d postgres
    cd backend && uv run pytest tests/integration -v

The first run auto-creates the test database if missing (no manual
`CREATE DATABASE sec_agent_test;` required).
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import AsyncIterator
from datetime import date
from decimal import Decimal

import asyncpg
import pytest
import pytest_asyncio
from alembic.config import Config
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from alembic import command
from app.data.models import EMBEDDING_DIM, Chunk, Company, Filing, XbrlFact
from app.settings import settings

# Modules that bind `async_session_factory` at import time
# (`from app.data.db import async_session_factory`). The session-scoped
# `_isolate_test_db` fixture reaches into each and reassigns the
# attribute so agent writes during tests go to the test DB. Adding a
# new module with the same import pattern means appending to this list.
_NODE_MODULES_USING_FACTORY: tuple[str, ...] = (
    "app.agent.nodes.fetch",
    "app.agent.nodes.locate",
    "app.agent.nodes.validate",
    "app.agent.nodes.retrieve",
    "app.data.ingestion",
)

# Snapshot the dev DATABASE_URL at module-import time, BEFORE
# `_isolate_test_db` redirects settings.DATABASE_URL to the test URL.
# The tripwire compares against THIS snapshot so it can't be silently
# bypassed by the redirect itself.
_DEV_DATABASE_URL_AT_IMPORT: str = settings.DATABASE_URL


def _resolve_test_url() -> str:
    """The test URL must always be set (settings auto-derives it).
    Raising here surfaces a config bug loudly rather than silently
    falling back to DATABASE_URL."""
    url = settings.TEST_DATABASE_URL
    if not url:
        raise RuntimeError(
            "TEST_DATABASE_URL is not set and could not be derived from "
            "DATABASE_URL. Refusing to run integration tests. Set "
            "TEST_DATABASE_URL explicitly."
        )
    return url


def _assert_safe_to_wipe(url: str) -> None:
    """The tripwire. Refuses to proceed if the resolved test URL could
    target the dev DB. Belt-and-braces: checks both equality with
    DATABASE_URL and that the db name ends in `_test`."""
    # Compare against the snapshot captured at module import — NOT the
    # live `settings.DATABASE_URL`, which `_isolate_test_db` rebinds to
    # the test URL on purpose (so alembic env.py picks it up). Without
    # the snapshot the tripwire would falsely fire after redirect.
    if url == _DEV_DATABASE_URL_AT_IMPORT:
        raise RuntimeError(
            "TEST_DATABASE_URL == DATABASE_URL (dev) — refusing to run "
            "integration tests against the dev DB (would wipe it via "
            "`alembic downgrade base` + TRUNCATE). Set TEST_DATABASE_URL to "
            "a distinct database (e.g. suffix the dev db name with `_test`)."
        )
    parsed = make_url(url)
    db_name = parsed.database or ""
    if not db_name.endswith("_test"):
        raise RuntimeError(
            f"Refusing to run integration tests against database {db_name!r} "
            "— the test database name must end in `_test`. Adjust "
            "TEST_DATABASE_URL or rename the database."
        )


async def _ensure_test_db_exists(url: str) -> None:
    """Auto-create the test database if missing — saves a manual
    `CREATE DATABASE` step on a fresh dev environment. Connects to the
    admin `postgres` database with the same credentials."""
    parsed = make_url(url)
    target_db = parsed.database
    if not target_db:
        raise RuntimeError(f"Cannot derive db name from {url!r}")
    admin = await asyncpg.connect(
        host=parsed.host,
        port=parsed.port or 5432,
        user=parsed.username,
        password=parsed.password,
        database="postgres",
    )
    try:
        exists = await admin.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", target_db)
        if not exists:
            # CREATE DATABASE cannot run in a transaction; asyncpg
            # auto-commits each statement on a plain connection.
            await admin.execute(f'CREATE DATABASE "{target_db}"')
    finally:
        await admin.close()


def db_reachable(url: str | None = None, timeout: float = 1.0) -> bool:
    """Probe the test DB (NOT the dev DB) by opening + closing one
    async connection. Returns True only when the DB exists, accepts our
    credentials, and is actually a Postgres."""
    target = url or _resolve_test_url()

    async def _probe() -> bool:
        # Try to ensure the DB exists first (best-effort, swallow errors).
        try:
            await _ensure_test_db_exists(target)
        except Exception:
            return False
        engine = create_async_engine(target, connect_args={"timeout": timeout})
        try:
            async with engine.connect() as conn:
                await conn.exec_driver_sql("SELECT 1")
            return True
        except Exception:
            return False
        finally:
            await engine.dispose()

    try:
        return asyncio.run(_probe())
    except RuntimeError:
        # Already inside a loop (rare in test collection); fall back to optimistic.
        return True


SKIP_REASON = (
    f"Postgres test DB not reachable at {settings.TEST_DATABASE_URL!r}. "
    "Run `docker compose up -d postgres` to enable integration tests."
)


@pytest.fixture(scope="session", autouse=True)
def _isolate_test_db():
    """Run BEFORE any integration test. Reroutes the app's module-level
    DB engine and every node module's imported session factory to the
    test DB. Without this rebind, agent code under test would still
    write to the dev DB (because the node modules captured the original
    factory at import time)."""
    test_url = _resolve_test_url()
    _assert_safe_to_wipe(test_url)

    # 1. Temporarily redirect settings.DATABASE_URL so alembic env.py
    #    (which reads it at import time) connects to the test DB.
    original_database_url = settings.DATABASE_URL
    settings.DATABASE_URL = test_url

    # 2. Dispose the existing module-level engine and replace it +
    #    its session factory with versions bound to the test URL.
    from app.data import db as db_module

    # Best-effort sync dispose: schedule on a temporary loop.
    asyncio.run(db_module.engine.dispose())

    new_engine = create_async_engine(
        test_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=1800,
        echo=settings.DB_ECHO,
    )
    new_factory = async_sessionmaker(new_engine, expire_on_commit=False)
    db_module.engine = new_engine
    db_module.async_session_factory = new_factory

    # 3. Reassign the name in every module that did
    #    `from app.data.db import async_session_factory`. Without this,
    #    those modules keep their stale reference to the dev-bound factory.
    for mod_name in _NODE_MODULES_USING_FACTORY:
        mod = importlib.import_module(mod_name)
        if hasattr(mod, "async_session_factory"):
            mod.async_session_factory = new_factory

    yield

    # Teardown — restore (best-effort; session is ending anyway).
    settings.DATABASE_URL = original_database_url


@pytest_asyncio.fixture(scope="session")
async def _migrated_db(_isolate_test_db) -> AsyncIterator[None]:
    """Run `alembic upgrade head` once per session against the TEST DB.
    The `_isolate_test_db` fixture has already redirected
    settings.DATABASE_URL → TEST_DATABASE_URL, so alembic's env.py
    picks up the test URL."""
    cfg = Config("alembic.ini")
    await asyncio.to_thread(command.downgrade, cfg, "base")
    await asyncio.to_thread(command.upgrade, cfg, "head")
    yield


@pytest_asyncio.fixture
async def session(_migrated_db) -> AsyncIterator[AsyncSession]:
    """Per-test async session. Truncates all data tables before yielding."""
    test_url = _resolve_test_url()
    _assert_safe_to_wipe(test_url)  # defence-in-depth at every truncate
    engine = create_async_engine(test_url)
    async with engine.begin() as conn:
        await conn.exec_driver_sql(
            "TRUNCATE TABLE query_log, answer_cache, xbrl_facts, chunks, filings, companies "
            "RESTART IDENTITY CASCADE"
        )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


@pytest_asyncio.fixture
async def agent_session(_migrated_db) -> AsyncIterator[AsyncSession]:
    """Session for agent tests. Depends on `_migrated_db` so the schema is
    guaranteed present, then truncates and seeds a minimal AAPL fixture
    (company + FY2024 filing + revenue XBRL fact) inline. Tests are fully
    self-contained — no dependency on `seed_dev` or external XBRL ingestion.

    Disposes the global engine (`app.data.db.engine` — already rebound
    to the test DB by `_isolate_test_db`) before yielding so the agent's
    own DB calls create a fresh connection pool bound to this test's
    event loop — pytest-asyncio resets the loop per test, and a pool
    from a prior loop produces 'another operation in progress'.
    """
    from app.data import db as db_module

    await db_module.engine.dispose()

    test_url = _resolve_test_url()
    _assert_safe_to_wipe(test_url)
    engine = create_async_engine(test_url)
    async with engine.begin() as conn:
        await conn.exec_driver_sql(
            "TRUNCATE TABLE query_log, answer_cache, xbrl_facts, chunks, filings, companies "
            "RESTART IDENTITY CASCADE"
        )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        s.add(Company(cik="0000320193", ticker="AAPL", name="Apple Inc."))
        await s.flush()
        s.add(
            Filing(
                accession_no="0000320193-24-000123",
                cik="0000320193",
                form_type="10-K",
                filing_date=date(2024, 11, 1),
                period_of_report=date(2024, 9, 28),
                fiscal_year=2024,
                primary_doc_url="https://www.sec.gov/Archives/edgar/data/320193/aapl-10k-2024.htm",
                ingestion_status="embedded",
            )
        )
        await s.flush()
        s.add(
            XbrlFact(
                cik="0000320193",
                concept="revenue",
                unit="USD",
                value=Decimal("391035000000"),
                period_start=date(2023, 10, 1),
                period_end=date(2024, 9, 28),
                fiscal_year=2024,
                fiscal_period="FY",
                accession_no="0000320193-24-000123",
                form="10-K",
            )
        )
        # Placeholder Item 1A chunk so the locate node sees the section
        # as parsed. Tests that exercise the qualitative path mock
        # `vector_search`, so the zero-vector embedding is never queried.
        s.add(
            Chunk(
                accession_no="0000320193-24-000123",
                cik="0000320193",
                item="1A",
                seq=0,
                text="placeholder",
                token_count=1,
                embedding=[0.0] * EMBEDDING_DIM,
                char_start=0,
                char_end=11,
            )
        )
        await s.commit()
        yield s
    await engine.dispose()


@pytest.fixture(autouse=True)
def offline_graph_services(request, monkeypatch):
    if request.node.get_closest_marker("live"):
        yield
        return
    from types import SimpleNamespace
    from app.agent.nodes import retrieve, refusal_composer
    from app.api import token_budget

    monkeypatch.setattr(settings, "ASK_TOKEN_LIMITER_ENABLED", False)
    monkeypatch.setattr(settings, "ANON_LIFETIME_ENABLED", False)
    monkeypatch.setattr(settings, "LANGGRAPH_CHECKPOINTER_ENABLED", False)
    monkeypatch.setattr(settings, "EXPOSE_TRACE_URLS", False)
    monkeypatch.setattr(settings, "PLAN_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setattr(settings, "VALIDATE_MODEL", "openai/gpt-oss-120b")
    async def one_query(question):
        return [question]
    class RefusalChat:
        async def ainvoke(self, prompt):
            return SimpleNamespace(content="This question cannot be answered from the loaded filings.")
    monkeypatch.setattr(retrieve, "_decompose_question", one_query)
    monkeypatch.setattr(refusal_composer, "get_chat_model", lambda *a, **kw: RefusalChat())
    token_budget.BUDGET._buckets.clear()
    token_budget.REQUESTS._buckets.clear()
    yield
    token_budget.BUDGET._buckets.clear()
    token_budget.REQUESTS._buckets.clear()
