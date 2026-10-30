"""Run unit, isolated PostgreSQL, or explicitly opted-in live test suites.

Examples (run from backend):
    uv run --extra dev python scripts/run_tests.py --suite unit
    uv run --extra dev python scripts/run_tests.py --suite database
    uv run --extra dev python scripts/run_tests.py --suite live
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("unit", "database", "all", "live", "corpus"), default="unit")
    parser.add_argument("--database-url")
    parser.add_argument("--test-database-url")
    args, pytest_args = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root.parent))
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Override .env database values for this process only. Do not let a
    # publishing checkout's Neon URL accidentally become a test target.
    os.environ["DATABASE_URL"] = args.database_url or os.environ.get(
        "DATABASE_URL", "postgresql+asyncpg://sec_agent:dev_only@localhost:5433/sec_agent")
    os.environ["TEST_DATABASE_URL"] = args.test_database_url or os.environ.get(
        "TEST_DATABASE_URL", "postgresql+asyncpg://sec_agent:dev_only@localhost:5433/sec_agent_test")
    if args.suite != "live":
        os.environ.setdefault("SEC_USER_AGENT", "ci-tests@example.com")
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY",
                     "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "CLERK_JWKS_URL", "CLERK_ISSUER"):
            os.environ[name] = ""
        for name in ("LANGGRAPH_CHECKPOINTER_ENABLED", "ASK_TOKEN_LIMITER_ENABLED",
                     "ANON_LIFETIME_ENABLED", "EXPOSE_TRACE_URLS"):
            os.environ[name] = "false"

    if args.suite in ("database", "all", "live"):
        import psycopg
        from sqlalchemy.engine import make_url
        from app.agent.memory import checkpoint_conn_string
        dev = make_url(os.environ["DATABASE_URL"])
        test = make_url(os.environ["TEST_DATABASE_URL"])
        if not (test.database or "").endswith("_test"):
            parser.error("The test database name must end in _test")
        if (dev.host, dev.port or 5432, dev.database) == (test.host, test.port or 5432, test.database):
            parser.error("The test database must differ from the application database")
        try:
            with psycopg.connect(checkpoint_conn_string(os.environ["TEST_DATABASE_URL"]), connect_timeout=5) as conn:
                conn.execute("SELECT 1")
        except psycopg.Error as exc:
            raise SystemExit("Test database is unreachable (" + type(exc).__name__ + "). Start the existing local PostgreSQL container and check the test URL.") from None

    if args.suite in ("unit", "database", "all"):
        # The first use may download public tokenizer vocabulary. No model
        # API is called; subsequent runs use tiktoken's local cache.
        import tiktoken
        tiktoken.get_encoding("cl100k_base")

    if args.suite == "live":
        from app.settings import settings
        if not settings.GROQ_API_KEY or not settings.GEMINI_API_KEY:
            parser.error("Live checks require private GROQ_API_KEY and GEMINI_API_KEY configuration")
        if settings.SEC_USER_AGENT == "ci-tests@example.com" or "@" not in settings.SEC_USER_AGENT:
            parser.error("Live SEC calls require SEC_USER_AGENT with your real contact email")

    import pytest
    class NoSkippedDatabaseChecks:
        def __init__(self):
            self.skipped = 0
        def pytest_runtest_logreport(self, report):
            if report.skipped:
                self.skipped += 1
        def pytest_sessionfinish(self, session, exitstatus):
            if self.skipped:
                session.exitstatus = 1
                reporter = session.config.pluginmanager.getplugin("terminalreporter")
                if reporter:
                    reporter.write_line("Database checks were skipped; refusing a false-green result.", red=True)

    if args.suite == "unit":
        selected = ["tests/unit", "../evals/tests", "-m", "not corpus"]
    elif args.suite == "database":
        selected = ["tests/integration", "-m", "not live"]
    elif args.suite == "all":
        selected = ["tests", "../evals/tests", "-m", "not live and not corpus"]
    elif args.suite == "corpus":
        selected = ["tests/unit/test_parser.py", "-m", "corpus", "--run-corpus"]
    else:
        selected = ["tests/integration", "-m", "live", "--run-live"]
    plugins = [NoSkippedDatabaseChecks()] if args.suite in ("database", "all") else []
    return pytest.main([*selected, "-q", "-ra", *pytest_args], plugins=plugins)


if __name__ == "__main__":
    raise SystemExit(main())
