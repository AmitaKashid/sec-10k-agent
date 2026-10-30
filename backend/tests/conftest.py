"""Shared test isolation and explicit opt-in for network/corpus checks."""
from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-live", action="store_true", default=False,
                     help="Permit explicitly selected tests to call external services")
    parser.addoption("--run-corpus", action="store_true", default=False,
                     help="Run checks against downloaded real 10-K fixtures")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.get_closest_marker("live") and not config.getoption("--run-live"):
            item.add_marker(pytest.mark.skip(reason="Live checks require --run-live"))
        if item.get_closest_marker("corpus") and not config.getoption("--run-corpus"):
            item.add_marker(pytest.mark.skip(reason="Corpus checks require --run-corpus"))


@pytest.fixture(autouse=True)
def isolate_external_credentials(request, monkeypatch):
    if request.node.get_closest_marker("live"):
        return
    import os
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    from app.settings import settings
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY",
                 "GEMINI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
                 "CLERK_JWKS_URL", "CLERK_ISSUER"):
        monkeypatch.setattr(settings, name, "")
