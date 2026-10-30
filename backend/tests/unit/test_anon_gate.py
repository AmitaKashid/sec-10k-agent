"""Unit tests for the anon_gate node.

Covers tier branching, allowlist case-insensitivity, meta passthrough
(no ticker), and that the refusal payload is shaped for the
refusal_composer downstream."""

from __future__ import annotations

import pytest

from app.agent.nodes.anon_gate import anon_gate
from app.agent.state import AgentState


@pytest.fixture(autouse=True)
def fixed_allowlist(monkeypatch):
    from app.settings import settings

    monkeypatch.setattr(
        settings,
        "ANON_TICKER_ALLOWLIST",
        ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    )


def _state(*, tier: str, ticker: str | None) -> AgentState:
    return AgentState(question="q", request_id="r", tier=tier, ticker=ticker)


async def test_user_tier_always_passes():
    out = await anon_gate(_state(tier="user", ticker="TSLA"))
    assert out == {}


async def test_anon_with_allowlisted_ticker_passes():
    out = await anon_gate(_state(tier="anon", ticker="AAPL"))
    assert out == {}


async def test_anon_case_insensitive_match():
    out = await anon_gate(_state(tier="anon", ticker="aapl"))
    assert out == {}


async def test_anon_no_ticker_passes_through():
    """Meta/concierge questions resolve no ticker — must not be blocked."""
    out = await anon_gate(_state(tier="anon", ticker=None))
    assert out == {}


async def test_anon_with_disallowed_ticker_refuses():
    out = await anon_gate(_state(tier="anon", ticker="TSLA"))
    assert out["question_type"] == "REFUSE"
    assert out["refusal_code"] == "ANON_TICKER_NOT_ALLOWED"
    assert "TSLA" in out["refusal_reason"]
    # The composer downstream uses these to build user-facing prose.
    for allowed in ("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"):
        assert allowed in out["refusal_reason"]
