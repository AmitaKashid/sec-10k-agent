"""Focused test of routes._finalize_budget — reserve, record usage, finalize,
then assert the bucket's used count reflects (input_tokens + node_tokens)."""

from __future__ import annotations

from app.api import usage_tracker
from app.api.auth import Identity
from app.api.routes import _finalize_budget
from app.api.token_budget import BUDGET


def setup_function():
    BUDGET._buckets.clear()
    usage_tracker._USAGE.clear()


async def test_finalize_replaces_estimate_with_actual_from_tracker(monkeypatch):
    from app.settings import settings

    monkeypatch.setattr(settings, "ASK_TOKEN_LIMITER_ENABLED", True)
    monkeypatch.setattr(settings, "ASK_ANON_TOKENS_PER_MIN", 100_000)

    identity = Identity(tier="anon", key="9.9.9.9")
    # Reserve a generous estimate up front (as /ask does).
    reservation, used_after_reserve, _ = await BUDGET.check_and_reserve(identity, est_tokens=5000)
    assert reservation is not None
    assert used_after_reserve == 5000

    request_id = "req-fin-1"
    # Two LLM calls recorded by nodes.
    usage_tracker.record(request_id, "plan", input_tokens=100, output_tokens=50)
    usage_tracker.record(request_id, "validate", input_tokens=800, output_tokens=200)

    # A short question; tiktoken's cl100k counts it deterministically.
    question = "What was Apple's revenue in 2024?"
    await _finalize_budget(request_id, reservation, question)

    # After finalize: budget contains input_tokens + node totals.
    # node total = (100+50) + (800+200) = 1150
    # input_tokens: count via tiktoken — verify total = input + 1150.
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    expected = len(enc.encode(question)) + 1150
    actual = await BUDGET.used(identity)
    assert actual == expected
    # And the tracker bucket was popped.
    assert usage_tracker.collect(request_id) == []


async def test_finalize_with_no_recorded_usage_charges_input_only(monkeypatch):
    """When the LLM provider doesn't surface usage (or no nodes ran),
    finalize still charges the input_tokens — never under-counts that
    floor."""
    from app.settings import settings

    monkeypatch.setattr(settings, "ASK_TOKEN_LIMITER_ENABLED", True)
    monkeypatch.setattr(settings, "ASK_ANON_TOKENS_PER_MIN", 10_000)

    identity = Identity(tier="anon", key="9.9.9.9")
    reservation, _, _ = await BUDGET.check_and_reserve(identity, est_tokens=3000)
    assert reservation is not None

    question = "hi"
    await _finalize_budget("req-fin-2", reservation, question)

    import tiktoken

    expected = len(tiktoken.get_encoding("cl100k_base").encode(question))
    assert await BUDGET.used(identity) == expected
