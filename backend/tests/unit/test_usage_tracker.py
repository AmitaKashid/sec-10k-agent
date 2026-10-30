"""Unit tests for app.api.usage_tracker."""

from __future__ import annotations

from types import SimpleNamespace

from app.api import usage_tracker
from app.api.usage_tracker import (
    NodeUsage,
    collect,
    extract_langchain_usage,
    record,
    total_tokens,
)


def setup_function():
    usage_tracker._USAGE.clear()


def test_record_then_collect_returns_entries_and_pops():
    record("req-1", "plan", 100, 50)
    record("req-1", "validate", 800, 200)
    entries = collect("req-1")
    assert [e.node for e in entries] == ["plan", "validate"]
    assert [e.total for e in entries] == [150, 1000]
    # collect pops — second call yields empty
    assert collect("req-1") == []


def test_record_no_op_when_request_id_empty():
    record("", "plan", 100, 50)
    assert collect("") == []


def test_total_tokens_sums_correctly():
    entries = [
        NodeUsage(node="plan", input_tokens=10, output_tokens=5),
        NodeUsage(node="validate", input_tokens=100, output_tokens=200),
    ]
    assert total_tokens(entries) == 315


def test_extract_langchain_usage_modern_metadata():
    msg = SimpleNamespace(
        usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    )
    assert extract_langchain_usage(msg) == (100, 50)


def test_extract_langchain_usage_legacy_response_metadata():
    msg = SimpleNamespace(
        usage_metadata=None,
        response_metadata={"token_usage": {"prompt_tokens": 75, "completion_tokens": 25}},
    )
    assert extract_langchain_usage(msg) == (75, 25)


def test_extract_langchain_usage_returns_none_when_missing():
    msg = SimpleNamespace(usage_metadata=None, response_metadata=None)
    assert extract_langchain_usage(msg) is None


def test_extract_langchain_usage_returns_none_for_empty_metadata_dict():
    # Provider returned a `usage_metadata` dict but with neither
    # input_tokens nor output_tokens — treat as "no usage" instead of
    # recording a spurious (0, 0) entry that pollutes ask.tokens logs.
    msg = SimpleNamespace(usage_metadata={}, response_metadata=None)
    assert extract_langchain_usage(msg) is None


def test_extract_langchain_usage_handles_none_message():
    assert extract_langchain_usage(None) is None


def test_concurrent_request_ids_isolated():
    record("req-A", "plan", 10, 5)
    record("req-B", "plan", 100, 200)
    record("req-A", "validate", 50, 25)
    a = collect("req-A")
    b = collect("req-B")
    assert total_tokens(a) == 90  # 15 + 75
    assert total_tokens(b) == 300
