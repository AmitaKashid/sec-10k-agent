"""Langfuse v4 wrapper — the hard no-op contract.

Keys unset (CI/test default) ⇒ zero client, zero network, every helper
a safe no-op. Keys set ⇒ client built once (mocked — no real network).
See docs/langfuse_observability_plan.md §3a (Option A) and §7.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar

import pytest

from app.observability import langfuse as lf


@pytest.fixture(autouse=True)
def _clear_client_cache(monkeypatch):
    monkeypatch.setattr(lf.settings, "LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setattr(lf.settings, "LANGFUSE_SECRET_KEY", "")
    monkeypatch.setattr(lf.settings, "LANGFUSE_CAPTURE_IO", True)
    # get_client is lru_cached; reset around every test so monkeypatched
    # settings take effect and state never leaks between cases.
    lf.get_client.cache_clear()
    yield
    lf.get_client.cache_clear()


# --- disabled-by-default (no keys) ---------------------------------------


def test_disabled_when_keys_unset(monkeypatch):
    monkeypatch.setattr(lf.settings, "LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setattr(lf.settings, "LANGFUSE_SECRET_KEY", "")
    assert lf.langfuse_enabled() is False
    assert lf.get_client() is None
    # No active trace + disabled ⇒ None, no raise.
    assert lf.current_trace_url() is None
    # Flush is a no-op and must not raise.
    lf.flush()


def test_disabled_when_only_one_key_set(monkeypatch):
    monkeypatch.setattr(lf.settings, "LANGFUSE_PUBLIC_KEY", "pk-123")
    monkeypatch.setattr(lf.settings, "LANGFUSE_SECRET_KEY", "")
    assert lf.langfuse_enabled() is False
    assert lf.get_client() is None


# --- enabled (keys set, client mocked — no network) ----------------------


def test_client_built_once_when_enabled(monkeypatch):
    monkeypatch.setattr(lf.settings, "LANGFUSE_PUBLIC_KEY", "pk-123")
    monkeypatch.setattr(lf.settings, "LANGFUSE_SECRET_KEY", "sk-123")

    built = []

    class _FakeLangfuse:
        def __init__(self, **kwargs):
            built.append(kwargs)

        def flush(self):
            self.flushed = True

    # Patch the symbol at its source so the local import inside
    # get_client() picks up the fake (no real Langfuse construction).
    monkeypatch.setattr("langfuse.Langfuse", _FakeLangfuse)

    assert lf.langfuse_enabled() is True
    c1 = lf.get_client()
    c2 = lf.get_client()
    assert isinstance(c1, _FakeLangfuse)
    assert c1 is c2  # lru_cache ⇒ one client per process
    assert len(built) == 1
    assert built[0]["public_key"] == "pk-123"
    assert built[0]["secret_key"] == "sk-123"

    lf.flush()  # routes through the fake, must not raise


def test_client_init_failure_degrades_to_none(monkeypatch):
    monkeypatch.setattr(lf.settings, "LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setattr(lf.settings, "LANGFUSE_SECRET_KEY", "sk")

    def _boom(**_kwargs):
        raise RuntimeError("simulated SDK init failure")

    monkeypatch.setattr("langfuse.Langfuse", _boom)
    # Observability must never break /ask: init failure ⇒ None, no raise.
    assert lf.get_client() is None
    assert lf.current_trace_url() is None
    lf.flush()


def test_flush_swallows_sdk_errors(monkeypatch):
    monkeypatch.setattr(lf.settings, "LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setattr(lf.settings, "LANGFUSE_SECRET_KEY", "sk")

    class _FlakyClient:
        def __init__(self, **_kwargs):
            pass

        def flush(self):
            raise RuntimeError("network down")

    monkeypatch.setattr("langfuse.Langfuse", _FlakyClient)
    # A flush failure must not propagate into the request path.
    lf.flush()


def test_module_import_has_no_side_effects(monkeypatch):
    # Importing the module must not construct a client. Force the
    # "no keys" state explicitly so the test doesn't depend on whether
    # the developer has Langfuse configured in their local .env.
    monkeypatch.setattr(lf.settings, "LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setattr(lf.settings, "LANGFUSE_SECRET_KEY", "")
    lf.get_client.cache_clear()
    assert lf.get_client() is None


# --- request_trace / node_span / current_trace_url (Stage B) -------------


class _FakeSpan:
    def __init__(self):
        self.ended = False
        self.updates: list[dict] = []

    def update(self, **kw):
        self.updates.append(kw)

    def end(self):
        self.ended = True


class _FakeTrace:
    def __init__(self):
        self.spans: list[tuple[str, dict]] = []
        self._span_objs: list[_FakeSpan] = []
        # Captures every trace.update(...) call so tests can assert the
        # final answer reached the dashboard's Output panel.
        self.updates: list[dict] = []

    def start_observation(self, name, as_type=None, metadata=None):
        self.spans.append((name, metadata or {}))
        s = _FakeSpan()
        self._span_objs.append(s)
        return s

    def update(self, **kw):
        self.updates.append(kw)

    def end(self):
        self.ended = True


class _FakeClientWithTrace:
    def __init__(self, **_kw):
        self.traces: list[dict] = []
        self.last_trace: _FakeTrace | None = None
        self.sessions = []

    @staticmethod
    def create_trace_id(*, seed):
        return hashlib.sha256(seed.encode()).hexdigest()[:32]

    def get_trace_url(self, **kwargs):
        return "https://cloud.langfuse.com/trace/abc"

    def start_observation(self, **kw):
        self.sessions.append(_session.get())
        self.traces.append(kw)
        self.last_trace = _FakeTrace()
        return self.last_trace

    def flush(self):
        pass


_session = ContextVar("test-langfuse-session", default=None)


@contextmanager
def _propagate(**attributes):
    token = _session.set(attributes.get("session_id"))
    try:
        yield
    finally:
        _session.reset(token)


def _enable(monkeypatch):
    monkeypatch.setattr("langfuse.propagate_attributes", _propagate)
    monkeypatch.setattr(lf.settings, "LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setattr(lf.settings, "LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setattr("langfuse.Langfuse", _FakeClientWithTrace)


def test_disabled_request_trace_is_noop_yields_none(monkeypatch):
    monkeypatch.setattr(lf.settings, "LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setattr(lf.settings, "LANGFUSE_SECRET_KEY", "")
    with lf.request_trace("req-1") as t:
        assert t is None
        # node_span must also no-op (no active trace) and not raise.
        with lf.node_span("node:plan", ticker="AAPL") as s:
            assert s is None
    assert lf.current_trace_url() is None


def test_enabled_request_trace_creates_trace_and_spans(monkeypatch):
    _enable(monkeypatch)
    with lf.request_trace("req-42", metadata={"mode": "sync"}) as trace:
        assert trace is not None
        # Request ID remains in metadata; v4 uses a 32-hex-character trace ID.
        client = lf.get_client()
        assert client.traces[0]["trace_context"]["trace_id"] == client.create_trace_id(seed="req-42")
        assert client.traces[0]["metadata"] == {"mode": "sync", "request_id": "req-42"}
        # url readable while the trace context is active...
        assert lf.current_trace_url() == "https://cloud.langfuse.com/trace/abc"
        with lf.node_span("node:plan", ticker="AAPL"):
            pass
        assert trace.spans == [("node:plan", {"ticker": "AAPL"})]
        assert trace._span_objs[0].ended is True
    # ...and None again once the context is closed (contextvar reset).
    assert lf.current_trace_url() is None


def test_node_span_marks_error_then_reraises(monkeypatch):
    _enable(monkeypatch)
    with lf.request_trace("req-err") as trace:
        with pytest.raises(ValueError, match="boom"):
            with lf.node_span("node:validate"):
                raise ValueError("boom")
        span = trace._span_objs[0]
        assert span.ended is True
        assert span.updates and span.updates[0]["level"] == "ERROR"
        assert span.updates[0]["status_message"] == "ValueError"


def test_trace_start_failure_degrades_to_none(monkeypatch):
    _enable(monkeypatch)

    class _BoomClient:
        def __init__(self, **_kw):
            pass

        def trace(self, **_kw):
            raise RuntimeError("trace API down")

        def flush(self):
            pass

    monkeypatch.setattr("langfuse.Langfuse", _BoomClient)
    # request_trace must still yield (None) so the request proceeds.
    with lf.request_trace("req-x") as t:
        assert t is None
        with lf.node_span("node:plan") as s:
            assert s is None
    assert lf.current_trace_url() is None


# --- trace input / session_id / output (Q&A + Sessions wiring) -----------


def test_request_trace_forwards_input_and_session_id(monkeypatch):
    _enable(monkeypatch)
    with lf.request_trace(
        "req-qa",
        input="What was AAPL revenue in FY2024?",
        session_id="thread-abc",
        metadata={"mode": "sync"},
    ):
        client = lf.get_client()
        kw = client.traces[0]
        # Both fields reach the SDK call — populates Input + Sessions.
        assert kw["input"] == "What was AAPL revenue in FY2024?"
        assert client.sessions[0] == "thread-abc"
        assert kw["metadata"] == {"mode": "sync", "request_id": "req-qa"}


def test_request_trace_omits_session_id_when_none(monkeypatch):
    _enable(monkeypatch)
    # No session_id supplied (e.g. checkpointer off, no thread).
    with lf.request_trace("req-no-session", input="hi"):
        client = lf.get_client()
        kw = client.traces[0]
        # Sessions view must not be cluttered with sessionless groupings —
        # the key must be absent, not present-and-None.
        assert client.sessions[0] is None
        assert kw["input"] == "hi"


def test_update_trace_output_sets_output_on_active_trace(monkeypatch):
    _enable(monkeypatch)
    with lf.request_trace("req-out", input="hi") as trace:
        lf.update_trace_output("Hello! I'm an SEC 10-K analyst.")
        assert trace.updates == [{"output": "Hello! I'm an SEC 10-K analyst."}]


def test_update_trace_output_is_noop_when_output_empty(monkeypatch):
    _enable(monkeypatch)
    with lf.request_trace("req-empty") as trace:
        lf.update_trace_output(None)
        lf.update_trace_output("")
        assert trace.updates == []


def test_update_trace_output_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(lf.settings, "LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setattr(lf.settings, "LANGFUSE_SECRET_KEY", "")
    # No active trace + disabled — must not raise.
    lf.update_trace_output("anything")


def test_update_trace_output_is_noop_outside_request_trace(monkeypatch):
    _enable(monkeypatch)
    # Enabled but no `with request_trace(...)` open ⇒ no trace handle ⇒ no-op.
    lf.update_trace_output("anything")


# --- LANGFUSE_CAPTURE_IO kill-switch -------------------------------------


def test_capture_io_off_omits_input_keeps_session_and_trace(monkeypatch):
    """When the operator disables I/O capture, the trace and Sessions
    grouping must still flow (latency/observability stays intact) — only
    the user-content fields (input/output) get stripped from egress."""
    _enable(monkeypatch)
    monkeypatch.setattr(lf.settings, "LANGFUSE_CAPTURE_IO", False)
    with lf.request_trace(
        "req-noio",
        input="What was AAPL revenue in FY2024?",
        session_id="thread-abc",
        metadata={"mode": "sync"},
    ):
        client = lf.get_client()
        kw = client.traces[0]
        # input must NOT have left the process...
        assert "input" not in kw
        # ...but trace + session grouping + metadata still flow.
        assert kw["trace_context"]["trace_id"] == client.create_trace_id(seed="req-noio")
        assert client.sessions[0] == "thread-abc"
        assert kw["metadata"] == {"mode": "sync", "request_id": "req-noio"}


def test_capture_io_off_blocks_update_trace_output(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(lf.settings, "LANGFUSE_CAPTURE_IO", False)
    with lf.request_trace("req-noio2") as trace:
        lf.update_trace_output("the agent's final answer")
        # Kill-switch wins — no trace.update(output=...) was issued.
        assert trace.updates == []


def test_capture_io_on_sends_input_and_output(monkeypatch):
    """Regression guard: with the flag on (the default), the existing
    behavior — input + output captured — is unchanged."""
    _enable(monkeypatch)
    monkeypatch.setattr(lf.settings, "LANGFUSE_CAPTURE_IO", True)
    with lf.request_trace("req-yesio", input="hi") as trace:
        lf.update_trace_output("Hello!")
        client = lf.get_client()
        assert client.traces[0]["input"] == "hi"
        assert trace.updates == [{"output": "Hello!"}]
