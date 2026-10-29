"""Real SDK export tests, using only an in-memory exporter and dummy credentials."""
import os
import asyncio
import uuid
from types import SimpleNamespace

import pytest
import httpx
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.observability import langfuse as tracing


@pytest.fixture
def exported(monkeypatch):
    for key in list(os.environ):
        if key.lower().endswith("_proxy"):
            monkeypatch.delenv(key, raising=False)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    client = Langfuse(public_key="pk-test-" + uuid.uuid4().hex,
                      secret_key="sk-test", base_url="https://example.invalid",
                      tracer_provider=provider, span_exporter=InMemorySpanExporter(),
                      httpx_client=httpx.Client(trust_env=False, transport=httpx.MockTransport(
                          lambda request: httpx.Response(200, json={"data": [{"id": "project-test", "name": "test"}]}))))
    monkeypatch.setattr(client.api.projects, "get", lambda: SimpleNamespace(
        data=[SimpleNamespace(id="project-test")]))
    monkeypatch.setattr(tracing, "get_client", lambda: client)
    monkeypatch.setattr(tracing.settings, "LANGFUSE_CAPTURE_IO", True)
    yield client, exporter
    provider.shutdown()


def test_real_sdk_exports_parented_spans_and_session(exported):
    client, exporter = exported
    with tracing.request_trace("request-1", input="question", session_id="conversation"):
        with tracing.node_span("node:plan", model="test-model"):
            pass
        tracing.update_trace_output("answer")
        url = tracing.current_trace_url()
        assert client.create_trace_id(seed="request-1") in url
    client.flush()
    spans = {s.name: s for s in exporter.get_finished_spans()}
    root, node = spans["ask"], spans["node:plan"]
    assert node.parent.span_id == root.context.span_id
    assert node.context.trace_id == root.context.trace_id
    assert root.attributes["langfuse.observation.input"] == "question"
    assert root.attributes["langfuse.observation.output"] == "answer"
    assert root.attributes["session.id"] == "conversation"
    assert tracing.current_trace_url() is None


def test_capture_disabled_keeps_content_out(exported, monkeypatch):
    client, exporter = exported
    monkeypatch.setattr(tracing.settings, "LANGFUSE_CAPTURE_IO", False)
    with tracing.request_trace("private", input="secret question"):
        tracing.update_trace_output("secret answer")
    client.flush()
    attributes = dict(exporter.get_finished_spans()[0].attributes)
    assert "secret question" not in str(attributes)
    assert "secret answer" not in str(attributes)


def test_disabled_tracing_does_not_construct_client(monkeypatch):
    tracing.get_client.cache_clear()
    monkeypatch.setattr(tracing.settings, "LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setattr(tracing.settings, "LANGFUSE_SECRET_KEY", "")
    with tracing.request_trace("disabled") as root:
        assert root is None
        with tracing.node_span("node") as child:
            assert child is None
    tracing.flush()
    tracing.get_client.cache_clear()


def test_application_exception_propagates_and_closes_spans(exported):
    client, exporter = exported
    error = ValueError("application failure")
    with pytest.raises(ValueError) as caught:
        with tracing.request_trace("failing"):
            with tracing.node_span("node"):
                raise error
    assert caught.value is error
    client.flush()
    assert len(exporter.get_finished_spans()) == 2
    assert tracing.current_trace_url() is None


def test_sdk_start_failure_does_not_swallow_application_error(monkeypatch):
    class Broken:
        def create_trace_id(self, **kwargs):
            raise RuntimeError("SDK failure")
    monkeypatch.setattr(tracing, "get_client", lambda: Broken())
    with pytest.raises(ValueError, match="app"):
        with tracing.request_trace("broken"):
            raise ValueError("app")


def test_flush_failure_does_not_break_request(monkeypatch):
    class Broken:
        def flush(self):
            raise RuntimeError("SDK flush failure")
    monkeypatch.setattr(tracing, "get_client", lambda: Broken())
    tracing.flush()


def test_concurrent_requests_do_not_share_trace(exported):
    client, exporter = exported
    async def one(name):
        with tracing.request_trace(name, session_id=name):
            await asyncio.sleep(0)
            with tracing.node_span(name):
                pass
    async def run():
        await asyncio.gather(one("a"), one("b"))
    asyncio.run(run())
    client.flush()
    spans = exporter.get_finished_spans()
    children = {s.name: s for s in spans if s.name in ("a", "b")}
    assert children["a"].context.trace_id != children["b"].context.trace_id
    roots = {s.context.trace_id: s for s in spans if s.name == "ask"}
    for name, child in children.items():
        root = roots[child.context.trace_id]
        assert child.parent.span_id == root.context.span_id
        assert root.attributes["session.id"] == name

