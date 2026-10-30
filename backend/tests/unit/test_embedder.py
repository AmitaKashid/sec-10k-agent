"""Unit tests for app.data.embedder (LLD 3 test plan).

The OpenAI client is monkeypatched at the module level so no network
calls happen.
"""

from __future__ import annotations

import httpx
import pytest
from openai import APIError, RateLimitError

from app.data import embedder
from app.data.chunker import Chunk

EMBED_DIM = 1536


@pytest.fixture(autouse=True)
def openai_branch(monkeypatch):
    # This file covers the retained OpenAI adapter. Gemini HTTP behavior
    # is covered separately in test_gemini_embeddings.py.
    monkeypatch.setattr(embedder.settings, "EMBEDDING_PROVIDER", "openai")


class _FakeData:
    def __init__(self, vec: list[float]):
        self.embedding = vec


class _FakeResponse:
    def __init__(self, vecs: list[list[float]]):
        self.data = [_FakeData(v) for v in vecs]


class _FakeEmbeddings:
    def __init__(self, behavior):
        self._behavior = behavior
        self.calls: list[list[str]] = []

    async def create(self, *, model: str, input: list[str]) -> _FakeResponse:  # noqa: A002
        self.calls.append(list(input))
        return await self._behavior(input)


class _FakeClient:
    def __init__(self, behavior):
        self.embeddings = _FakeEmbeddings(behavior)


def _install_client(monkeypatch, behavior) -> _FakeClient:
    fake = _FakeClient(behavior)
    monkeypatch.setattr(embedder, "_client", lambda: fake)
    return fake


async def _ok(vec_factory):
    async def behavior(inputs: list[str]) -> _FakeResponse:
        return _FakeResponse([vec_factory(i, t) for i, t in enumerate(inputs)])

    return behavior


@pytest.mark.asyncio
async def test_single_input_single_vector(monkeypatch):
    behavior = await _ok(lambda i, t: [0.1] * EMBED_DIM)
    fake = _install_client(monkeypatch, behavior)

    out = await embedder.embed_texts(["hello"])
    assert len(out) == 1
    assert len(out[0]) == EMBED_DIM
    assert len(fake.embeddings.calls) == 1


@pytest.mark.asyncio
async def test_batched_calls_for_large_input(monkeypatch):
    # tag each vector with its input string so we can verify ordering is preserved
    async def behavior(inputs):
        return _FakeResponse([[float(hash(t) % 1000)] * EMBED_DIM for t in inputs])

    fake = _install_client(monkeypatch, behavior)
    texts = [f"text-{i}" for i in range(1000)]
    out = await embedder.embed_texts(texts, batch_size=256)

    assert len(out) == 1000
    assert [len(c) for c in fake.embeddings.calls] == [256, 256, 256, 232]
    # ordering: the first element of each returned vector encodes the input string's hash
    for t, v in zip(texts, out, strict=True):
        assert v[0] == float(hash(t) % 1000)


@pytest.mark.asyncio
async def test_retries_on_429_then_succeeds(monkeypatch):
    state = {"calls": 0}

    async def behavior(inputs):
        state["calls"] += 1
        if state["calls"] <= 2:
            raise RateLimitError(
                "rate limited",
                response=httpx.Response(429, request=httpx.Request("POST", "http://x")),
                body=None,
            )
        return _FakeResponse([[0.5] * EMBED_DIM for _ in inputs])

    _install_client(monkeypatch, behavior)
    # speed up tenacity in tests
    monkeypatch.setattr(embedder, "_RETRYABLE", embedder._RETRYABLE)
    out = await embedder.embed_texts(["a", "b"])
    assert state["calls"] == 3
    assert len(out) == 2


@pytest.mark.asyncio
async def test_exhausts_retries_raises(monkeypatch):
    async def behavior(inputs):
        raise APIError(
            "boom",
            request=httpx.Request("POST", "http://x"),
            body=None,
        )

    _install_client(monkeypatch, behavior)
    with pytest.raises(APIError):
        await embedder.embed_texts(["x"])


@pytest.mark.asyncio
async def test_empty_input_no_api_call(monkeypatch):
    called = {"n": 0}

    async def behavior(inputs):
        called["n"] += 1
        return _FakeResponse([])

    _install_client(monkeypatch, behavior)
    out = await embedder.embed_texts([])
    assert out == []
    assert called["n"] == 0


@pytest.mark.asyncio
async def test_embed_chunks_mutates_in_place(monkeypatch):
    async def behavior(inputs):
        return _FakeResponse([[float(len(t))] * EMBED_DIM for t in inputs])

    _install_client(monkeypatch, behavior)
    chunks = [
        Chunk(item="1A", seq=0, text="aa", token_count=1, char_start=0, char_end=2),
        Chunk(item="1A", seq=1, text="bbbb", token_count=1, char_start=2, char_end=6),
    ]
    out = await embedder.embed_chunks(chunks)
    assert out is chunks
    assert chunks[0].embedding is not None
    assert chunks[0].embedding[0] == 2.0
    assert chunks[1].embedding[0] == 4.0
    assert len(chunks[0].embedding) == EMBED_DIM


@pytest.mark.asyncio
async def test_gemini_dispatch_does_not_construct_openai_client(monkeypatch):
    from app.data import gemini_embeddings
    monkeypatch.setattr(embedder.settings, "EMBEDDING_PROVIDER", "gemini")
    calls = []
    async def google(texts):
        calls.append(list(texts))
        return [[float(i)] * EMBED_DIM for i, _ in enumerate(texts)], 0
    monkeypatch.setattr(gemini_embeddings, "embed_with_usage", google)
    monkeypatch.setattr(embedder, "_client", lambda: pytest.fail("OpenAI client constructed for Gemini"))
    vectors = await embedder.embed_texts(["first", "second"])
    assert calls == [["first", "second"]]
    assert [v[0] for v in vectors] == [0.0, 1.0]


@pytest.mark.asyncio
async def test_unknown_provider_fails_without_constructing_client(monkeypatch):
    monkeypatch.setattr(embedder.settings, "EMBEDDING_PROVIDER", "unsupported")
    monkeypatch.setattr(embedder, "_client", lambda: pytest.fail("Client constructed for invalid provider"))
    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        await embedder.embed_texts(["text"])
