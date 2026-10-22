"""Provider compatibility tests; all HTTP is mocked."""

import json

import httpx
import pytest
import respx
from openai import APIError

from app.data import embedder, gemini_embeddings, ingestion
from app.settings import settings

URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent"


@pytest.fixture(autouse=True)
def google_settings(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-key")


@pytest.mark.asyncio
@respx.mock
async def test_one_vector_per_text_and_both_call_paths():
    def respond(request):
        body = json.loads(request.content)
        assert body["outputDimensionality"] == 1536
        assert len(body["content"]["parts"]) == 1
        value = {"first": 1.0, "second": 2.0}[body["content"]["parts"][0]["text"]]
        return httpx.Response(200, json={"embedding": {"values": [value] * 1536}})

    route = respx.post(URL).mock(side_effect=respond)
    vectors = await embedder.embed_texts(["first", "second"])
    ingestion_vectors, _ = await ingestion._embed_with_usage(["first", "second"])
    assert vectors == ingestion_vectors
    assert [v[0] for v in vectors] == [1.0, 2.0]
    assert route.call_count == 4


@pytest.mark.asyncio
@respx.mock
async def test_wrong_dimensions_rejected():
    respx.post(URL).respond(200, json={"embedding": {"values": [0.1] * 768}})
    with pytest.raises(ValueError, match="1536"):
        await gemini_embeddings.embed_with_usage(["text"])


@pytest.mark.asyncio
@respx.mock
async def test_quota_retry_is_bounded(monkeypatch):
    async def no_wait(_):
        pass

    monkeypatch.setattr(gemini_embeddings.asyncio, "sleep", no_wait)
    route = respx.post(URL).respond(429, json={"error": {"message": "quota"}})
    with pytest.raises(APIError, match="429"):
        await gemini_embeddings.embed_with_usage(["text", "must not be sent"])
    assert route.call_count == 3


@pytest.mark.asyncio
@respx.mock
async def test_auth_error_not_retried():
    route = respx.post(URL).respond(403)
    with pytest.raises(APIError, match="403"):
        await gemini_embeddings.embed_with_usage(["text"])
    assert route.call_count == 1


def test_gpt_oss_routes_to_groq(monkeypatch):
    from app.agent import llm

    monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key")
    model = llm.get_chat_model("openai/gpt-oss-20b")
    assert isinstance(model, llm.ChatGroq)
    assert model.model_name == "openai/gpt-oss-20b"
