"""OpenAI embeddings for chunks. See docs/lld.md §LLD 3.

Batched, retried, async. No caching — DB unique constraint already makes
re-ingest a no-op.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.settings import settings

if TYPE_CHECKING:
    from app.data.chunker import Chunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 256
EMBEDDING_DIM = 1536

_RETRYABLE = (RateLimitError, APIConnectionError, APIError)


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


async def embed_texts(
    texts: list[str],
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    if not texts:
        return []
    if settings.EMBEDDING_PROVIDER == "gemini":
        from app.data.gemini_embeddings import embed_with_usage

        vectors, _ = await embed_with_usage(texts)
        return vectors
    if settings.EMBEDDING_PROVIDER != "openai":
        raise ValueError("EMBEDDING_PROVIDER must be openai or gemini")
    client = _client()

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _embed_batch(batch: list[str]) -> list[list[float]]:
        resp = await client.embeddings.create(model=model, input=batch)
        # OpenAI guarantees response order matches input order.
        return [d.embedding for d in resp.data]

    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        out.extend(await _embed_batch(batch))
    return out


async def embed_chunks(
    chunks: list[Chunk],
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[Chunk]:
    if not chunks:
        return chunks
    vectors = await embed_texts([c.text for c in chunks], model=model, batch_size=batch_size)
    for chunk, vec in zip(chunks, vectors, strict=True):
        chunk.embedding = vec
    return chunks
