"""Google-hosted embeddings compatible with the existing vector(1536) column.

One request per text is intentional: Embedding 2 aggregates multiple parts
into one vector. The same symmetric input format is used for ingestion,
query retrieval, and citation similarity checks.
"""

from __future__ import annotations

import asyncio
import math

import httpx
from openai import APIConnectionError, APIError

from app.settings import settings

DIMENSIONS = 1536


async def embed_with_usage(texts: list[str]) -> tuple[list[list[float]], int]:
    if not texts:
        return [], 0
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is required for Google embeddings")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.GEMINI_EMBEDDING_MODEL}:embedContent"
    )
    vectors: list[list[float]] = []
    tokens = 0
    async with httpx.AsyncClient(timeout=60) as client:
        for text in texts:
            payload = {
                "content": {"parts": [{"text": text}]},
                "outputDimensionality": DIMENSIONS,
            }
            for attempt in range(3):
                try:
                    response = await client.post(
                        url, headers={"x-goog-api-key": settings.GEMINI_API_KEY}, json=payload
                    )
                except httpx.TransportError as exc:
                    if attempt == 2:
                        raise APIConnectionError(request=exc.request) from exc
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < 2:
                        try:
                            delay = float(response.headers.get("retry-after", "60"))
                        except ValueError:
                            delay = 60
                        await asyncio.sleep(max(1, min(delay, 60)))
                        continue
                if response.is_error:
                    # APIError integrates with ingestion's existing systemic-error
                    # handling, so exhausted quotas stop the run rather than
                    # retrying every filing. Never include request headers.
                    raise APIError(
                        f"Google embedding request failed (HTTP {response.status_code}). "
                        "Check Google AI Studio model access and quota.",
                        request=response.request,
                        body=None,
                    )
                break

            data = response.json()
            vector = data.get("embedding", {}).get("values", [])
            if len(vector) != DIMENSIONS or not all(math.isfinite(v) for v in vector):
                raise ValueError("Google returned an invalid or non-1536-dimensional vector")
            vectors.append(vector)
            tokens += data.get("usageMetadata", {}).get("promptTokenCount", 0)
    return vectors, tokens
