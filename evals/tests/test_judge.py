"""Tests for evals.judge — exercise cache + verdict shape without
calling the live LLM. Live judge calls are covered by the integration
run, not unit tests."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.judge import (  # noqa: E402
    JudgeVerdict,
    _cache_key,
    judge_hallucination,
)


def test_cache_key_is_deterministic_and_input_sensitive():
    s = [{"value": 100, "concept_key": "revenue", "accession_no": "X"}]
    a = _cache_key("m", "q", "a", s)
    b = _cache_key("m", "q", "a", s)
    assert a == b
    # any change in input → different key
    assert a != _cache_key("m2", "q", "a", s)
    assert a != _cache_key("m", "q2", "a", s)
    assert a != _cache_key("m", "q", "a2", s)
    assert a != _cache_key("m", "q", "a", s + [{"value": 200}])


def test_cache_hit_short_circuits_llm(tmp_path):
    cache_path = tmp_path / "cache.json"
    sources = [{"value": "100", "concept_key": "revenue", "accession_no": "X"}]
    key = _cache_key("test-model", "Q?", "answer", sources)
    cache_path.write_text(
        json.dumps(
            {
                key: {
                    "claims": [
                        {"claim": "supported fact", "supported": True},
                        {"claim": "fabricated", "supported": False},
                    ],
                    "reasoning": "cached",
                }
            }
        )
    )

    verdict = asyncio.run(
        judge_hallucination(
            question="Q?",
            answer="answer",
            sources=sources,
            model="test-model",
            cache_path=cache_path,
        )
    )
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.supported is False  # rolled up from per-claim verdicts
    assert verdict.unsupported_claims == ["fabricated"]
    assert verdict.total_claims == 2
    assert verdict.unsupported_count == 1
    assert verdict.reasoning == "cached"


def test_render_source_includes_evidence_after_1200_characters():
    from evals.judge import _render_source

    late_evidence = "The company relies on a small number of suppliers."
    source = {
        "item": "1A",
        "accession_no": "example",
        "text": "A" * 1300 + late_evidence,
    }

    rendered = _render_source(source)

    assert late_evidence in rendered
