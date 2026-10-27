"""LLM-as-judge for semantic hallucination detection.

The structural scorer in `metrics.py` catches numeric mismatches. Prose
hallucination is harder: an answer can cite the right filing yet still
state a claim the source doesn't support. We hand (question, answer,
cited sources) to a frontier judge and ask whether every factual claim
in the answer is grounded.

Design choices:
  - The judge model is configurable via JUDGE_MODEL env var. Default is
    a cheap-ish frontier model since judge quality > judge cost (≤40 rows).
  - Verdicts are cached on disk keyed by hash(model + question + answer
    + sources). The cache file is a flat JSON dict so it's diffable and
    needs no new dependency. Cache survives across runs so re-running
    the harness while iterating on the agent is cheap.
  - Pure scoring stays in `metrics.py`. This module is opt-in and is
    invoked by `run_evals.py` after the structural pass.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# Default to gpt-4o: frontier-quality entailment, uses OPENAI_API_KEY
# which the rest of the stack already requires for embeddings (no extra
# key to provision). Cost is trivial at ≤40 rows with the on-disk cache.
# Override with JUDGE_MODEL env var if needed.
DEFAULT_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.4")
# Schema bumped to v2: per-claim verdicts. Old v1 cache file is left behind
# rather than migrated (≤40 rows; re-judging is cheap).
DEFAULT_CACHE_PATH = Path(__file__).parent / "results" / ".judge_cache.v2.json"


# --- Verdict schema ------------------------------------------------------


class ClaimVerdict(BaseModel):
    claim: str = Field(description="One distinct factual claim extracted from the answer.")
    supported: bool = Field(
        description="True iff this specific claim is directly supported by the cited sources."
    )


class JudgeVerdict(BaseModel):
    # Reject unknown top-level keys so a regression to the v1 shape
    # ({"supported": bool, "unsupported_claims": [...]}) raises a
    # ValidationError instead of parsing to an empty-claims, vacuously-
    # supported verdict — which would silently mask CI-gate failures.
    model_config = ConfigDict(extra="forbid")

    claims: list[ClaimVerdict] = Field(
        default_factory=list,
        description=(
            "One verdict per distinct factual claim in the answer. Empty "
            "iff the answer makes no factual claims (rare; treat as supported)."
        ),
    )
    reasoning: str = Field(
        default="",
        description="One or two sentences explaining the overall verdict.",
    )

    @property
    def supported(self) -> bool:
        """Row-level rollup: every claim must be supported."""
        return all(c.supported for c in self.claims) if self.claims else True

    @property
    def unsupported_claims(self) -> list[str]:
        return [c.claim for c in self.claims if not c.supported]

    @property
    def total_claims(self) -> int:
        return len(self.claims)

    @property
    def unsupported_count(self) -> int:
        return sum(1 for c in self.claims if not c.supported)


# --- Cache ---------------------------------------------------------------


def _cache_key(model: str, question: str, answer: str, sources: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        {"m": model, "q": question, "a": answer, "s": sources},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("judge cache at %s is corrupt; ignoring", path)
        return {}


def _save_cache(path: Path, data: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))


# --- Source rendering ----------------------------------------------------


def _render_source(s: dict[str, Any]) -> str:
    """Compact one-source rendering for the judge prompt.

    XBRL facts and chunk hits get distinguished forms so the judge knows
    which part of the answer should match which source kind.
    """
    if "value" in s and "concept_key" in s:
        return (
            f"[XBRL fact] concept={s.get('concept_key')} "
            f"fy={s.get('fiscal_year')} value={s.get('value')} "
            f"unit={s.get('unit')} accn={s.get('accession_no')}"
        )
    if "text" in s:
        # `char_start`/`char_end` are document-level offsets used by the
        # UI for citation highlighting — they are NOT offsets into
        # `text`. Slicing `text[char_start:char_end]` yields empty for
        # every chunk past seq=0. Render the chunk body directly,
        # truncated to keep the judge prompt bounded.
        slice_text = (s["text"] or "")[:1200]
        return (
            f"[Chunk] item={s.get('item')} accn={s.get('accession_no')}\n"
            f"---\n{slice_text}\n---"
        )
    return f"[Source] {json.dumps(s, default=str)[:1200]}"


# --- Prompt --------------------------------------------------------------


_SYSTEM = (
    "You are a strict fact-grounding auditor for an SEC 10-K research agent. "
    "Given a question, the agent's answer, and the exact sources the agent "
    "cited (XBRL facts and 10-K text chunks), enumerate every distinct "
    "factual claim in the answer and return a per-claim verdict.\n\n"
    "For each claim, set supported=True iff that claim is directly supported "
    "by the cited sources. Be conservative: if a claim is plausible but not "
    "literally in the sources, mark it unsupported. Ignore stylistic phrasing; "
    "focus on factual content. Numerical values must match the cited XBRL fact.\n\n"
    "A 'distinct factual claim' is one assertion about the company, filing, "
    "or its disclosures. Split compound sentences into separate claims. "
    "Ignore connective phrasing, hedges, and citations themselves."
)


def _build_user_prompt(question: str, answer: str, sources: list[dict[str, Any]]) -> str:
    if not sources:
        rendered = "(no sources provided)"
    else:
        rendered = "\n\n".join(_render_source(s) for s in sources)
    return (
        f"Question:\n{question}\n\n"
        f"Agent answer:\n{answer}\n\n"
        f"Cited sources:\n{rendered}\n\n"
        "Return your verdict."
    )


# --- Public API ----------------------------------------------------------


async def judge_hallucination(
    *,
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    model: str = DEFAULT_JUDGE_MODEL,
    cache_path: Path = DEFAULT_CACHE_PATH,
    use_cache: bool = True,
) -> JudgeVerdict:
    """Ask the judge model whether `answer` is grounded in `sources`.

    Caches verdicts on disk so re-running the harness while iterating on
    the agent does not re-bill the judge for unchanged rows.
    """
    key = _cache_key(model, question, answer, sources)
    cache = _load_cache(cache_path) if use_cache else {}

    if use_cache and key in cache:
        return JudgeVerdict.model_validate(cache[key])

    # Imported lazily so test code paths that don't call the judge can
    # avoid loading the LangChain stack and the LLM SDKs.
    from app.agent.llm import get_chat_model

    chat = get_chat_model(model)
    structured = chat.with_structured_output(JudgeVerdict)
    msg = await structured.ainvoke(
        [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _build_user_prompt(question, answer, sources)},
        ]
    )
    verdict = msg if isinstance(msg, JudgeVerdict) else JudgeVerdict.model_validate(msg)

    if use_cache:
        cache[key] = verdict.model_dump()
        _save_cache(cache_path, cache)

    return verdict


async def judge_many(
    rows: list[dict[str, Any]],
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    cache_path: Path = DEFAULT_CACHE_PATH,
    concurrency: int = 4,
) -> list[JudgeVerdict]:
    """Score many rows with bounded concurrency."""
    sem = asyncio.Semaphore(concurrency)

    async def _one(row: dict[str, Any]) -> JudgeVerdict:
        async with sem:
            return await judge_hallucination(
                question=row["question"],
                answer=row["answer"],
                sources=row.get("sources") or [],
                model=model,
                cache_path=cache_path,
            )

    return await asyncio.gather(*(_one(r) for r in rows))


__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "JudgeVerdict",
    "judge_hallucination",
    "judge_many",
]
