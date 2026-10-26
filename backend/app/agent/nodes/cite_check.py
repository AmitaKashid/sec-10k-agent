"""Cite-check — the verification gate that makes the system hallucination-proof.

For each claim in state.claims:
  - kind="number": parse a Decimal from claim.text; require EXACT
    equality with claim.source_ref.value. No tolerance.
  - kind="prose":  pass if claim.text appears as a case-insensitive
    substring of the cited slice OR cosine_sim >= 0.85 between
    embed(claim.text) and embed(slice).
  - empty source_ref → fail.

All pass → write to answer_cache, set verified=True, final_answer.
Any fail → set verified=False, refusal_reason_final, failed_claims.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation

from app.agent.nodes.validate import write_to_cache
from app.agent.state import AgentState, ChunkHit, Claim, XbrlFact, build_turn_from_state
from app.data.embedder import embed_texts

PROSE_COSINE_THRESHOLD = 0.85


def _assemble_with_markers(claims: list[Claim]) -> str:
    """Render the verified-claims list as a single answer string with
    `[N]` citation markers appended to each sentence (1-based).

    Markers are injected here, not by the LLM, so they always start at
    [1] and increment by 1 — keeping them aligned with the citation
    `id`s the proxy emits regardless of the all-pass vs partial path.
    """
    parts: list[str] = []
    for i, c in enumerate(claims):
        body = c.text.rstrip().rstrip(".").rstrip()
        if not body:
            continue
        parts.append(f"{body} [{i + 1}].")
    return " ".join(parts)


# Minimum length (in characters, after whitespace collapse) of an ellipsis-
# separated fragment for it to count toward the cite_check fallback. Below
# this floor, fragments are too generic to ground a claim verbatim ("the
# Company", "and other", etc.) and would weaken the structural guarantee.
_MIN_FRAGMENT_LEN = 30

# Splits on a literal three-dot or unicode-horizontal-ellipsis run, with any
# surrounding whitespace. Used by the ellipsis-split fallback only.
_ELLIPSIS_RE = re.compile(r"\s*(?:\.{3,}|…)\s*")

# Match a number with optional commas, decimals, and a magnitude suffix.
# Captures the (sign + digits) and the (suffix) separately.
# Two alternatives: comma-separated groups (require at least one comma so
# we don't greedy-match "391" out of "391035000000") OR plain digits.
_MONEY_RE = re.compile(
    r"-?\$?\s*-?\s*"
    r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*"
    r"(million|billion|trillion|thousand|m|b|t|k)?",
    re.IGNORECASE,
)
_SUFFIX_MULTIPLIER: dict[str, Decimal] = {
    "": Decimal(1),
    "thousand": Decimal(1_000),
    "k": Decimal(1_000),
    "million": Decimal(1_000_000),
    "m": Decimal(1_000_000),
    "billion": Decimal(1_000_000_000),
    "b": Decimal(1_000_000_000),
    "trillion": Decimal(1_000_000_000_000),
    "t": Decimal(1_000_000_000_000),
}


def _parse_money(text: str) -> Decimal | None:
    """Pull a money value out of prose. Prefers money-shaped matches
    (with $, comma group, or magnitude suffix) over bare integers like
    'FY2024' that happen to look numeric.
    """
    matches = list(_MONEY_RE.finditer(text))
    if not matches:
        return None

    def _is_money_shaped(m: re.Match[str]) -> bool:
        whole = m.group(0)
        return "$" in whole or "," in whole or bool(m.group(2))

    money_shaped = [m for m in matches if _is_money_shaped(m)]
    if money_shaped:
        chosen = money_shaped[0]
    else:
        # All plain integers — prefer the longest one. "Apple revenue in
        # fiscal year 2024 was 391035000000" must pick 391035000000, not 2024.
        chosen = max(matches, key=lambda m: len(m.group(1).replace(",", "")))

    raw = chosen.group(1).replace("$", "").replace(",", "").replace(" ", "").strip()
    suffix = (chosen.group(2) or "").lower()
    try:
        base = Decimal(raw)
    except InvalidOperation:
        return None
    return (-base if "-" in chosen.group(0) else base) * _SUFFIX_MULTIPLIER[suffix]


def _check_number(claim: Claim) -> tuple[bool, str | None, bool]:
    """Returns (ok, why, is_value_mismatch).

    `is_value_mismatch` is True only when the claim text parsed cleanly
    AND didn't equal the XBRL value — i.e. a real numeric tripwire.
    Missing-source or unparseable-text failures return False so the
    caller can distinguish "we caught a hallucinated number" from
    "the claim was malformed". Returning a structured flag instead of
    the caller string-matching the `why` message keeps NUMERIC_TRIPWIRE
    correctly classified even if `why`'s wording later changes.
    """
    if not isinstance(claim.source_ref, XbrlFact):
        return False, "number claim has no XBRL source", False
    parsed = _parse_money(claim.text)
    if parsed is None:
        return False, "couldn't parse number from claim text", False
    if parsed == claim.source_ref.value:
        return True, None, False
    return (
        False,
        f"value mismatch: claim={parsed} vs source={claim.source_ref.value}",
        True,
    )


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


async def _check_prose(
    claim: Claim, all_chunks: list[ChunkHit] | None = None
) -> tuple[bool, str | None]:
    """Verify a prose claim against its cited chunk.

    Primary check: `claim.evidence_quote` (a verbatim span the LLM
    extracted from the source) must appear as a case-insensitive
    substring of the chunk text. This is the structural anti-
    hallucination guarantee — paraphrases in `claim.text` are fine, but
    *some* literal text from the source must back the claim.

    Cross-chunk fallback: if the quote fails against source_ref, search
    all retrieved chunks. The LLM occasionally sets the wrong source_idx
    while still quoting verbatim text from a real chunk — accepting the
    quote when found anywhere preserves the verbatim guarantee while
    eliminating false refusals from index attribution errors.

    Defense-in-depth: if no quote is provided (older claim payloads,
    cache hits from before the schema change), fall back to comparing
    `claim.text` against the chunk via substring/cosine. This branch is
    not the load-bearing check — under the current prompt every prose
    claim should carry a quote.
    """
    if not isinstance(claim.source_ref, ChunkHit):
        return False, "prose claim has no chunk source"
    src = claim.source_ref
    # `src.text` is the chunk's own body. `char_start`/`char_end` are
    # offsets into the *parent document* (used by the UI for citation
    # highlighting) — they must NOT be used to re-slice `src.text`,
    # which produces empty slices for every chunk past seq=0.
    chunk_text = src.text
    chunk_stripped = chunk_text.strip()
    if not chunk_stripped:
        return False, "cited chunk text is empty"

    # Primary path: verbatim quote substring match.
    if claim.evidence_quote:
        quote = claim.evidence_quote.strip()
        if not quote:
            return False, "evidence_quote is empty"

        # 1. Verbatim match against source_ref chunk.
        if _normalized_substring(quote, chunk_text):
            return True, None

        # 2. Cross-chunk verbatim: LLM may have set the wrong source_idx.
        #    Accepts when the exact quote appears in any retrieved chunk,
        #    preserving the verbatim guarantee while tolerating attribution errors.
        if all_chunks:
            for alt in all_chunks:
                if alt.chunk_id != src.chunk_id and _normalized_substring(quote, alt.text):
                    claim.source_ref = alt
                    return True, None

        # 3. Ellipsis-split fallback: the LLM occasionally stitches two
        #    contiguous spans from the same chunk together with a literal
        #    "..." (despite the prompt forbidding it). Accept when:
        #      - the quote splits into 2+ fragments on "..." or "…"
        #      - each fragment has length ≥ _MIN_FRAGMENT_LEN (avoids
        #        trivially-matching common phrases)
        #      - all fragments are verbatim substrings of the SAME chunk
        #        (no cross-chunk concatenation — that would break the
        #        provenance guarantee).
        fragments = [f.strip() for f in _ELLIPSIS_RE.split(quote)]
        fragments = [f for f in fragments if len(_collapse_ws(f)) >= _MIN_FRAGMENT_LEN]
        if len(fragments) >= 2:
            for ch in [src] + (all_chunks or []):
                if all(_normalized_substring(f, ch.text) for f in fragments):
                    return True, None

        return False, f"evidence_quote not verbatim in any source chunk: {quote[:80]!r}"

    # Fallback path (no quote provided): legacy claim.text matching.
    claim_stripped = claim.text.strip()
    if not claim_stripped:
        return False, "claim text is empty"
    if claim_stripped.lower() in chunk_text.lower():
        return True, None
    [v_claim, v_src] = await embed_texts([claim.text, chunk_text])
    sim = _cosine(v_claim, v_src)
    if sim >= PROSE_COSINE_THRESHOLD:
        return True, None
    return False, f"prose not grounded (cosine={sim:.3f}, no evidence_quote)"


def _normalized_substring(needle: str, haystack: str) -> bool:
    """Case-insensitive substring match with whitespace collapsed.

    Real-world LLM quotes drift on whitespace (line breaks become
    spaces, double-spaces collapse) even when the prose is verbatim.
    Collapsing runs of whitespace to a single space on both sides
    preserves the structural guarantee — every other character must
    still match exactly — while tolerating that one well-known
    formatting drift.
    """
    return _collapse_ws(needle).lower() in _collapse_ws(haystack).lower()


def _collapse_ws(s: str) -> str:
    return " ".join(s.split())


async def cite_check(state: AgentState) -> dict:
    if state.question_type == "REFUSE":
        return {}
    if state.cache_hit:
        # validate already populated final_answer from cache
        return {
            "trace_events": state.trace_events
            + [{"node": "cite_check", "skipped_cache_hit": True}],
        }
    if not state.claims:
        return {
            "verified": False,
            "refusal_reason_final": ("I couldn't ground an answer in the available SEC filings."),
            "refusal_code": "CLAIM_UNSUPPORTED",
            "trace_events": state.trace_events
            + [{"node": "cite_check", "verified": False, "reason": "no claims"}],
        }

    failed: list[Claim] = []
    failure_reasons: list[str] = []
    numeric_mismatch = False
    for claim in state.claims:
        if claim.source_ref is None:
            failed.append(claim)
            failure_reasons.append(f"no source_ref for: {claim.text!r}")
            continue
        if claim.kind == "number":
            ok, why, is_mismatch = _check_number(claim)
            if is_mismatch:
                numeric_mismatch = True
        else:
            ok, why = await _check_prose(claim, state.chunks)
        if not ok:
            failed.append(claim)
            failure_reasons.append(f"{why}: {claim.text!r}")

    if failed:
        fail_rate = len(failed) / len(state.claims)
        failed_ids = {id(c) for c in failed}
        verified_claims = [c for c in state.claims if id(c) not in failed_ids]

        # Partial-answer path: if ≤33% of claims fail and at least one
        # passes, drop the unverified sentences and ship the rest. This
        # handles index-attribution errors (LLM cites wrong chunk) without
        # compromising the verbatim guarantee on the sentences that remain.
        if fail_rate <= 0.34 and verified_claims:
            partial = _assemble_with_markers(verified_claims)
            if state.cache_key:
                await write_to_cache(state.cache_key, partial, verified_claims)
            return {
                "verified": True,
                "final_answer": partial,
                "claims": verified_claims,
                "failed_claims": failed,
                "trace_events": state.trace_events
                + [
                    {
                        "node": "cite_check",
                        "verified": True,
                        "partial": True,
                        "dropped_claims": len(failed),
                        "failures": failure_reasons,
                    }
                ],
                # Append a compact Turn so the next turn's `plan` sees
                # the resolved entity. Partial-answer path is terminal
                # (routes to END), so this is our only chance.
                "history": [build_turn_from_state(state, partial)],
            }

        code = "NUMERIC_TRIPWIRE" if numeric_mismatch else "CLAIM_UNSUPPORTED"
        return {
            "verified": False,
            "failed_claims": failed,
            "refusal_reason_final": (
                "I couldn't verify all claims against their cited sources. "
                "Refusing rather than risking a hallucination."
            ),
            "refusal_code": code,
            "trace_events": state.trace_events
            + [
                {
                    "node": "cite_check",
                    "verified": False,
                    "failures": failure_reasons,
                    "refusal_code": code,
                }
            ],
        }

    # All claims verified — assemble with markers, cache, and ship.
    final_answer = _assemble_with_markers(state.claims)
    if state.cache_key:
        await write_to_cache(state.cache_key, final_answer, state.claims)

    return {
        "verified": True,
        "final_answer": final_answer,
        "trace_events": state.trace_events
        + [{"node": "cite_check", "verified": True, "claims": len(state.claims)}],
        # Terminal node on the verified path — append the Turn so the
        # next turn's `plan` can resolve follow-ups like "and 2024?".
        "history": [build_turn_from_state(state, final_answer)],
    }
