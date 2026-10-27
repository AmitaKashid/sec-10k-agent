"""10-K section extraction. See docs/lld.md §LLD 2 for the full contract.

Two-tier strategy:
1. edgartools `TenK` parser fed via a minimal in-memory shim (no re-download).
2. Regex fallback over BeautifulSoup-cleaned text.

Returns plain text for Item 1A, 7, 8 only. No chunking, embedding, or DB writes.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from bs4 import BeautifulSoup
from pydantic import BaseModel

logger = logging.getLogger(__name__)

ItemKey = Literal["1A", "7", "8"]
TARGET_ITEMS: tuple[ItemKey, ...] = ("1A", "7", "8")


# --- Errors ---------------------------------------------------------------


class ParserError(Exception):
    """Base for parser errors."""


class ParserAllTiersFailed(ParserError):
    """Both tier-1 and tier-2 returned zero items."""


# --- Result type ----------------------------------------------------------


class ParsedFiling(BaseModel):
    items: dict[str, str]  # subset of {"1A","7","8"}; missing keys = section not found
    tier: Literal["edgartools", "regex"]
    char_counts: dict[str, int]
    warnings: list[str] = []


# --- Tier 1: edgartools shim ---------------------------------------------


class _BytesFiling:
    """Minimal duck-typed Filing that lets edgartools' TenK parser work on bytes
    we already have on disk, without it re-downloading from SEC.

    Only the attributes/methods that TenK + its HTMLParser touch are populated.
    """

    def __init__(self, html_bytes: bytes, accession_no: str):
        self._html = html_bytes.decode("utf-8", errors="replace")
        self.form = "10-K"
        self.accession_number = accession_no
        self.filing_date = ""
        self.base_dir = "."

    def html(self) -> str:
        return self._html


def _tier1_edgartools(html: bytes, accession_no: str) -> dict[str, str]:
    """Try edgartools. Returns {} if anything goes wrong; caller falls through to tier 2."""
    try:
        from edgar.company_reports import TenK

        tenk = TenK(_BytesFiling(html, accession_no))  # type: ignore[arg-type]
        out: dict[str, str] = {}
        for key in TARGET_ITEMS:
            text = tenk[f"Item {key}"]
            if text and text.strip():
                out[key] = text
        return out
    except Exception as e:
        logger.warning("tier-1 edgartools failed for %s: %s", accession_no, e)
        return {}


# --- Tier 2: regex fallback ----------------------------------------------

# Item header pattern: "Item 1A", "ITEM 1A.", "Item&nbsp;1A —", with optional
# punctuation/whitespace. Captures the item number+letter group.
_ITEM_HEADER = re.compile(
    r"item\s*[\xa0\s]*"  # "Item" + any whitespace incl. nbsp
    r"(?P<num>\d{1,2}[A-Z]?)"  # 1, 1A, 7, 7A, 8, etc.
    r"\s*[.\-:—–)]?",  # optional trailing punctuation
    re.IGNORECASE,
)

# After 1A comes 1B; after 7 comes 7A; after 8 comes 9. Slice between.
_NEXT_AFTER: dict[str, tuple[str, ...]] = {
    "1A": ("1B", "2"),
    "7": ("7A", "8"),
    "8": ("9", "9A"),
}


def _html_to_text(html: bytes) -> str:
    """Strip tags, drop scripts/styles, leave paragraph breaks intact."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    # Get text with newline separator so item headers stay on their own line.
    text = soup.get_text(separator="\n")
    return text


def _find_last_item_offset(text: str, item_num: str) -> int | None:
    """Return the start offset of the LAST occurrence of `Item <item_num>` header.
    The first hit is usually the Table of Contents; the last is the actual section.
    """
    pattern = re.compile(
        rf"item\s*[\xa0\s]*{re.escape(item_num)}(?![A-Z0-9])\s*[.\-:—–)]?",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return matches[-1].start()


def _tier2_regex(html: bytes) -> tuple[dict[str, str], list[str]]:
    text = _html_to_text(html)
    out: dict[str, str] = {}
    warnings: list[str] = []

    for item in TARGET_ITEMS:
        start = _find_last_item_offset(text, item)
        if start is None:
            continue

        end: int | None = None
        for next_item in _NEXT_AFTER[item]:
            cand = _find_last_item_offset(text[start + 1 :], next_item)
            if cand is not None:
                end = start + 1 + cand
                break

        body = text[start:end] if end else text[start:]
        if len(body) < 500:
            warnings.append(f"Item {item}: only {len(body)} chars — may be truncated")
            continue
        out[item] = body

    return out, warnings


# --- Cleanup --------------------------------------------------------------

_PAGE_NUM_LINE = re.compile(r"^\s*(?:page\s+)?\d{1,4}\s*$", re.IGNORECASE | re.MULTILINE)
_IXBRL_TAG = re.compile(r"<ix:[^>]*>|</ix:[^>]*>", re.IGNORECASE)
_MULTI_BLANK = re.compile(r"\n\s*\n\s*\n+")


def _clean(text: str) -> str:
    text = _IXBRL_TAG.sub("", text)
    text = _PAGE_NUM_LINE.sub("", text)
    # Collapse internal whitespace runs but keep paragraph breaks.
    lines = [re.sub(r"[ \t\xa0]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


# --- Public entry point ---------------------------------------------------


def parse_10k(html: bytes, accession_no: str) -> ParsedFiling:
    """Extract Item 1A / 7 / 8 from raw 10-K HTML bytes.

    Tries edgartools first; falls back to regex. Raises ParserAllTiersFailed
    if both tiers return zero items.
    """
    warnings: list[str] = []

    items = _tier1_edgartools(html, accession_no)
    tier: Literal["edgartools", "regex"] = "edgartools"

    if not items:
        items, regex_warnings = _tier2_regex(html)
        warnings.extend(regex_warnings)
        tier = "regex"

    if not items:
        raise ParserAllTiersFailed(f"Both tiers returned zero items for accession {accession_no}")

    cleaned = {k: _clean(v) for k, v in items.items()}
    char_counts = {k: len(v) for k, v in cleaned.items()}

    missing = [k for k in TARGET_ITEMS if k not in cleaned]
    if missing:
        warnings.append(f"items not found: {', '.join(missing)}")

    return ParsedFiling(
        items=cleaned,
        tier=tier,
        char_counts=char_counts,
        warnings=warnings,
    )


def extract_headings(text: str) -> list[tuple[int, str]]:
    """Return (char_offset, heading_text) pairs from a section's plain text.

    10-K sub-sections (e.g. "Competition", "Cybersecurity Risks") appear as
    short standalone lines between paragraph breaks.  We detect them by
    looking for text that:
      - sits between two blank lines  (\\n\\n ... \\n\\n)
      - is 4–70 characters long
      - starts with an uppercase letter
      - does not end with sentence-closing punctuation (those are short
        paragraphs, not headings)

    The caller uses these offsets to assign each chunk the last heading
    whose offset is ≤ the chunk's char_start.
    """
    headings: list[tuple[int, str]] = []
    for m in re.finditer(r"\n\n([^\n]{4,70})\n\n", text):
        candidate = m.group(1).strip()
        if not candidate:
            continue
        if not candidate[0].isupper():
            continue
        if candidate[-1] in ".,:;?!":
            continue
        headings.append((m.start(1), candidate))
    return headings


__all__ = [
    "ParsedFiling",
    "ParserError",
    "ParserAllTiersFailed",
    "extract_headings",
    "parse_10k",
]
