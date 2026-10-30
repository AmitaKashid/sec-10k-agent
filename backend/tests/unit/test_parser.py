"""Unit tests for app.data.parser (LLD 2 test plan).

Strategy:
- Synthetic HTML covers the regex tier and error path (deterministic, runs in CI).
- Fixture-based tests covering the edgartools tier are skipped when fixtures are
  not present (fixtures live under backend/tests/fixtures/10ks/, gitignored).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.data.parser import (
    ParsedFiling,
    ParserAllTiersFailed,
    parse_10k,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "10ks"


@pytest.fixture
def force_regex_tier(monkeypatch):
    """Disable tier-1 so the test exercises the regex fallback path."""
    from app.data import parser

    monkeypatch.setattr(parser, "_tier1_edgartools", lambda html, accession: {})


# --- Synthetic HTML for regex tier ---------------------------------------


def _synthetic_10k_html() -> bytes:
    """Hand-crafted minimal 10-K-shaped HTML. edgartools won't recognize it,
    forcing the regex fallback."""
    body_1a = "We face material risks. " * 40
    body_7 = "Management discusses operations. " * 40
    body_8 = "Notes to consolidated financial statements. " * 40
    return f"""
    <html><body>
      <p>Table of Contents</p>
      <p>Item 1A Risk Factors ......... 5</p>
      <p>Item 7 MD&amp;A ............... 30</p>
      <p>Item 8 Financial Statements .. 50</p>

      <h2>Item 1A. Risk Factors</h2>
      <p>{body_1a}</p>

      <h2>Item 1B. Unresolved Staff Comments</h2>
      <p>None.</p>

      <h2>Item 7. Management's Discussion and Analysis</h2>
      <p>{body_7}</p>

      <h2>Item 7A. Quantitative and Qualitative Disclosures</h2>
      <p>Interest-rate sensitivity table omitted.</p>

      <h2>Item 8. Financial Statements and Supplementary Data</h2>
      <p>{body_8}</p>

      <h2>Item 9. Changes In and Disagreements With Accountants</h2>
      <p>None.</p>
    </body></html>
    """.encode()


def test_regex_tier_extracts_three_items(force_regex_tier):
    result = parse_10k(_synthetic_10k_html(), "0000000000-99-000001")
    assert isinstance(result, ParsedFiling)
    assert result.tier == "regex"
    assert set(result.items.keys()) == {"1A", "7", "8"}
    # Every item must have body text, not just the header.
    for key, text in result.items.items():
        assert len(text) > 200, f"Item {key} suspiciously short: {len(text)}"
    # No HTML tags survived cleanup.
    assert all("<" not in t and ">" not in t for t in result.items.values())


def test_regex_tier_picks_last_header_not_toc(force_regex_tier):
    """The TOC entries appear before the real sections; regex should slice
    starting at the *last* occurrence of each header."""
    result = parse_10k(_synthetic_10k_html(), "0000000000-99-000002")
    # If we accidentally sliced from the TOC, Item 1A would be tiny (just the TOC line).
    assert result.char_counts["1A"] > 500


def test_all_tiers_failed_raises():
    """HTML with no recognizable item headers should raise."""
    junk = b"<html><body><p>Just some random prose with no item markers.</p></body></html>"
    with pytest.raises(ParserAllTiersFailed):
        parse_10k(junk, "0000000000-99-000003")


def test_partial_extraction_returns_partial_dict():
    """When only Item 1A is present, return {1A: ...} with a warning."""
    body_1a = "We face material risks. " * 40
    html = f"""
    <html><body>
      <h2>Item 1A. Risk Factors</h2>
      <p>{body_1a}</p>
      <h2>Item 1B. Unresolved Staff Comments</h2>
      <p>None.</p>
    </body></html>
    """.encode()

    result = parse_10k(html, "0000000000-99-000004")
    assert "1A" in result.items
    assert "7" not in result.items
    assert "8" not in result.items
    assert any("not found" in w for w in result.warnings)


def test_char_counts_match_item_lengths():
    result = parse_10k(_synthetic_10k_html(), "0000000000-99-000005")
    for key, text in result.items.items():
        assert result.char_counts[key] == len(text)


def test_parsed_filing_serializable(force_regex_tier):
    result = parse_10k(_synthetic_10k_html(), "0000000000-99-000006")
    dumped = result.model_dump_json()
    assert '"tier":"regex"' in dumped


# --- Fixture-based tests (edgartools tier) -------------------------------


def _list_fixtures() -> list[Path]:
    if not FIXTURES_DIR.exists():
        return []
    return sorted(FIXTURES_DIR.glob("*.htm")) + sorted(FIXTURES_DIR.glob("*.html"))


@pytest.mark.corpus
@pytest.mark.parametrize("fixture", _list_fixtures())
def test_edgartools_tier_on_fixture(fixture: Path):
    """For each downloaded fixture, edgartools should extract at least one
    target item and produce sane char counts."""
    html = fixture.read_bytes()
    accession = fixture.stem  # use filename as accession proxy
    result = parse_10k(html, accession)

    # Must extract at least one of the three target items.
    assert result.items, f"No items extracted from {fixture.name}"

    # Property assertions per LLD 2 test plan.
    for key, text in result.items.items():
        assert 5_000 <= len(text) <= 1_000_000, (
            f"{fixture.name} Item {key}: {len(text)} chars out of 5KB-1MB sanity range"
        )
        assert "<" not in text and ">" not in text, (
            f"{fixture.name} Item {key}: HTML markup survived cleanup"
        )


@pytest.mark.corpus
def test_fixtures_dir_documented_when_empty():
    """If no fixtures exist, surface a helpful skip message rather than silent pass."""
    if not _list_fixtures():
        pytest.skip(
            f"No 10-K fixtures at {FIXTURES_DIR}. "
            f"Run `uv run python scripts/fetch_fixtures.py` to download."
        )
