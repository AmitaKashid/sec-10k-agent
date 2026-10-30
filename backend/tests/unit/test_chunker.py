"""Unit tests for app.data.chunker (LLD 3 test plan)."""

from __future__ import annotations

import tiktoken

from app.data.chunker import Chunk, chunk_section

ENC = tiktoken.get_encoding("cl100k_base")


def _tokens(s: str) -> int:
    return len(ENC.encode(s))


def test_empty_input():
    assert chunk_section("", "1A") == []
    assert chunk_section("   \n  \t ", "1A") == []


def test_short_text_single_chunk():
    text = "We face material risks. " * 10  # well under 500 tokens
    out = chunk_section(text, "1A")
    assert len(out) == 1
    c = out[0]
    assert isinstance(c, Chunk)
    assert c.item == "1A"
    assert c.seq == 0
    assert c.char_start == 0
    assert c.char_end == len(text)
    assert c.text == text
    assert c.token_count == _tokens(text)


def test_long_text_chunked_with_overlap():
    sentence = "This sentence carries some real but generic content for chunking. "
    text = sentence * 400  # ~3200 tokens
    out = chunk_section(text, "7", target_tokens=500, overlap_tokens=50)
    assert len(out) >= 4
    for c in out:
        # never exceed target; allow generous lower bound (snapping can shrink chunks)
        assert c.token_count <= 500
        assert c.token_count >= 200
    # consecutive chunks should overlap textually somewhere
    for a, b in zip(out, out[1:], strict=False):
        assert b.char_start < a.char_end, "expected overlap between consecutive chunks"


def test_char_offsets_reproduce_chunk_text():
    text = ("Risk factor sentence number {} ends here. " * 200).format(*range(200))[:8000]
    # build a real long string deterministically
    text = " ".join(f"Sentence {i} describing risk number {i}." for i in range(600))
    out = chunk_section(text, "1A")
    assert len(out) >= 2
    for c in out:
        assert text[c.char_start : c.char_end] == c.text


def test_no_chunk_cuts_mid_word():
    text = " ".join(f"alphabet{i}" for i in range(2500))
    out = chunk_section(text, "7")
    assert len(out) >= 2
    for c in out:
        # leading/trailing chars should be word-clean
        assert not c.text.startswith(" ")
        assert c.text[0].isalnum() or c.text[0] in "(\"'"
        # last char should not be a partial word — either punctuation or alnum
        # immediately followed by whitespace/EOF in the source
        end_in_source = c.char_end
        assert (
            end_in_source == len(text)
            or text[end_in_source - 1].isspace()
            or not text[end_in_source - 1].isalnum()
            or text[end_in_source].isspace()
            if end_in_source < len(text)
            else True
        )


def test_seq_is_zero_indexed_and_contiguous():
    text = " ".join(f"word{i}" for i in range(3000))
    out = chunk_section(text, "8")
    assert [c.seq for c in out] == list(range(len(out)))


def test_chunks_cover_full_input_modulo_overlap():
    text = " ".join(f"token{i}" for i in range(3000))
    out = chunk_section(text, "1A")
    assert out[0].char_start == 0
    assert out[-1].char_end == len(text)
    # neighbors should be contiguous-or-overlapping (no gaps)
    for a, b in zip(out, out[1:], strict=False):
        assert b.char_start <= a.char_end


def test_paragraph_boundary_preferred():
    # Build text where a \n\n falls comfortably before the 500-token mark.
    big = "Filler sentence with content. " * 80  # ~480 tokens
    tail = "More filler here. " * 200
    text = big + "\n\n" + tail
    out = chunk_section(text, "7", target_tokens=500, overlap_tokens=50)
    # First chunk should end at-or-before the paragraph break.
    pb = text.index("\n\n")
    assert out[0].char_end <= pb + 2
