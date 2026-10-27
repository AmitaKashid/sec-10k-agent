"""Section text → token-bounded chunks with char-offset citation anchors.

See docs/lld.md §LLD 3. Pure function; no I/O.

Boundary preference: paragraph (\\n\\n) → sentence (`[.!?] + \\s + [A-Z]`) →
word (whitespace) → never mid-word. ``char_start``/``char_end`` are offsets
into the input text such that ``text[char_start:char_end] == chunk.text``.
"""

from __future__ import annotations

import re
from typing import Literal

import tiktoken
from pydantic import BaseModel

ItemKey = Literal["1A", "7", "8"]

_ENCODING = "cl100k_base"  # text-embedding-3-small uses cl100k_base
_SENTENCE_RE = re.compile(r"[.!?](?=\s+[A-Z\"'(\[])")


class Chunk(BaseModel):
    item: ItemKey
    seq: int
    text: str
    token_count: int
    char_start: int
    char_end: int
    embedding: list[float] | None = None
    section_title: str | None = None


def chunk_section(
    text: str,
    item: ItemKey,
    target_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    if not text or not text.strip():
        return []
    if overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be < target_tokens")

    enc = tiktoken.get_encoding(_ENCODING)
    ids = enc.encode(text)
    if len(ids) <= target_tokens:
        return [
            Chunk(
                item=item,
                seq=0,
                text=text,
                token_count=len(ids),
                char_start=0,
                char_end=len(text),
            )
        ]

    # Map each token boundary to a char offset in `text`.
    # Tokens can split a UTF-8 character; accumulate bytes, then translate
    # byte offsets to char offsets.
    text_bytes = text.encode("utf-8")
    byte_offsets = [0]
    for tok in ids:
        byte_offsets.append(byte_offsets[-1] + len(enc.decode_single_token_bytes(tok)))
    assert byte_offsets[-1] == len(text_bytes), "tokenizer byte accounting drift"

    char_at_byte = [0] * (len(text_bytes) + 1)
    b = 0
    for ci, ch in enumerate(text):
        ch_len = len(ch.encode("utf-8"))
        for k in range(ch_len):
            char_at_byte[b + k] = ci
        b += ch_len
    char_at_byte[len(text_bytes)] = len(text)

    def token_to_char(tok_idx: int) -> int:
        return char_at_byte[byte_offsets[tok_idx]]

    chunks: list[Chunk] = []
    n = len(ids)
    step = target_tokens - overlap_tokens
    i = 0
    seq = 0

    while i < n:
        j = min(i + target_tokens, n)
        raw_start = token_to_char(i)
        raw_end = token_to_char(j)

        cs = _snap_start(text, raw_start)
        ce = len(text) if j >= n else _snap_end(text, raw_end, cs)
        if ce <= cs:
            # Pathological snap collapse — keep raw window so we make progress.
            cs, ce = raw_start, raw_end

        chunk_text = text[cs:ce]
        token_count = len(enc.encode(chunk_text))
        chunks.append(
            Chunk(
                item=item,
                seq=seq,
                text=chunk_text,
                token_count=token_count,
                char_start=cs,
                char_end=ce,
            )
        )
        seq += 1
        if j >= n:
            break
        i += step

    return chunks


def _snap_start(text: str, cs: int) -> int:
    """Move ``cs`` forward to the start of a word, skipping leading whitespace.

    If we land mid-word, advance to the next word start so chunks never begin
    with a partial token.
    """
    if cs <= 0:
        return 0
    n = len(text)
    while cs < n and text[cs].isspace():
        cs += 1
    # If previous char and current char are both non-whitespace we are mid-word.
    if 0 < cs < n and not text[cs - 1].isspace() and not text[cs].isspace():
        while cs < n and not text[cs].isspace():
            cs += 1
        while cs < n and text[cs].isspace():
            cs += 1
    return cs


def _snap_end(text: str, ce: int, cs: int, search: int = 300) -> int:
    """Pull ``ce`` backward to the nearest preferred boundary in ``[cs+1, ce]``.

    Preference order: paragraph break, sentence end, word boundary. Falls
    through to ``ce`` (mid-word) only if no whitespace exists in the window.
    """
    n = len(text)
    if ce >= n:
        return n
    lo = max(cs + 1, ce - search)
    region = text[lo:ce]
    # paragraph break
    p = region.rfind("\n\n")
    if p != -1:
        return lo + p
    # sentence terminator followed by whitespace + capital
    last_sentence = -1
    for m in _SENTENCE_RE.finditer(region):
        last_sentence = m.end()
    if last_sentence != -1:
        return lo + last_sentence
    # last whitespace → end-of-word boundary
    w = region.rfind(" ")
    if w == -1:
        w = max(region.rfind("\n"), region.rfind("\t"))
    if w != -1:
        return lo + w
    return ce
