"use client";

import { Fragment } from "react";
import type { Citation } from "@/lib/types";

type Props = {
  text: string;
  activeCiteId: number | null;
  citations: Citation[];
  onCiteClick: (id: number | null) => void;
};

// Match `[N]` where N is one or more digits. The text the agent emits puts a
// space before the marker (e.g. "...risks [1]."), but we don't enforce that
// here — any standalone [N] in the answer turns into a chip.
const CITE_RE = /\[(\d+)\]/g;

export function AnswerText({ text, activeCiteId, citations, onCiteClick }: Props) {
  const knownIds = new Set(citations.map((c) => c.id));

  // Split into paragraphs on blank lines so multi-paragraph answers render
  // with proper spacing instead of running together.
  const paragraphs = text.split(/\n{2,}/);

  return (
    <div className="answer-text">
      {paragraphs.map((para, pIdx) => (
        <p key={pIdx}>{renderInline(para, knownIds, activeCiteId, onCiteClick)}</p>
      ))}
    </div>
  );
}

function renderInline(
  text: string,
  knownIds: Set<number>,
  activeCiteId: number | null,
  onCiteClick: (id: number | null) => void,
) {
  const out: React.ReactNode[] = [];
  let last = 0;
  let key = 0;
  for (const m of text.matchAll(CITE_RE)) {
    const idx = m.index ?? 0;
    if (idx > last) out.push(<Fragment key={key++}>{text.slice(last, idx)}</Fragment>);
    const id = Number(m[1]);
    if (knownIds.has(id)) {
      const isActive = activeCiteId === id;
      out.push(
        <button
          key={key++}
          type="button"
          onClick={() => onCiteClick(isActive ? null : id)}
          className={`cite-chip ${isActive ? "is-active" : ""}`}
          aria-pressed={isActive}
          aria-label={`Citation ${id}`}
        >
          {id}
        </button>,
      );
    } else {
      // Unknown id — keep the literal `[N]` so the answer stays faithful.
      out.push(<Fragment key={key++}>{m[0]}</Fragment>);
    }
    last = idx + m[0].length;
  }
  if (last < text.length) out.push(<Fragment key={key++}>{text.slice(last)}</Fragment>);
  return out;
}
