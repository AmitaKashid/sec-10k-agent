"use client";

import { useEffect, useState } from "react";
import type { PipelineStep } from "@/lib/types";

type Props = {
  steps: PipelineStep[];
  isStreaming: boolean;
  elapsed: number;
};

function IconChevronDown({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M6 9l6 6 6-6" />
    </svg>
  );
}
function IconSparkles({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 3l1.9 4.6L18.5 9.5l-4.6 1.9L12 16l-1.9-4.6L5.5 9.5l4.6-1.9L12 3z" />
      <path d="M19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9L19 15z" />
    </svg>
  );
}
function IconCheckCircle({ size = 13 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="10" />
      <path d="M9 12l2 2 4-4" />
    </svg>
  );
}
function IconAlert({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
  );
}

export function Thinking({ steps, isStreaming, elapsed }: Props) {
  const errored = steps.some((s) => s.status === "error");
  const hasSteps = steps.length > 0;
  const allDone = hasSteps && steps.every((s) => s.status === "done");

  const [open, setOpen] = useState(true);
  const [userToggled, setUserToggled] = useState(false);
  useEffect(() => {
    if (userToggled) return;
    if (allDone || errored) setOpen(false);
    else if (isStreaming) setOpen(true);
  }, [allDone, errored, isStreaming, userToggled]);

  const summary = errored
    ? "Error during reasoning"
    : isStreaming
      ? "Thinking…"
      : `Thought for ${elapsed.toFixed(1)}s`;

  return (
    <div className="thinking">
      <button
        type="button"
        onClick={() => {
          if (!hasSteps) return;
          setUserToggled(true);
          setOpen((o) => !o);
        }}
        className={`thinking-summary ${hasSteps ? "is-clickable" : ""}`}
        aria-expanded={hasSteps ? open : undefined}
      >
        {isStreaming && !errored ? (
          <span className="spinner" aria-hidden="true" />
        ) : errored ? (
          <span className="thinking-icon is-error"><IconAlert /></span>
        ) : (
          <span className="thinking-icon is-accent"><IconSparkles /></span>
        )}
        <span className="thinking-label">{summary}</span>
        {hasSteps && (
          <span className={`thinking-chev ${open ? "is-open" : ""}`}>
            <IconChevronDown />
          </span>
        )}
      </button>

      {open && hasSteps && (
        <ul className="thinking-steps">
          {steps.map((s) => (
            <li key={s.id} className="thinking-step">
              {s.status === "done" && (
                <span className="thinking-step-icon is-done"><IconCheckCircle /></span>
              )}
              {s.status === "active" && <span className="spinner spinner-sm" aria-hidden="true" />}
              {s.status === "pending" && <span className="thinking-step-pending" aria-hidden="true" />}
              {s.status === "error" && (
                <span className="thinking-step-icon is-error"><IconAlert size={13} /></span>
              )}
              <span className={`thinking-step-label is-${s.status}`}>{s.label}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
