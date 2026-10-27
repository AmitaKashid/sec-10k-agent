"use client";

import type { Citation } from "@/lib/types";

type Props = {
  cite: Citation;
  onClose?: () => void;
};

function IconArrowUpRight({ size = 12 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M7 17L17 7" />
      <path d="M7 7h10v10" />
    </svg>
  );
}
function IconX({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M18 6L6 18M6 6l12 12" />
    </svg>
  );
}

export function SourceCard({ cite, onClose }: Props) {
  const isXbrl = cite.kind === "xbrl";
  const xbrlNamespace = isXbrl ? deriveNamespace(cite.concept_key) : null;
  const xbrlConceptName = isXbrl ? deriveConceptName(cite.concept_key) : null;

  return (
    <div className="source-card">
      <div className="source-card-head">
        <div className="source-card-head-l">
          <span className="source-id">{cite.id}</span>
          <span className="source-kind">
            {isXbrl ? "XBRL fact" : cite.section || "Source"}
          </span>
        </div>
        <div className="source-card-head-r">
          {isXbrl && xbrlNamespace && (
            <span className="source-namespace">{xbrlNamespace}</span>
          )}
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              className="source-close"
              aria-label="Close source"
            >
              <IconX />
            </button>
          )}
        </div>
      </div>

      <div className="source-filing">{cite.filing}</div>

      <MetaStrip cite={cite} />

      {isXbrl ? (
        <pre className="source-xbrl">
          {xbrlConceptName}
          {cite.fiscal_year != null && cite.value != null && cite.unit
            ? `\nFY${cite.fiscal_year} = ${cite.value.toLocaleString("en-US")} ${cite.unit}`
            : null}
        </pre>
      ) : (
        <blockquote className="source-quote">“{cite.quote}”</blockquote>
      )}

      <a
        href={cite.edgar_url}
        target="_blank"
        rel="noreferrer noopener"
        className="source-edgar"
      >
        View on EDGAR
        <IconArrowUpRight />
      </a>
    </div>
  );
}

function MetaStrip({ cite }: { cite: Citation }) {
  const period = cite.fiscal_year != null ? `FY${cite.fiscal_year}` : null;
  const form = cite.form || null;
  const section = cite.kind === "chunk" ? cite.section || null : null;
  const accession = cite.accession_no || null;

  const items: Array<[string, string]> = [];
  if (form) items.push(["Form", form]);
  if (period) items.push(["Period", period]);
  if (section) items.push(["Section", section]);
  if (accession) items.push(["Acc.", accession]);

  if (!items.length) return null;
  return (
    <div className="source-meta">
      {items.map(([k, v]) => (
        <span key={k} className="source-meta-pair">
          <span className="source-meta-key">{k}</span> {v}
        </span>
      ))}
    </div>
  );
}

function deriveNamespace(conceptKey: string | undefined): string | null {
  if (!conceptKey) return null;
  const idx = conceptKey.indexOf(":");
  return idx > 0 ? conceptKey.slice(0, idx) : null;
}

function deriveConceptName(conceptKey: string | undefined): string {
  if (!conceptKey) return "";
  const idx = conceptKey.indexOf(":");
  return idx > 0 ? conceptKey.slice(idx + 1) : conceptKey;
}
