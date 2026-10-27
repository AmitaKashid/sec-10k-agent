export type CoverageStatus = "complete" | "pending" | "missing";

export type Company = {
  ticker: string;
  name: string;
  sector?: string | null;
  latest_10k_period?: number | null;
  latest_10k_filed_at?: string | null;
  filings_count?: number | null;
  coverage_status?: CoverageStatus | null;
};

export type CompanyStats = {
  issuers: number;
  filings_indexed: number;
  xbrl_facts: number;
  coverage_complete: number;
  coverage_total: number;
  last_refresh: string | null;
};

export type Health = {
  status: "ok" | "degraded" | "unreachable" | string;
  db: string;
  graph: string;
};

export type ActiveRoute = "thread" | "companies";

// ── Citations ─────────────────────────────────────────────────────────────────

export type Citation = {
  id: number;
  kind: "chunk" | "xbrl";
  accession_no: string;
  filing: string;
  form: string;
  fiscal_year: number | null;
  ticker: string;
  company_name: string;
  section: string;
  quote: string;
  edgar_url: string;
  concept_key?: string;
  value?: number;
  unit?: string;
};

// ── Pipeline (Thinking node) ──────────────────────────────────────────────────

export type PipelineStepStatus = "pending" | "active" | "done" | "error";
export type PipelineStepId =
  | "plan"
  | "fetch"
  | "locate"
  | "retrieve"
  | "validate"
  | "cite_check";
export type PipelineStep = {
  id: PipelineStepId;
  label: string;
  status: PipelineStepStatus;
};

export const PIPELINE_LABELS: Record<PipelineStepId, string> = {
  plan: "Planning query path",
  fetch: "Fetching 10-K filings",
  locate: "Locating filing sections",
  retrieve: "Retrieving evidence",
  validate: "Validating answer",
  cite_check: "Cite-check cross ref",
};

// ── Turn / thread state ───────────────────────────────────────────────────────

export type Turn = {
  id: string;
  question: string;
  ticker: string;
  statusLine: string;
  isStreaming: boolean;
  streamedText: string;
  citations: Citation[];
  followups: string[];
  pipeline: PipelineStep[];
  elapsed: number;
  startedAt: number;
};

export type ActiveCiteFor = { turnId: string; id: number } | null;

// ── Backend wire types (FastAPI → /api/ask proxy) ─────────────────────────────

export type BackendChunkHit = {
  chunk_id: number;
  accession_no: string;
  item: "1A" | "7" | "8";
  seq: number;
  text: string;
  score: number;
  char_start: number;
  char_end: number;
  ticker: string | null;
  company_name: string | null;
  cik: string | null;
  fiscal_year: number | null;
  filing_date: string | null;
  section_title: string | null;
};

export type BackendXbrlFact = {
  cik: string;
  concept: string;
  concept_key: string;
  unit: string;
  value: number;
  period_end: string;
  fiscal_year: number;
  fiscal_period: string;
  accession_no: string;
  form: string;
};

export type BackendClaim = {
  text: string;
  kind: "number" | "prose";
  source_ref: BackendChunkHit | BackendXbrlFact | null;
  evidence_quote: string | null;
};

export type DonePayload = {
  request_id: string;
  // Echoed by the backend (mirrors the request thread_id, or a
  // server-minted one if the client omitted it). The client is
  // authoritative today, so this is informational.
  thread_id?: string | null;
  question_type: string | null;
  answer: string | null;
  claims: BackendClaim[];
  verified: boolean;
  refused: boolean;
  refusal_reason: string | null;
  refusal_code: string | null;
  refusal_message: string | null;
  meta_message: string | null;
  ticker?: string;
  fiscal_year?: number;
  // Langfuse trace deep-link, present only in dev/demo (EXPOSE_TRACE_URLS).
  // Wire-completeness for now; UI affordance is a follow-up.
  trace_url?: string | null;
};

// ── Tiered limiter (auth + rate limits) ──────────────────────────────────────

// Mirrors app/api/token_budget.py limit_response shape. See
// docs/auth_and_rate_limit_plan.md §Response shape.
export type LimitReason =
  | "anonymous_limit"
  | "authed_limit"
  | "anonymous_req_limit"
  | "authed_req_limit"
  | "input_too_large"
  | "anonymous_lifetime_limit"
  | "anon_ticker_not_allowed";

export type LimitErrorPayload = {
  detail?: string;
  reason: LimitReason;
  auth_required: boolean;
  limit: { used: number; budget: number; window_s: number };
};

export function isLimitError(data: unknown): data is LimitErrorPayload {
  if (typeof data !== "object" || data === null) return false;
  const d = data as Record<string, unknown>;
  return (
    typeof d.reason === "string" &&
    typeof d.auth_required === "boolean" &&
    typeof d.limit === "object" &&
    d.limit !== null
  );
}

// ── Frontend-facing constants ─────────────────────────────────────────────────

export const COMPANY_NAMES: Record<string, string> = {
  AAPL: "Apple",
  MSFT: "Microsoft",
  GOOGL: "Alphabet",
  AMZN: "Amazon",
  NVDA: "NVIDIA",
  META: "Meta",
};

export const ITEM_LABELS: Record<string, string> = {
  "1A": "Item 1A — Risk Factors",
  "7": "Item 7 — MD&A",
  "8": "Item 8 — Financial Statements",
};
