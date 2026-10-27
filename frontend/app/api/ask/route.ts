import type { NextRequest } from "next/server";
import { auth } from "@clerk/nextjs/server";
import type {
  BackendChunkHit,
  BackendClaim,
  BackendXbrlFact,
  Citation,
  DonePayload,
} from "@/lib/types";
import { COMPANY_NAMES, ITEM_LABELS } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const FASTAPI_URL = process.env.FASTAPI_URL ?? "http://localhost:8000";

// Build upstream headers, attaching the Clerk JWT as a Bearer token when
// the user is signed in. Anonymous requests carry no Authorization
// header — the backend resolves them as the anon tier (IP-keyed).
// X-API-Key is no longer forwarded for /ask; the backend's identity
// dependency replaces it (see docs/auth_and_rate_limit_plan.md).
async function buildUpstreamHeaders(accept: string): Promise<Record<string, string>> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: accept,
  };
  try {
    const { getToken } = await auth();
    const token = await getToken();
    if (token) headers.Authorization = `Bearer ${token}`;
  } catch {
    // No Clerk session — proceed as anonymous.
  }
  return headers;
}

const FOLLOWUPS: Record<string, string[]> = {
  AAPL: [
    "How did Services margin compare to Products?",
    "What did Apple disclose about Greater China?",
    "Quantify Apple's FX headwind in FY2024",
  ],
  MSFT: [
    "What was Azure's reported growth rate?",
    "How does Microsoft describe AI capex commitments?",
    "Which enterprise customers does Microsoft single out?",
  ],
  GOOGL: [
    "What's Google Cloud's operating income trend?",
    "Which antitrust cases does Alphabet disclose?",
    "How is YouTube ad revenue described?",
  ],
  AMZN: [
    "What did Amazon say about International profitability?",
    "How much capex is Amazon guiding for 2025?",
    "What's Amazon's stated AWS competitive position?",
  ],
  NVDA: [
    "Which customers are NVIDIA's largest concentration risks?",
    "What did NVIDIA disclose about China revenue?",
    "How is Blackwell architecture referenced?",
  ],
  META: [
    "How much did Reality Labs lose in FY2024?",
    "What ad-tech changes is Meta navigating?",
    "How is AI capex discussed in Meta's 10-K?",
  ],
};

function isChunkHit(ref: BackendChunkHit | BackendXbrlFact): ref is BackendChunkHit {
  return "chunk_id" in ref && "item" in ref;
}

function buildEdgarUrl(accessionNo: string, ticker: string, cik?: string): string {
  // Filing-index deep link: needs CIK (without leading zeros) + accession number.
  // SEC's browse-edgar `filenum` is the SEC file number, NOT an accession — using
  // it for accessions returned an empty page.
  if (accessionNo && cik) {
    const cikInt = parseInt(cik, 10);
    if (!Number.isNaN(cikInt)) {
      const noDashes = accessionNo.replace(/-/g, "");
      return `https://www.sec.gov/Archives/edgar/data/${cikInt}/${noDashes}/${accessionNo}-index.htm`;
    }
  }
  // Fallback: company 10-K list. Used when we don't have a CIK
  // (chunk hits don't carry one on the wire yet).
  return `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=${ticker}&type=10-K`;
}

function buildCitations(
  claims: BackendClaim[],
  fallbackTicker: string,
  fallbackFiscalYear?: number,
): Citation[] {
  // Citation `id` MUST equal the claim's 1-based position so the validate
  // prompt's [N] markers align with the citation list.
  const out: Citation[] = [];
  claims.forEach((c, i) => {
    if (c.source_ref == null) return;
    const ref = c.source_ref;
    const id = i + 1;
    if (isChunkHit(ref)) {
      const ticker = ref.ticker ?? fallbackTicker;
      const companyName = ref.company_name ?? COMPANY_NAMES[ticker] ?? ticker;
      const fy = ref.fiscal_year ?? fallbackFiscalYear ?? null;
      const itemLabel = ITEM_LABELS[ref.item] ?? `Item ${ref.item}`;
      const section = ref.section_title ? `${itemLabel} · ${ref.section_title}` : itemLabel;
      out.push({
        id,
        kind: "chunk",
        accession_no: ref.accession_no,
        filing: `${companyName} 10-K${fy ? ` FY${fy}` : ""}`,
        form: "10-K",
        fiscal_year: fy,
        ticker,
        company_name: companyName,
        section,
        quote: c.evidence_quote ?? c.text,
        edgar_url: buildEdgarUrl(ref.accession_no, ticker, ref.cik ?? undefined),
      });
      return;
    }
    const xbrl = ref as BackendXbrlFact;
    const ticker = fallbackTicker;
    const companyName = COMPANY_NAMES[ticker] ?? ticker;
    out.push({
      id,
      kind: "xbrl",
      accession_no: xbrl.accession_no,
      filing: `${companyName} ${xbrl.form} FY${xbrl.fiscal_year}`,
      form: xbrl.form,
      fiscal_year: xbrl.fiscal_year,
      ticker,
      company_name: companyName,
      section: "Financial Data (XBRL)",
      quote: `${xbrl.concept_key}: ${xbrl.value} ${xbrl.unit}`,
      edgar_url: buildEdgarUrl(xbrl.accession_no, ticker, xbrl.cik),
      concept_key: xbrl.concept_key,
      value: xbrl.value,
      unit: xbrl.unit,
    });
  });
  return out;
}

function emit(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function parseSseBlock(block: string): { event: string; data: unknown } | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  if (!dataLines.length) return null;
  try {
    return { event, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return null;
  }
}

export async function POST(request: NextRequest): Promise<Response> {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: "Invalid JSON body" }, { status: 400 });
  }
  const question = (body as { question?: unknown })?.question;
  if (typeof question !== "string" || question.trim().length === 0) {
    return Response.json({ error: "Missing or empty 'question'" }, { status: 400 });
  }
  // Conversation thread id (client-authoritative — see
  // docs/conversation_memory_plan.md §9). Forwarded so the backend
  // checkpointer can resolve follow-ups. Only pass through a sane
  // string; never relay arbitrary client data into the upstream body.
  // Charset-constrained: must match the backend AskRequest.thread_id
  // pattern (^[A-Za-z0-9_-]{1,128}$). Anything else is dropped here so a
  // malformed id never reaches the upstream / the checkpointer key.
  const rawThreadId = (body as { thread_id?: unknown })?.thread_id;
  const threadId =
    typeof rawThreadId === "string" && /^[A-Za-z0-9_-]{1,128}$/.test(rawThreadId)
      ? rawThreadId
      : undefined;
  const upstreamBodyJson = JSON.stringify(
    threadId ? { question, thread_id: threadId } : { question },
  );

  // Consume FastAPI's SSE on the server so we can collect executed_steps
  // (sse_starlette can lose flushes through the Next proxy on Node 20+).
  // Browser receives one consolidated `result` event.
  let upstream: Response;
  try {
    upstream = await fetch(`${FASTAPI_URL}/ask`, {
      method: "POST",
      headers: await buildUpstreamHeaders("text/event-stream"),
      body: upstreamBodyJson,
    });
  } catch {
    return new Response(emit("error", { message: "Could not reach backend" }), {
      status: 502,
      headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" },
    });
  }

  if (!upstream.ok || !upstream.body) {
    // 413 / 429 from the backend's tiered limiter carry a structured body
    // (reason, auth_required, limit). Pass it through verbatim so the
    // client can decide between "sign in" modal vs "slow down" toast.
    // See docs/auth_and_rate_limit_plan.md §Response shape.
    const raw = await upstream.text().catch(() => "");
    let parsed: Record<string, unknown> | null = null;
    try {
      parsed = raw ? (JSON.parse(raw) as Record<string, unknown>) : null;
    } catch {
      parsed = null;
    }
    const errorPayload =
      parsed ?? { message: `Backend ${upstream.status}: ${raw.slice(0, 200)}` };
    return new Response(emit("error", errorPayload), {
      status: upstream.status,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
      },
    });
  }

  // Streaming pass-through. We forward `event: token` chunks immediately
  // (so the refusal_composer's typing effect is real, not simulated) and
  // accumulate `event: status` into `executed_steps` for the final
  // `event: result` we emit when the upstream `event: done` arrives.
  const upstreamBody = upstream.body;
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const reader = upstreamBody.getReader();
      const decoder = new TextDecoder();
      const encoder = new TextEncoder();
      let buf = "";
      const executedSteps: string[] = [];
      let done: DonePayload | null = null;
      let errorMsg: string | null = null;

      // Yield to the I/O loop after every enqueue. Without this, Node may
      // buffer many small writes into a single TCP segment and the browser
      // sees the entire stream as one chunk — which defeats the per-token
      // streaming UX. A microtask hop forces the runtime to flush the
      // queued bytes to the response socket between tokens.
      const send = async (event: string, data: unknown) => {
        controller.enqueue(encoder.encode(emit(event, data)));
        await new Promise((resolve) => setTimeout(resolve, 0));
      };

      try {
        while (true) {
          const { done: streamDone, value } = await reader.read();
          if (streamDone) break;
          buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
          let sepIdx: number;
          while ((sepIdx = buf.indexOf("\n\n")) !== -1) {
            const block = buf.slice(0, sepIdx);
            buf = buf.slice(sepIdx + 2);
            const parsed = parseSseBlock(block);
            if (!parsed) continue;
            if (parsed.event === "status") {
              const step = (parsed.data as { step?: string }).step;
              if (step) executedSteps.push(step);
            } else if (parsed.event === "token") {
              // Pass-through immediately for real streaming UX.
              const text = (parsed.data as { text?: string }).text;
              if (text) await send("token", { text });
            } else if (parsed.event === "done") {
              done = parsed.data as DonePayload;
            } else if (parsed.event === "error") {
              errorMsg =
                (parsed.data as { detail?: string; error?: string }).detail ??
                (parsed.data as { error?: string }).error ??
                "Backend error";
            }
          }
        }
      } catch (err) {
        errorMsg = err instanceof Error ? err.message : "stream read error";
      }

      if (!done) {
        // The JSON-sync fallback re-issues the upstream POST, which
        // burns a second request slot AND a second token reservation
        // on the backend's tiered limiter. Only retry when we believe
        // the stream ended cleanly without a `done` event (a parser /
        // transport edge case) — never when the backend already
        // emitted `event: error` (rate-limit, graph exception, etc.)
        // or when the local read raised an exception. In those cases
        // we already have a definitive answer; retrying just double-
        // charges the user's quota. See
        // docs/auth_and_rate_limit_plan.md.
        const shouldRetry = errorMsg == null;
        if (shouldRetry) {
          console.warn(
            "[/api/ask] SSE ended without done event — falling back to JSON sync. " +
              `executedSteps=${JSON.stringify(executedSteps)} bufLen=${buf.length}`,
          );
          try {
            const sync = await fetch(`${FASTAPI_URL}/ask`, {
              method: "POST",
              headers: await buildUpstreamHeaders("application/json"),
              body: upstreamBodyJson,
            });
            if (sync.ok) {
              done = (await sync.json()) as DonePayload;
            }
          } catch (e) {
            console.error("[/api/ask] JSON fallback also failed:", e);
          }
        }
        if (!done) {
          await send("error", {
            message: errorMsg ?? "Backend stream ended without a done event",
          });
          controller.close();
          return;
        }
      }

      const ticker = done.ticker ?? "";
      const citations = buildCitations(done.claims ?? [], ticker, done.fiscal_year);
      const followups = FOLLOWUPS[ticker] ?? [];

      console.log(
        `[/api/ask] ticker=${ticker} answer.len=${done.answer?.length ?? 0} ` +
          `claims=${done.claims?.length ?? 0} ` +
          `claims_with_source=${(done.claims ?? []).filter((c) => c.source_ref != null).length} ` +
          `citations=${citations.length} executed_steps=${JSON.stringify(executedSteps)}`,
      );

      await send("result", {
        question_type: done.question_type,
        answer: done.answer,
        citations,
        refused: done.refused,
        refusal_reason: done.refusal_reason,
        refusal_code: done.refusal_code,
        refusal_message: done.refusal_message,
        meta_message: done.meta_message,
        ticker,
        followups,
        executed_steps: executedSteps,
      });
      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
