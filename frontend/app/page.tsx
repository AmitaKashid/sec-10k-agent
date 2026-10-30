"use client";

import {
  Suspense,
  useCallback,
  useEffect,
  useRef,
  useState,
  type FormEvent,
} from "react";
import { flushSync } from "react-dom";
import { useRouter, useSearchParams } from "next/navigation";
import { useUser } from "@clerk/nextjs";
import Sidebar from "@/components/Sidebar";
import { AnswerText } from "@/components/AnswerText";
import { FollowUps } from "@/components/FollowUps";
import { LimitModal } from "@/components/LimitModal";
import { LimitToast } from "@/components/LimitToast";
import { SourceCard } from "@/components/SourceCard";
import { Thinking } from "@/components/Thinking";
import {
  COMPANY_NAMES,
  PIPELINE_LABELS,
  isLimitError,
  type ActiveCiteFor,
  type Citation,
  type Company,
  type Health,
  type LimitErrorPayload,
  type PipelineStep,
  type PipelineStepId,
  type Turn,
} from "@/lib/types";

function IconSearch({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="11" cy="11" r="7" />
      <path d="M21 21l-4.3-4.3" />
    </svg>
  );
}
function IconArrowRight({ size = 16, strokeWidth = 2 }: { size?: number; strokeWidth?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M5 12h14M13 5l7 7-7 7" />
    </svg>
  );
}
function IconArrowUp({ size = 16, strokeWidth = 2 }: { size?: number; strokeWidth?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 19V5M5 12l7-7 7 7" />
    </svg>
  );
}
function IconChevronDown({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M6 9l6 6 6-6" />
    </svg>
  );
}

const VALID_STEP_IDS: PipelineStepId[] = ["plan", "fetch", "locate", "retrieve", "validate", "cite_check"];

function buildPipelineFromExecuted(executed: string[]): PipelineStep[] {
  const seen = new Set<string>();
  return executed
    .filter((id) => {
      if (seen.has(id)) return false;
      seen.add(id);
      return (VALID_STEP_IDS as string[]).includes(id);
    })
    .map((id) => ({
      id: id as PipelineStepId,
      label: PIPELINE_LABELS[id as PipelineStepId],
      status: "done" as const,
    }));
}

function makeTurn(question: string): Turn {
  return {
    id: `t-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    question,
    ticker: "",
    statusLine: "Thinking…",
    isStreaming: true,
    streamedText: "",
    citations: [],
    followups: [],
    pipeline: [],
    elapsed: 0,
    startedAt: performance.now(),
  };
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

export default function Page() {
  const [query, setQuery] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [activeCiteFor, setActiveCiteFor] = useState<ActiveCiteFor>(null);
  const [companies, setCompanies] = useState<Company[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  // Tier-aware limit signaling. modalPayload drives the LimitModal
  // (anonymous tier — sign-in gate). toastPayload drives the
  // LimitToast (authed tier — try-again-in-Ns). pendingQuestion is the
  // question text we preserved so the user can resubmit after sign-in.
  const [modalPayload, setModalPayload] = useState<LimitErrorPayload | null>(null);
  const [toastPayload, setToastPayload] = useState<LimitErrorPayload | null>(null);
  const [pendingQuestion, setPendingQuestion] = useState("");

  const abortRef = useRef<AbortController | null>(null);
  const typingTimers = useRef<ReturnType<typeof setTimeout>[]>([]);
  const rafRef = useRef<number | null>(null);

  // One conversation thread per visible thread of turns. Sent with every
  // /ask so the backend checkpointer can resolve follow-ups ("and 2024?").
  // Rotated whenever the user starts a new thread. Lazy-init once; the
  // null check keeps crypto.randomUUID() from running every render.
  const threadIdRef = useRef<string | null>(null);
  if (threadIdRef.current === null) threadIdRef.current = crypto.randomUUID();

  // Sign-out → reset to a clean home view. UserButton's afterSignOutUrl
  // handles the URL (when sign-out happens from /companies etc.); this
  // effect handles the React-state cleanup so we don't carry a logged-in
  // user's thread / citations / pending question into the anon session.
  const { isSignedIn } = useUser();
  const prevSignedInRef = useRef<boolean | undefined>(undefined);
  useEffect(() => {
    if (isSignedIn === undefined) return; // Clerk not loaded yet
    const wasSignedIn = prevSignedInRef.current;
    prevSignedInRef.current = isSignedIn;
    if (wasSignedIn === true && isSignedIn === false) {
      abortRef.current?.abort();
      setTurns([]);
      setQuery("");
      setActiveCiteFor(null);
      setModalPayload(null);
      setToastPayload(null);
      setPendingQuestion("");
      threadIdRef.current = crypto.randomUUID();
    }
  }, [isSignedIn]);

  const isStreaming = turns.some((t) => t.isStreaming);
  const showEmpty = turns.length === 0;

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const c = await fetch("/api/companies");
        if (!cancelled && c.ok) setCompanies((await c.json()) as Company[]);
      } catch {}
      try {
        const h = await fetch("/api/health", { signal: AbortSignal.timeout(3000) });
        if (!cancelled && h.ok) setHealth((await h.json()) as Health);
        else if (!cancelled) setHealth({ status: "unreachable", db: "-", graph: "-" });
      } catch {
        if (!cancelled) setHealth({ status: "unreachable", db: "-", graph: "-" });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const clearTimers = useCallback(() => {
    typingTimers.current.forEach(clearTimeout);
    typingTimers.current = [];
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  useEffect(() => () => clearTimers(), [clearTimers]);

  const updateTurn = useCallback(
    (id: string, patch: Partial<Turn> | ((t: Turn) => Partial<Turn>)) => {
      setTurns((prev) =>
        prev.map((t) =>
          t.id === id ? { ...t, ...(typeof patch === "function" ? patch(t) : patch) } : t,
        ),
      );
    },
    [],
  );

  const startElapsedTick = useCallback(
    (turnId: string, startedAt: number) => {
      const tick = () => {
        updateTurn(turnId, { elapsed: (performance.now() - startedAt) / 1000 });
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    },
    [updateTurn],
  );

  const simulateTyping = useCallback(
    (
      turnId: string,
      fullText: string,
      citations: Citation[],
      followups: string[],
      ticker: string,
    ) => {
      const words = fullText.split(/(\s+)/);
      let acc = "";
      words.forEach((w, i) => {
        const t = setTimeout(() => {
          acc += w;
          updateTurn(turnId, { streamedText: acc });
        }, i * 28);
        typingTimers.current.push(t);
      });
      const endMs = words.length * 28;
      const t1 = setTimeout(() => {
        clearTimers();
        updateTurn(turnId, () => ({
          isStreaming: false,
          citations,
          followups,
          ticker,
          statusLine: `Answered with ${citations.length} citation${citations.length !== 1 ? "s" : ""}`,
        }));
      }, endMs + 250);
      typingTimers.current.push(t1);
    },
    [updateTurn, clearTimers],
  );

  const submit = useCallback(
    async (overrideQ?: string) => {
      const q = (overrideQ ?? query).trim();
      if (!q) return;
      setQuery("");
      setTurns((prev) =>
        prev.map((t) =>
          t.isStreaming ? { ...t, isStreaming: false, statusLine: "Stopped (superseded)" } : t,
        ),
      );
      abortRef.current?.abort();
      clearTimers();

      const turn = makeTurn(q);
      setTurns((prev) => [...prev, turn]);
      setActiveCiteFor(null);
      startElapsedTick(turn.id, turn.startedAt);

      const ctrl = new AbortController();
      abortRef.current = ctrl;

      try {
        const res = await fetch("/api/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: q, thread_id: threadIdRef.current }),
          signal: ctrl.signal,
        });
        if (ctrl.signal.aborted) return;
        if (!res.body) {
          const errText = await res.text().catch(() => "");
          throw new Error(`HTTP ${res.status}: ${errText.slice(0, 120)}`);
        }
        // Non-2xx with an SSE body is the limit-error path (413/429) —
        // the proxy passes the structured payload through event:error.
        // Fall through to the reader loop instead of throwing.
        const contentType = res.headers.get("Content-Type") ?? "";
        if (!res.ok && !contentType.includes("text/event-stream")) {
          const errText = await res.text().catch(() => "");
          throw new Error(`HTTP ${res.status}: ${errText.slice(0, 120)}`);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buf = "";
        let resolved = false;

        // Tracks whether any token chunks have been streamed for this turn.
        // If yes, we keep the assembled stream as-is on `result`; if no
        // (e.g. OFF_TOPIC short-circuit), we fall back to refusal_message.
        let streamedTokens = "";

        outer: while (true) {
          const { done, value } = await reader.read();
          if (ctrl.signal.aborted) {
            await reader.cancel().catch(() => {});
            return;
          }
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          let sepIdx: number;
          while ((sepIdx = buf.indexOf("\n\n")) !== -1) {
            const block = buf.slice(0, sepIdx);
            buf = buf.slice(sepIdx + 2);
            const parsed = parseSseBlock(block);
            if (!parsed) continue;

            if (parsed.event === "token") {
              const piece = (parsed.data as { text?: string }).text ?? "";
              if (!piece) continue;
              if (streamedTokens === "") {
                // First token: stop the elapsed-time spinner messaging
                // and switch to the typing display. We don't yet know if
                // this is a refusal (composer) or META (concierge), so
                // use a neutral status line.
                clearTimers();
                updateTurn(turn.id, () => ({
                  statusLine: "Composing reply…",
                }));
              }
              streamedTokens += piece;
              const accumulated = streamedTokens;
              // flushSync forces React to render this state update
              // synchronously instead of batching it with later tokens
              // processed in the same inner-while iteration. Without
              // it, every token in the same network chunk collapses
              // into a single render and the user sees the whole
              // message appear at once.
              flushSync(() => {
                updateTurn(turn.id, () => ({
                  streamedText: accumulated,
                }));
              });
              continue;
            }

            if (parsed.event === "result") {
              console.log("[/api/ask result]", parsed.data);
              const r = parsed.data as {
                question_type: string | null;
                answer: string | null;
                citations: Citation[];
                refused: boolean;
                refusal_reason: string | null;
                refusal_code: string | null;
                refusal_message: string | null;
                meta_message: string | null;
                ticker: string;
                followups: string[];
                executed_steps: string[];
              };
              resolved = true;
              const realPipeline = buildPipelineFromExecuted(r.executed_steps ?? []);
              if (r.question_type === "META") {
                // Concierge path — greetings, capability questions.
                // Tokens already streamed; trust the accumulated text and
                // fall back to meta_message only if the stream was empty
                // (LLM-failure path on the backend).
                clearTimers();
                const metaText =
                  streamedTokens ||
                  r.meta_message ||
                  "Hi! I'm an SEC 10-K analyst. Ask me about a company's filing.";
                updateTurn(turn.id, () => ({
                  isStreaming: false,
                  statusLine: "Ready",
                  streamedText: metaText,
                  ticker: "",
                  citations: [],
                  followups: r.followups ?? [],
                  pipeline: realPipeline,
                }));
              } else if (r.refused) {
                clearTimers();
                // If tokens streamed (composer LLM path), trust the
                // accumulated text — it's already on screen. If not
                // (LLM-failure fallback, sync-only client), fall back
                // to refusal_message and finally to refusal_reason.
                const refusalText =
                  streamedTokens ||
                  r.refusal_message ||
                  r.refusal_reason ||
                  "I can only answer questions grounded in SEC 10-K filings.";
                updateTurn(turn.id, () => ({
                  isStreaming: false,
                  statusLine: "Not answerable from 10-K",
                  streamedText: refusalText,
                  ticker: r.ticker ?? "",
                  citations: r.citations ?? [],
                  followups: r.followups ?? [],
                  pipeline: realPipeline,
                }));
              } else if (r.answer) {
                updateTurn(turn.id, () => ({
                  statusLine: "Composing answer…",
                  pipeline: realPipeline,
                }));
                simulateTyping(turn.id, r.answer, r.citations, r.followups ?? [], r.ticker ?? "");
              } else {
                clearTimers();
                updateTurn(turn.id, () => ({
                  isStreaming: false,
                  statusLine: "No answer returned",
                  streamedText: "The agent returned no answer.",
                  pipeline: realPipeline,
                }));
              }
              break outer;
            } else if (parsed.event === "error") {
              // Tier-aware limit error — pass-through from the FastAPI
              // 413/429 body (see api/ask/route.ts). Drop the in-flight
              // turn, restore the question into the composer so the
              // user can retry, and show the modal (anon) or toast
              // (authed). See docs/auth_and_rate_limit_plan.md.
              if (isLimitError(parsed.data)) {
                clearTimers();
                setTurns((prev) => prev.filter((t) => t.id !== turn.id));
                setQuery(q);
                setPendingQuestion(q);
                if (parsed.data.auth_required) {
                  setModalPayload(parsed.data);
                } else {
                  setToastPayload(parsed.data);
                }
                resolved = true;
                break outer;
              }
              const msg = (parsed.data as { message?: string }).message ?? "An error occurred";
              clearTimers();
              updateTurn(turn.id, {
                isStreaming: false,
                statusLine: `Error: ${msg}`,
                streamedText: msg,
              });
              resolved = true;
              break outer;
            }
          }
        }

        if (!resolved) {
          clearTimers();
          updateTurn(turn.id, { isStreaming: false, statusLine: "Stream ended without result" });
        }
      } catch (err: unknown) {
        if (ctrl.signal.aborted) return;
        if (err instanceof Error && err.name === "AbortError") return;
        clearTimers();
        updateTurn(turn.id, {
          isStreaming: false,
          statusLine: `Error: ${err instanceof Error ? err.message : "request failed"}`,
        });
      }
    },
    [query, clearTimers, startElapsedTick, updateTurn, simulateTyping],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
    clearTimers();
    setTurns((prev) =>
      prev.map((t) => (t.isStreaming ? { ...t, isStreaming: false, statusLine: "Stopped" } : t)),
    );
  }, [clearTimers]);

  const onTickerHandoff = useCallback(
    (ticker: string) => {
      abortRef.current?.abort();
      clearTimers();
      setTurns([]);
      setActiveCiteFor(null);
      setQuery(`$${ticker.toUpperCase()} `);
      // Switching companies starts a fresh conversation.
      threadIdRef.current = crypto.randomUUID();
    },
    [clearTimers],
  );

  const [openSources, setOpenSources] = useState<Set<string>>(new Set());
  const onNewThread = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    clearTimers();
    threadIdRef.current = crypto.randomUUID();
    setTurns([]);
    setQuery("");
    setActiveCiteFor(null);
    setOpenSources(new Set());
    setModalPayload(null);
    setToastPayload(null);
    setPendingQuestion("");
  }, [clearTimers]);

  const toggleSources = useCallback((turnId: string) => {
    setOpenSources((prev) => {
      const next = new Set(prev);
      if (next.has(turnId)) next.delete(turnId);
      else next.add(turnId);
      return next;
    });
  }, []);

  const onCiteClick = useCallback((turnId: string, citeId: number | null) => {
    if (citeId == null) {
      setActiveCiteFor(null);
      return;
    }
    setActiveCiteFor({ turnId, id: citeId });
    setOpenSources((prev) => {
      if (prev.has(turnId)) return prev;
      const next = new Set(prev);
      next.add(turnId);
      return next;
    });
    setTimeout(() => {
      document
        .getElementById(`source-${turnId}-${citeId}`)
        ?.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 240);
  }, []);

  const companyCount = companies.length;
  const healthOk = health?.status === "ok";

  return (
    <div className="app">
      <Suspense fallback={null}>
        <TickerPrefillEffect onTicker={onTickerHandoff} />
      </Suspense>

      <LimitModal
        payload={modalPayload}
        pendingQuestion={pendingQuestion}
        onDismiss={() => setModalPayload(null)}
      />
      <LimitToast payload={toastPayload} onDismiss={() => setToastPayload(null)} />

      <Sidebar activeRoute="thread" companyCount={companyCount} healthOk={healthOk} onNewThread={onNewThread} />

      <section className="main">
        <div className="topbar">
          {showEmpty ? (
            <div className="wordmark">
              SEC <b>Analyst</b>
            </div>
          ) : (
            <ThreadCrumb turns={turns} />
          )}
        </div>

        {showEmpty ? (
          <Empty
            companyCount={companyCount}
            query={query}
            setQuery={setQuery}
            onSubmit={(e) => {
              e.preventDefault();
              void submit();
            }}
          />
        ) : (
          <div className="thread">
            {turns.map((turn, i) => {
              const activeCiteId = activeCiteFor?.turnId === turn.id ? activeCiteFor.id : null;
              const sourcesOpen = openSources.has(turn.id);
              const showSources = !turn.isStreaming && turn.citations.length > 0;
              const showFollowups = !turn.isStreaming && turn.followups.length > 0;
              return (
                <article key={turn.id} className="turn">
                  {i > 0 && <div className="turn-sep" aria-hidden="true" />}
                  <header className="turn-head">
                    {turn.ticker && (
                      <div className="turn-ticker-row">
                        <span className="turn-ticker">${turn.ticker}</span>
                      </div>
                    )}
                    <h2 className="turn-question">{turn.question}</h2>
                  </header>

                  <Thinking steps={turn.pipeline} isStreaming={turn.isStreaming} elapsed={turn.elapsed} />

                  <div>
                    <AnswerText
                      text={turn.streamedText}
                      activeCiteId={activeCiteId}
                      citations={turn.citations}
                      onCiteClick={(id) => onCiteClick(turn.id, id)}
                    />
                    {turn.isStreaming && <span className="caret" aria-hidden="true" />}
                  </div>

                  {showSources && (
                    <div className="sources">
                      <button
                        type="button"
                        onClick={() => toggleSources(turn.id)}
                        className="sources-toggle"
                        aria-expanded={sourcesOpen}
                      >
                        <span>Sources</span>
                        <span className="sources-count">{turn.citations.length}</span>
                        <span className={`sources-chev ${sourcesOpen ? "is-open" : ""}`}>
                          <IconChevronDown />
                        </span>
                      </button>
                      {sourcesOpen && (
                        <div className="sources-grid">
                          {turn.citations.map((c) => (
                            <div key={c.id} id={`source-${turn.id}-${c.id}`}>
                              <SourceCard cite={c} />
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {showFollowups && (
                    <div className="turn-followups">
                      <FollowUps items={turn.followups} onPick={(q) => void submit(q)} />
                    </div>
                  )}
                </article>
              );
            })}
          </div>
        )}

        {!showEmpty && (
          <div className="composer-sticky">
            <form
              className="composer-wrap"
              onSubmit={(e) => {
                e.preventDefault();
                void submit();
              }}
            >
              <div className="composer-glow" aria-hidden="true" />
              <div className="composer">
                <span className="search-ic"><IconSearch /></span>
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  disabled={isStreaming}
                  placeholder="Ask a follow-up about a 10-K…"
                />
                {isStreaming ? (
                  <button
                    type="button"
                    onClick={stop}
                    className="submit is-stop"
                    aria-label="Stop"
                  >
                    <span className="btn-stop-square" aria-hidden="true" />
                  </button>
                ) : (
                  <button
                    type="submit"
                    className="submit"
                    disabled={!query.trim()}
                    aria-label="Submit"
                  >
                    <IconArrowUp />
                  </button>
                )}
              </div>
            </form>
          </div>
        )}
      </section>
    </div>
  );
}

function Empty({
  companyCount,
  query,
  setQuery,
  onSubmit,
}: {
  companyCount: number;
  query: string;
  setQuery: (v: string) => void;
  onSubmit: (e: FormEvent<HTMLFormElement>) => void;
}) {
  return (
    <div className="hero">
      <div className="eyebrow">
        <span className="dot"></span>
        <span>Grounded in {companyCount || "—"} US issuers · 10-Ks since 2015</span>
      </div>
      <h1>
        Ask anything about a US public company.{" "}
        <em>Get the citation, or get nothing.</em>
      </h1>
      <p>
        Every claim links to the exact 10-K paragraph or XBRL fact. When the
        filing doesn&apos;t say it, we say so.
      </p>
      <form className="composer-wrap" onSubmit={onSubmit}>
        <div className="composer-glow" aria-hidden="true" />
        <div className="composer">
          <span className="search-ic"><IconSearch /></span>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask anything about a 10-K…"
            autoFocus
          />
          <button type="submit" className="submit" disabled={!query.trim()} aria-label="Submit">
            <IconArrowRight />
          </button>
        </div>
      </form>
    </div>
  );
}

function ThreadCrumb({ turns }: { turns: Turn[] }) {
  const wordmark = (
    <div className="wordmark">
      SEC <b>Analyst</b>
    </div>
  );
  if (!turns.length) return wordmark;
  const latest = turns[turns.length - 1];
  const firstCite = latest.citations[0];
  const ticker = firstCite?.ticker || latest.ticker;
  if (!ticker) return wordmark;
  const companyName = firstCite?.company_name || COMPANY_NAMES[ticker] || null;
  const form = firstCite?.form || "10-K";
  const fy = firstCite?.fiscal_year ?? null;
  return (
    <div className="crumb">
      <span className="crumb-ticker">${ticker}</span>
      {companyName && <span className="crumb-name">{companyName}</span>}
      <span className="crumb-sep">·</span>
      <span className="crumb-meta">
        {form}
        {fy != null ? ` · FY${fy}` : ""}
      </span>
    </div>
  );
}

function TickerPrefillEffect({ onTicker }: { onTicker: (ticker: string) => void }) {
  const params = useSearchParams();
  const router = useRouter();
  const ticker = params.get("ticker");
  useEffect(() => {
    if (!ticker) return;
    onTicker(ticker);
    router.replace("/");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticker]);
  return null;
}
