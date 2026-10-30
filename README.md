# SEC 10-K Diligence Agent

A deployable AI agent that answers natural-language questions about US public companies, grounding answers in SEC filings with verifiable citations. Fetches directly from SEC EDGAR (no web search), validates numerical claims against structured XBRL data, and **refuses** to answer when it cannot ground a claim.

The grounding approach is structural: supported numerical answers come from XBRL rather than prose, and unsupported requests follow a structured refusal path. Citation checks verify source alignment; they do not establish complete answer quality.

## What it does

- **Type A — Numerical lookup** (e.g. "What was Apple's R&D spend in FY2024?"): plan → XBRL fact lookup → format. Restricted to a 10-item us-gaap whitelist (revenue, net income, R&D, total assets, long-term debt, cash, diluted EPS, shares out, opex, gross profit).
- **Type B — Qualitative single-filing** (e.g. "What cybersecurity risks did Microsoft disclose in their latest 10-K?"): plan → locate Item → hybrid retrieval → validate → cite-check. Restricted to Item 1A (Risk Factors), Item 7 (MD&A), Item 8 (Financial Notes).
- **META** — greetings / capability / how-to small talk, handled by a `concierge` node.
- **REFUSE** — anything else (YoY, cross-company, non-whitelisted concept, non-10-K source) is refused with HTTP 200 and `refused=true`.

Year-over-year and cross-company questions are explicit **non-goals** for the MVP.

## Architecture

Two parallel data pipelines feed one LangGraph agent:

1. **Prose pipeline** — EDGAR HTML → `edgartools` section extraction (Item 1A / 7 / 8) → sliding-window chunks (~500 tokens) → Gemini `gemini-embedding-2` (1,536 dimensions) → pgvector. Retrieval is cosine similarity restricted to a single `(company, item)` and uses hybrid semantic + PostgreSQL full-text search fused with Reciprocal Rank Fusion, plus multi-query decomposition for compound questions.
2. **XBRL pipeline** — `data.sec.gov` company-facts JSON → structured rows. Numbers are **never** taken from prose chunks — only from XBRL. The cite-check node enforces this with exact-equality on numerical claims.

### Agent graph (LangGraph)

```text
                   ┌─ Filing ─→ access / fetch ─→ locate ─→ retrieve ─→ validate ─→ cite_check ─→ END
                   │                                                   └─ cache hit ──────────→ END
plan ──────────────┤
                   ├─ META ───→ concierge ─────────────────────────────────────────────────────→ END
                   │
                   └─ REFUSE ─→ refusal_composer ───────────────────────────────────────────────→ END
```

Access and source gates can also route to refusal. Groq-hosted `openai/gpt-oss-20b` handles planning and refusal composition; `openai/gpt-oss-120b` handles qualitative evidence selection. Numerical formatting in the Groq path uses Python. Every graph node is wrapped in a Langfuse span under a root `/ask` trace.

### Tech stack

- **Backend:** Python · `uv` · FastAPI · LangGraph · SQLAlchemy · Alembic
- **Storage:** Postgres + `pgvector` (chunks, XBRL facts, query log, answer cache, optional LangGraph checkpointer)
- **Embeddings / LLMs:** Gemini embeddings · Groq inference (allowlisted per-request model overrides); OpenAI / Anthropic adapters remain available
- **Frontend:** Next.js 16.4.3 (App Router) on Vercel, Clerk auth
- **Observability:** Langfuse v4; request traces, node spans, and conversation sessions; optional until credentials are configured
- **Deploy targets:** Render (Dockerized FastAPI) · Neon (PostgreSQL + pgvector) · Vercel (Next.js)
- **Streaming:** SSE from FastAPI through the Next.js proxy. Verified answer text is released after citation checking and animated by the UI; refusal and concierge responses can stream tokens.
- **Authentication:** Clerk JWT verification with user-scoped PostgreSQL conversation checkpoints

Operating cost depends on hosting plans, provider quotas, and usage. Free-tier limits and cold starts apply.

## Repository layout

```
sec-10k-agent/
├── README.md, .env.example, docker-compose.yml
├── backend/                  # FastAPI + LangGraph
│   ├── pyproject.toml        # uv-managed; dev extras: pytest, ruff, mypy
│   ├── alembic/              # migrations (0001…0005)
│   ├── scripts/              # ingest_universe.py, seed_dev.py, smoke_edgar.py, prune_checkpoints.py
│   ├── app/
│   │   ├── main.py
│   │   ├── api/              # routes, schemas, auth, token_budget, anon_quota, usage_tracker
│   │   ├── agent/            # graph.py, state.py, prompts.py, tools.py, nodes/*
│   │   ├── data/             # edgar_client.py, parser.py, chunker.py, embedder.py, xbrl.py, ingestion.py, db.py, models.py
│   │   ├── observability/    # Langfuse wiring
│   │   └── settings.py       # pydantic-settings
│   └── tests/                # unit / integration (live marker opt-in)
├── frontend/                 # Next.js 16.4.3 + React 19
│   ├── app/                  # page.tsx, companies/, layout.tsx
│   │   └── api/{ask,companies,health}/route.ts   # server-side proxies; hold API key + Clerk JWT
│   ├── components/           # AnswerText, FollowUps, Sidebar, SourceCard, Thinking, LimitModal, AuthControl
│   └── lib/types.ts
├── evals/                    # golden_dataset.json, adversarial.json, run_evals.py, judge.py, metrics.py
├── backend/config/tickers.csv  # 50-company demo universe (inside backend/ so it ships in the Docker image)
└── .github/workflows/        # backend, database, and frontend checks
```

The same Pydantic models defined in `backend/app/agent/state.py` are reused by `backend/app/api/schemas.py` — they are not duplicated.

## API

| Method | Path          | Purpose |
|--------|---------------|---------|
| `POST` | `/ask`        | Answer a question — SSE stream (default) or sync JSON (`Accept: application/json`). Optional `thread_id` for opt-in conversation memory; optional `plan_model_override` / `validate_model_override` (whitelist-gated). |
| `GET`  | `/companies`  | List ingested universe (ticker, name, sector) |
| `GET`  | `/companies/stats` | Corpus statistics; authenticated user or server-side API key |
| `GET`  | `/health`     | Liveness + DB + graph compilation check (no auth) |

### Identity & limits on `/ask`

`/ask` resolves an `Identity(tier, key)` per request:

- `tier="user"` when a valid Clerk JWT (`Authorization: Bearer …`) verifies against Clerk's JWKS. Present-but-invalid tokens → **401** (no silent downgrade).
- `tier="anon"` otherwise; key is the remote IP.

Defaults (env-configurable, in-process sliding window):

| Tier  | Reqs/min | Tokens/min | Max input tokens |
|-------|----------|------------|------------------|
| anon  | 5        | 5 000      | 1 000            |
| user  | 60       | 50 000     | 2 000            |

The limits above apply when the token limiter is enabled; the development template disables it. Anonymous users also have an optional lifetime per-IP request cap and a configured ticker allowlist (AAPL, MSFT, GOOGL, AMZN, NVDA). 413 / 429 responses share a single shape with `reason`, `auth_required`, and `limit: {used, budget, window_s}` so the frontend can branch between sign-in modal and toast.

`/companies` keeps the legacy `X-API-Key` header path; the browser never holds the API key — the Next.js route proxies and injects it server-side.

## Local development

Clone [AmitaKashid/sec-10k-agent](https://github.com/AmitaKashid/sec-10k-agent). Use Python 3.12, uv, Node.js 24, and Docker Desktop / Compose.

```bash
git clone https://github.com/AmitaKashid/sec-10k-agent.git
cd sec-10k-agent
```

Windows CMD — copy and fill the environment files:

```bat
copy .env.example .env
copy frontend\.env.example frontend\.env.local
```

On macOS/Linux, use `cp .env.example .env` and `cp frontend/.env.example frontend/.env.local`.

Set your real SEC contact email, Gemini and Groq keys in the root `.env`. Configure the same Clerk application in both environment files. Set `FASTAPI_URL=http://127.0.0.1:8000` in `frontend/.env.local`; match its private `API_KEY` to the backend. Include `http://localhost:3000` in `CLERK_AUTHORIZED_PARTIES` for local sign-in.

```bash
# From the repository root: database, migrations, demo ingestion
docker compose up -d postgres
cd backend
uv sync --frozen --extra dev
uv run alembic upgrade head
uv run python -m scripts.seed_dev --concurrency 1

# Start the API from backend/
uv run uvicorn app.main:app --reload
```

In another terminal, from `frontend/`:

```bash
npm ci
npm run dev
```

Open [localhost:3000](http://localhost:3000). Seeding makes real SEC and embedding requests; completion time depends on the corpus, network, and provider quotas.

### Backend commands (run from `backend/`)

| Task | Command |
|---|---|
| Install | `uv sync --extra dev` |
| Run API | `uv run uvicorn app.main:app --reload` |
| Lint | `uv run ruff check .` |
| Format check | `uv run ruff format --check .` |
| Type check | `uv run mypy app` |
| Tests (default) | `uv run --extra dev python scripts/run_tests.py --suite unit` |
| Database integration | `uv run --extra dev python scripts/run_tests.py --suite database` |
| Live tests (real EDGAR/LLMs) | `uv run --extra dev python scripts/run_tests.py --suite live` — opt-in only |
| Migrations | `uv run alembic upgrade head` |
| Ingest universe | `uv run python -m scripts.ingest_universe --concurrency 1` |
| Evals (from repo root) | `uv run --project backend python -m evals.run_evals` |

### Frontend commands (run from `frontend/`)

| Task | Command |
|---|---|
| Install | `npm ci` |
| Dev | `npm run dev` |
| Production build (also typechecks) | `npm run build` |
| Start built | `npm run start` |
| Type check | `npm run typecheck` |
| Lint | `npm run lint` |
| Dependency audit | `npm audit` |

### Verification status

The latest reported local run on **30 October 2026** passed **347 unit/evaluation-logic tests** and **29 PostgreSQL integration tests**: **376 selected tests total**. Two corpus checks and two live-service checks were deselected from those standard runs. Standard suites mock external model/service calls; database integration uses real PostgreSQL.

GitHub Actions runs the offline backend suite, PostgreSQL/pgvector integration tests, and frontend typecheck/lint. The test runner handles Windows selector-event-loop compatibility and validates the isolated `_test` database. Browser end-to-end verification remains manual.

## Configuration

All backend config flows through `app/settings.py` (pydantic-settings, reads `.env`). Key vars:

- `DATABASE_URL` — Postgres (with `pgvector`); only thing that differs between local and Neon / Render
- `SEC_USER_AGENT` — required; SEC rejects requests without contact info
- `GROQ_API_KEY`, `GEMINI_API_KEY` — configured chat and embedding providers; paid OpenAI/Anthropic credentials are not required for this configuration
- `PLAN_MODEL`, `VALIDATE_MODEL`, `REFUSAL_MODEL`, `ALLOWED_MODELS` — Groq model selection
- `EMBEDDING_PROVIDER=gemini`, `GEMINI_EMBEDDING_MODEL=gemini-embedding-2` — 1,536-dimensional embeddings
- `API_KEY`, `RATE_LIMIT_PER_MINUTE` — backend auth + coarse slowapi limit
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` — leave unset to no-op
- `LANGGRAPH_CHECKPOINTER_ENABLED` — PostgreSQL conversation memory for authenticated users; disabled in the example environment; falls back to stateless operation if initialization fails
- `CHECKPOINT_TTL_DAYS` — checkpoint cleanup age, default seven days
- `LANGFUSE_CAPTURE_IO` — controls whether question/answer content is sent to Langfuse
- `EXPOSE_TRACE_URLS` — dev/demo true, prod false
- `ANON_LIFETIME_ENABLED`, `ANON_LIFETIME_LIMIT`, `ANON_HASH_SALT`, `ANON_TICKER_ALLOWLIST` — anonymous-tier hardening

All EDGAR HTTP **must** go through `app/data/edgar_client.py` (token-bucket limiter at 5 req/s, half the 10 req/s SEC ceiling). Never call `requests.get` against EDGAR directly.

## Caching

- Ticker→CIK map: in-memory, refreshed daily
- Submissions feed: on-disk, short TTL
- Filing HTML: on-disk, **never invalidated** (filings are immutable once filed; keyed by accession number)
- Embeddings: persisted forever in pgvector for the same reason
- Answer cache: Postgres `answer_cache` table, exact-match on `(question_normalized, type, cik, fiscal_year, sources_hash)`; lookup point is between Retrieve and Validate
- Conversation memory (opt-in): LangGraph `AsyncPostgresSaver`, keyed by a hash of authenticated user ID and client thread ID. Anonymous requests remain stateless. New threads use new identifiers, and retrieved evidence resets between turns.

## Evals

`evals/` contains golden and adversarial datasets, an LLM-as-judge, and citation, refusal, numerical, unsupported-claim, retrieval, and latency metrics. Unit tests of this machinery run in CI; a complete live answer-quality benchmark does not.

A historical run on **27 October 2026** evaluated **six qualitative golden questions** across four companies:

| Metric | Observed result |
|---|---:|
| Citation accuracy | 100% within this sample |
| Unsupported claims, as assessed by the judge | 0 / 30 |
| Reported median graph latency | 10,308 ms |

This sample contained **no numerical or adversarial questions**. It does not establish numerical accuracy, adversarial refusal reliability, or general production performance. The judge shared the validation model family; the latency measurement was in-process rather than browser-to-hosted-service. Full live evaluation is deferred and remains a project limitation.

## Deployment status

- **Frontend:** [sec-10k-agent-three.vercel.app](https://sec-10k-agent-three.vercel.app/)
- **Backend:** [sec-10k-backend.onrender.com](https://sec-10k-backend.onrender.com/health)
- **Database:** Neon PostgreSQL + pgvector
- **Memory:** authenticated follow-ups and persistence across backend restarts verified
- **Observability:** live Langfuse v4 traces and conversation sessions verified

Render builds from `backend/`; Vercel uses `frontend/`. The Docker image runs as a non-root user, applies migrations at startup, and listens on the platform's `PORT`.

The verified Linux-container Neon TLS configuration uses `?ssl=verify-full` in the asyncpg database URL and `PGSSLROOTCERT=/etc/ssl/certs/ca-certificates.crt` as a separate environment variable. Keep database credentials and provider/Clerk secrets in private environment settings.

## Scope (explicitly out of MVP)

International filings · real-time prices · earnings call transcripts · year-over-year calculations · cross-company comparison · unsupported numerical concepts · sharded Postgres · Redis · background queues.

The configured universe contains **50 tickers**, but coverage depends on actual ingestion. The **29 October 2026 deployment snapshot** contained 5 companies, 15 filings, 1,358 chunks, and 343 XBRL facts. Older fiscal years are not automatically available.

### Limitations

- Full numerical, adversarial, and multi-turn live evaluation remains pending.
- A supported citation does not prove the answer is complete or the evidence selection is correct.
- Persistent memory requires sign-in and an initialized checkpointer; it supplies conversational context, not new financial evidence.
- Request/token windows are per process, not distributed quota accounting.
- Free hosting can introduce cold starts; provider quotas can delay or interrupt requests.
- The automated suite does not include browser end-to-end tests.

## Documentation

- [`backend/app/agent/graph.py`](backend/app/agent/graph.py) — graph routing and node execution
- [`backend/app/settings.py`](backend/app/settings.py) — configuration and defaults
- [`backend/scripts/run_tests.py`](backend/scripts/run_tests.py) — test suites and database safeguards
- [`evals/`](evals/) — evaluation datasets, runner, judge, and metrics
- [`.github/workflows/`](.github/workflows/) — continuous integration

## Maintainer

Amita Kashid
