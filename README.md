# SEC Analyst

**Ask a question. Inspect the evidence. Continue the conversation.**

[![Backend tests](https://github.com/AmitaKashid/sec-10k-agent/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/AmitaKashid/sec-10k-agent/actions/workflows/tests.yml)
[![Database integration](https://github.com/AmitaKashid/sec-10k-agent/actions/workflows/database.yml/badge.svg?branch=main)](https://github.com/AmitaKashid/sec-10k-agent/actions/workflows/database.yml)
[![Frontend checks](https://github.com/AmitaKashid/sec-10k-agent/actions/workflows/frontend.yml/badge.svg?branch=main)](https://github.com/AmitaKashid/sec-10k-agent/actions/workflows/frontend.yml)

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](backend/pyproject.toml)
[![Next.js](https://img.shields.io/badge/Next.js-16.4.3-000000?logo=nextdotjs&logoColor=white)](frontend/package.json)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white)](docker-compose.yml)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-168B77)](backend/app/agent/graph.py)

**Maintained and extended by [Amita Kashid](https://github.com/AmitaKashid).**

[Live application](https://sec-10k-agent-three.vercel.app/) · [API health](https://sec-10k-backend.onrender.com/health) · [Architecture](#architecture) · [Run locally](#run-locally) · [Tests](#tests-and-ci) · [Limitations](#scope-and-limitations)

SEC Analyst is a deployed financial-research application for asking questions about US public companies' SEC 10-K filings. Numerical answers use structured XBRL facts. Qualitative answers use retrieved filing passages, with citations and links back to SEC EDGAR. Signed-in users can ask follow-ups within a persistent conversation.

The application separates **finding evidence**, **constructing an answer**, and **checking its citations**. When the required filing, fact, or supporting passage is unavailable, it returns a structured refusal rather than treating model knowledge as evidence.

> **Project status:** deployed on Vercel, Render, and Neon; persistent conversation memory and real-time Langfuse tracing verified. The latest reported local run passed **376 selected automated tests**. Broad live answer-quality evaluation remains pending.

## Try it

Open the [live application](https://sec-10k-agent-three.vercel.app/) and start with:

| Ask | What it demonstrates |
|---|---|
| “What was Microsoft's total revenue in fiscal year 2025?” | An exact numerical lookup with an XBRL source |
| “What did Apple disclose about supply-chain risks in fiscal year 2024?” | Filing retrieval and cited source excerpts |
| “And in fiscal year 2024?” after a Microsoft revenue question | Company and concept resolution from conversation history; sign-in required for persistent memory |
| “Compare Apple and Microsoft's revenue growth.” | An explicit scope boundary: cross-company and growth calculations are unsupported |

Anonymous access is restricted to the configured preview tickers: **AAPL, MSFT, GOOGL, AMZN, and NVDA**. Sign in to use persistent conversation memory. **New thread** clears the visible conversation and creates a new thread identifier.

The application answers from **ingested filings**. A ticker appearing in the configured universe does not mean every filing or fiscal year is loaded.

## What this version adds

This repository extends the upstream implementation with changes to providers, deployment, memory, observability, and verification:

- **Gemini embeddings and Groq inference:** a provider configuration that does not require OpenAI or Anthropic credentials. Gemini vectors match the existing **1,536-dimensional** database schema.
- **Bounded Groq output handling:** strict structured planning, evidence selection by source identifiers, and controlled handling of invalid, incomplete, or quota-limited responses.
- **Persistent, user-scoped memory:** PostgreSQL checkpoints use a storage key derived from both the authenticated user and public thread ID. Another user cannot select the same checkpoint merely by reusing a client thread ID.
- **Langfuse v4 tracing:** request traces, child node spans, conversation sessions, and configurable input/output capture, with tracing failures kept separate from request failures.
- **A deployed application:** Next.js 16 / React 19 on Vercel, a Dockerized FastAPI backend on Render, and PostgreSQL with pgvector on Neon.
- **Broader automated verification:** restored and adapted tests, an isolated database runner, Windows-compatible checkpoint testing, and CI that selects complete test suites.

## How answers are grounded

### Numerical questions

For a supported annual metric, the agent resolves the company and fiscal year, retrieves an XBRL fact, and formats its value. In the configured Groq path, formatting a single numerical fact does **not** require an additional answer-generation call.

The citation gate parses the claim's number and compares it with the source's `Decimal` value using **exact equality**. A rounded or mismatched value fails that check.

<details>
<summary>Supported numerical concepts</summary>

| Concept key | Meaning |
|---|---|
| `revenue` | Revenue |
| `net_income` | Net income or loss |
| `rnd` | Research and development expense |
| `total_assets` | Total assets |
| `total_debt` | Long-term debt under the current XBRL mapping |
| `cash` | Cash and cash equivalents |
| `eps_diluted` | Diluted earnings per share |
| `shares_out` | Common shares outstanding |
| `opex` | Operating expenses |
| `gross_profit` | Gross profit |

The whitelist is defined in [`backend/app/data/xbrl.py`](backend/app/data/xbrl.py). Availability depends on the facts loaded for the company and year.

</details>

### Qualitative questions

The agent searches the supported filing sections:

- **Item 1A:** Risk Factors
- **Item 7:** Management's Discussion and Analysis
- **Item 8:** Financial Statements and Supplementary Data

Retrieval combines pgvector cosine search with PostgreSQL full-text search, fused using **Reciprocal Rank Fusion**. Compound questions can produce additional search queries. An explicit fiscal year restricts retrieval to the resolved filing, and the merged evidence set is capped before validation.

In the configured Groq path, the model selects evidence identifiers; Python copies the selected source wording into claims. Citation checks then validate source alignment and assemble numbered citation markers.

**Citation support is not the same as completeness or relevance.** A passage can be genuinely present in a filing while still answering only part of a question.

## Architecture

### Ingestion and storage

Two data paths feed the same PostgreSQL database:

| Path | Processing | Stored evidence |
|---|---|---|
| Filing prose | SEC EDGAR HTML → section extraction with regex fallback → overlapping token chunks → Gemini embeddings | Filing metadata, text, offsets, section labels, and vectors |
| Numerical facts | SEC company-facts JSON → annual US-GAAP whitelist extraction | Decimal values, units, fiscal years, and accession numbers |

The configured universe contains 50 tickers. The **29 October 2026 deployed-data snapshot** contained 5 companies, 15 filings, 1,358 embedded chunks, and 343 XBRL facts. Those are deployment snapshot counts, not a claim of complete universe coverage.

### Request workflow

```mermaid
flowchart TD
    P["Plan: classify question and resolve follow-up"] -->|Filing question| A["Anonymous access gate"]
    P -->|Greeting or capability question| M["Concierge"]
    P -->|Unsupported request| X["Refusal composer"]
    A --> F["Resolve company and filing"]
    F --> L["Locate required source"]
    L --> R["Retrieve XBRL fact or filing passages"]
    R --> V["Check answer cache; construct claims"]
    V -->|Verified cached answer| E["Return response"]
    V -->|New claims| C["Check numbers and citations"]
    C -->|Accepted| E
    C -->|Rejected| X
    M --> E
    X --> E
```

Source, access, and scope gates can also route to refusal. The graph implementation is in [`backend/app/agent/graph.py`](backend/app/agent/graph.py).

### Deployment

| Component | Technology | Hosting |
|---|---|---|
| Browser application | Next.js 16.4.3, React 19, TypeScript | Vercel |
| API and agent | Python 3.12, FastAPI, LangGraph | Render |
| Evidence and checkpoints | PostgreSQL, pgvector, SQLAlchemy, Alembic, Psycopg | Neon |
| Chat inference | Groq-hosted `openai/gpt-oss-20b` and `openai/gpt-oss-120b` | Groq |
| Embeddings | `gemini-embedding-2`, 1,536 output dimensions | Google |
| Authentication | Clerk sessions and backend JWT verification | Clerk |
| Observability | Langfuse v4 traces and sessions | Langfuse Cloud |

Provider names are model identifiers; `openai/gpt-oss-*` requests in this configuration are served through **Groq**.

## Memory, tracing, and streaming

**Conversation memory.** Signed-in requests can resume a PostgreSQL-backed LangGraph thread. History stores compact prior-turn summaries for follow-up resolution. Retrieved chunks, claims, and numerical facts are reset between turns so earlier evidence is not silently reused as new grounding. Memory becomes stateless if checkpoint initialization is unavailable; anonymous requests remain stateless.

**User isolation.** The public thread ID is combined with the authenticated user ID to derive the checkpoint key. Separate users and new threads are tested for isolation. Configurable checkpoint cleanup uses `CHECKPOINT_TTL_DAYS`; the current default is seven days.

**Tracing.** Each `/ask` request creates a root trace and graph-node spans. Conversation turns share a Langfuse session. `LANGFUSE_CAPTURE_IO` controls question/answer capture; `EXPOSE_TRACE_URLS` separately controls whether trace links reach API clients.

**Streaming.** FastAPI supports Server-Sent Events, and the Next.js server route forwards them to the browser. Verified answers are released after citation checking; the UI animates that final text. Refusal and concierge responses can stream token events. This is not a claim that unverified answer tokens are streamed directly to users.

**Caching.** The answer-cache key incorporates the effective question, classification, company/year, source identifiers or values, and model configuration. Previously verified cache entries can bypass answer construction and citation checking.

## Run locally

Tested with **Python 3.12, Node.js 24, uv, and Docker Desktop / Compose**. Docker is needed for the local PostgreSQL service.

### 1. Clone and configure

```bash
git clone https://github.com/AmitaKashid/sec-10k-agent.git
cd sec-10k-agent
```

Copy the environment templates:

<details open>
<summary>Windows CMD</summary>

```bat
copy .env.example .env
copy frontend\.env.example frontend\.env.local
```

</details>

<details>
<summary>macOS / Linux</summary>

```bash
cp .env.example .env
cp frontend/.env.example frontend/.env.local
```

</details>

Fill the private files before starting the application:

| File | Configure |
|---|---|
| Root `.env` | Your real `SEC_USER_AGENT` contact email, `GROQ_API_KEY`, and `GEMINI_API_KEY`; retain the Gemini/Groq model settings in the template |
| Root `.env` | `CLERK_ISSUER`, `CLERK_JWKS_URL`, and `CLERK_AUTHORIZED_PARTIES` for the same Clerk application used by the frontend |
| `frontend/.env.local` | `FASTAPI_URL=http://127.0.0.1:8000`, Clerk publishable/secret keys, and the matching backend `API_KEY` |

For local Clerk authentication, include `http://localhost:3000` in the backend's authorized-party configuration. Keep secrets out of Git and never expose backend secrets through `NEXT_PUBLIC_` variables. The `ci-tests@example.com` placeholder is for offline tests, not real SEC ingestion.

### 2. Start PostgreSQL, migrate, and ingest the demo data

From the repository root:

```bash
docker compose up -d postgres
cd backend
uv sync --frozen --extra dev
uv run alembic upgrade head
uv run python -m scripts.seed_dev --concurrency 1
```

Seeding downloads real filings and requests embeddings. Duration and quota requirements vary; it is not part of the offline test suite. To ingest the configured universe later, use `uv run python -m scripts.ingest_universe --concurrency 1` from `backend/`.

### 3. Start the API

From `backend/`:

```bash
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 4. Start the frontend in another terminal

From `frontend/`:

```bash
npm ci
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Check the backend at [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health).

<details>
<summary>Enable persistent memory locally, including on Windows</summary>

The example environment starts with checkpointing disabled. To enable it, set `LANGGRAPH_CHECKPOINTER_ENABLED=true`, configure Clerk, and use a signed-in conversation.

On Windows, the Linux backend container avoids Psycopg's native Windows event-loop restriction. Stop any API already using port 8000, then run these commands from the repository root while the PostgreSQL container is running:

```bash
docker build -t sec-10k-backend ./backend
docker run --rm -p 8000:8000 --env-file .env -e DATABASE_URL=postgresql+asyncpg://sec_agent:dev_only@host.docker.internal:5433/sec_agent -e LANGGRAPH_CHECKPOINTER_ENABLED=true sec-10k-backend
```

The `dev_only` credentials shown here are the local Compose defaults. The application container reaches that local database through Docker Desktop's host address. Do not use these development credentials for hosted databases.

</details>

## API

| Method | Endpoint | Behavior |
|---|---|---|
| `POST` | `/ask` | SSE by default; JSON with `Accept: application/json`. Accepts a question, optional thread ID, and allowlisted model overrides |
| `GET` | `/companies` | Ingested companies and coverage metadata; signed-in users or the administrative API-key path |
| `GET` | `/companies/stats` | Corpus statistics with the same access requirements |
| `GET` | `/health` | Database connectivity and graph availability |

### Example request

Windows CMD:

```bat
curl.exe -X POST "https://sec-10k-backend.onrender.com/ask" -H "Content-Type: application/json" -H "Accept: application/json" -d "{\"question\":\"What was Microsoft's total revenue in fiscal year 2025?\"}"
```

Excerpt from a verified smoke-test response:

```json
{
  "question_type": "A_numerical",
  "answer": "MSFT's revenue was $281,724,000,000 [1].",
  "verified": true,
  "refused": false,
  "ticker": "MSFT",
  "fiscal_year": 2025,
  "thread_id": null
}
```

The complete response also includes a request ID and structured claims with source references. For persistent follow-ups, use a valid Clerk bearer token and reuse the same `thread_id`. Sending a thread ID alone does not authenticate a request or enable anonymous memory.

Unsupported domain questions normally return a successful HTTP response with `refused=true` and structured refusal fields. Authentication, input-size, and usage-limit failures use HTTP error statuses such as **401, 403, 413, and 429**.

## Tests and CI

**Latest reported local run — 30 October 2026:**

| Suite | Result | What it checks |
|---|---|---|
| Unit and evaluation-logic tests | **347 passed** | Authentication, limits, usage accounting, parsing, chunking, ingestion orchestration, provider adapters, graph nodes, memory boundaries, citations, logging, and evaluation calculations |
| PostgreSQL integration | **29 passed** | Migrations, schema constraints, vector search, real graph/API database paths, checkpoint persistence after reconnecting, and user isolation |
| Total selected automated tests | **376 passed** | Software behavior under the selected test conditions |

Database integration tests use real PostgreSQL; model and external-service calls are mocked in the standard suites. Two real-filing corpus checks are excluded from the unit run, and two live-service checks are excluded from the database run. **Test counts are not an answer-accuracy benchmark.**

Run from `backend/`:

```bash
uv run --extra dev python scripts/run_tests.py --suite unit
uv run --extra dev python scripts/run_tests.py --suite database
```

The database runner checks that the target is a separate database with a name ending in `_test`, requires it to be reachable, and rejects skipped database checks. It configures the compatible selector event loop on Windows.

Run frontend checks from `frontend/`:

```bash
npm run typecheck
npm run lint
npm run build
npm audit
```

GitHub Actions runs the complete offline backend/evaluation suite, a separate database job using PostgreSQL with pgvector, and frontend TypeScript/ESLint checks. Dependabot is configured for scheduled frontend and GitHub Actions dependency-update PRs.

<details>
<summary>Opt-in live and downloaded-filing checks</summary>

From `backend/`:

```bash
uv run --extra dev python scripts/run_tests.py --suite live
uv run --extra dev python scripts/run_tests.py --suite corpus
```

Live checks require private Groq/Gemini keys, a valid SEC contact email, and the isolated test database. They make real external requests and consume provider quota. Corpus checks require real filing HTML fixtures under `backend/tests/fixtures/10ks/`. Neither category runs automatically in the standard CI suites.

</details>

## Evaluation evidence

A historical qualitative run on **27 October 2026** evaluated six golden questions across four companies:

| Measurement | Reported result |
|---|---:|
| Qualitative questions evaluated | 6 |
| Citation accuracy within that sample | 100% |
| Claims assessed by the LLM judge | 30 |
| Claims judged unsupported | 0 / 30 |
| Reported median graph latency | 10,308 ms |

This was a small, in-process evaluation, not a hosted latency benchmark or a fresh evaluation of every subsequent revision. The judge used the same model family as the configured validation stage, so its assessment was not independent. The harness also selects the upper middle observation for an even-sized median.

**There were no numerical or adversarial examples in that run.** It therefore establishes neither numerical accuracy nor adversarial refusal accuracy. Broader live numerical, multi-turn, and adversarial evaluation is deferred; these sample results must not be generalized to all questions.

The datasets, judge, and metric implementations are available in [`evals/`](evals/).

## Deployment and configuration

| Service | Main settings |
|---|---|
| Vercel | Project root `frontend`; backend URL and private Clerk/API settings |
| Render | Docker build context `backend`; `/health` health-check path; provider, database, Clerk, and tracing variables |
| Neon | PostgreSQL with the `vector` extension and Alembic migrations |

The backend Docker image installs CA certificates, runs as a non-root user, applies Alembic migrations at startup, and binds Uvicorn to the platform's `PORT`.

For the Linux container's verified Neon TLS configuration:

```dotenv
DATABASE_URL=postgresql+asyncpg://USER:PASSWORD@HOST/neondb?ssl=verify-full
PGSSLROOTCERT=/etc/ssl/certs/ca-certificates.crt
```

Configure optional memory and tracing privately in the backend environment:

```dotenv
LANGGRAPH_CHECKPOINTER_ENABLED=true
CHECKPOINT_TTL_DAYS=7
LANGFUSE_PUBLIC_KEY=your-project-public-key
LANGFUSE_SECRET_KEY=your-project-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_CAPTURE_IO=true
EXPOSE_TRACE_URLS=false
```

`LANGFUSE_CAPTURE_IO=true` sends question/answer text to the configured tracing service. Set it to `false` when content capture is not wanted. Restrict `CLERK_AUTHORIZED_PARTIES` to the frontend origins you intend to accept. Configure usage limits for the hosting and provider quotas you actually have; the checked-in development template disables several limits.

Configuration is defined in [`backend/app/settings.py`](backend/app/settings.py), with a Gemini/Groq example in [`.env.example`](.env.example).

## Repository guide

| Location | Responsibility |
|---|---|
| [`backend/app/agent/`](backend/app/agent/) | Graph, shared state, prompts, Groq runtime, retrieval tools, memory keys, and node implementations |
| [`backend/app/data/`](backend/app/data/) | EDGAR access, parsing, chunking, embeddings, XBRL extraction, ingestion, and database models |
| [`backend/app/api/`](backend/app/api/) | Request/response schemas, identity, usage limits, and API routes |
| [`backend/app/observability/`](backend/app/observability/) | Langfuse integration and structured logging |
| [`backend/alembic/`](backend/alembic/) | Database migrations |
| [`backend/scripts/`](backend/scripts/) | Ingestion, checkpoint cleanup, smoke checks, and the test runner |
| [`backend/tests/`](backend/tests/) | Unit and integration suites |
| [`frontend/`](frontend/) | Chat interface, company browser, source cards, Clerk integration, and server-side API proxies |
| [`evals/`](evals/) | Evaluation datasets, runner, judge, metrics, and tests |
| [`.github/workflows/`](.github/workflows/) | Backend, database, and frontend CI |

## Scope and limitations

- **Bounded research scope:** US SEC 10-K filings, the supported annual concepts, and Items 1A / 7 / 8. No real-time prices, earnings-call analysis, cross-company comparison, or year-over-year calculations.
- **Corpus availability:** answers depend on the filings actually ingested. The default seed targets five companies and recent fiscal years; older years and the full configured universe are not automatically present.
- **Limited quality evaluation:** six qualitative questions do not establish general accuracy, complete answers, robust refusals, or production latency.
- **Evidence checks have limits:** exact numbers and source-aligned quotations reduce specific failure modes; they do not prove every interpretation or evidence selection is correct.
- **Hosting and provider constraints:** idle-service cold starts and provider quotas can add delay or interrupt requests. No fixed latency or always-free operation is promised.
- **Memory is conditional:** persistent follow-ups require sign-in and an active checkpointer. They are conversation continuity, not a source of financial evidence.
- **Usage limits are per process:** the in-memory request/token windows are not distributed quota accounting. The optional database-backed anonymous lifetime cap is separately configurable.
- **Browser verification is manual:** sign-in, questions, follow-ups, and new-thread behavior were smoke-tested; the automated suite does not include browser end-to-end tests.

Use the linked filing and its reporting context when interpreting an answer. This is a research/demo application, not financial advice.

## Maintainer and provenance

**[Amita Kashid](https://github.com/AmitaKashid)** maintains this deployed version and its provider integration, memory isolation, tracing migration, frontend upgrades, deployment configuration, and expanded test/CI coverage.

The project builds on the architecture and base implementation in [Dhiraj Podar's SEC 10-K agent](https://github.com/dhirajpodar/sec-10k-agent). Credit for that upstream work remains with its author; the implementation changes described above are maintained in this repository.
