# Deployment



## Services



| Component | Hosting | Configuration |

|---|---|---|

| Frontend | Vercel | Next.js; root directory `frontend` |

| Backend | Render Free | Docker; root directory `backend` |

| Database | Neon | PostgreSQL with pgvector |



Frontend: https://sec-10k-agent-three.vercel.app  

Backend: https://sec-10k-backend.onrender.com



## Backend environment



Configure secrets privately in Render:

- DATABASE_URL

- GROQ_API_KEY

- GEMINI_API_KEY

- API_KEY

- SEC_USER_AGENT



Database URL format:



```text

postgresql+asyncpg://USER:PASSWORD@HOST/neondb?ssl=verify-full

```



Additional settings:



```dotenv

PGSSLROOTCERT=/etc/ssl/certs/ca-certificates.crt

EMBEDDING_PROVIDER=gemini

GEMINI_EMBEDDING_MODEL=gemini-embedding-2

PLAN_MODEL=openai/gpt-oss-20b

VALIDATE_MODEL=openai/gpt-oss-120b

REFUSAL_MODEL=openai/gpt-oss-20b

ALLOWED_MODELS=openai/gpt-oss-20b,openai/gpt-oss-120b

LANGGRAPH_CHECKPOINTER_ENABLED=false

EXPOSE_TRACE_URLS=false

ANON_LIFETIME_ENABLED=false

ASK_TOKEN_LIMITER_ENABLED=true

ASK_ANON_REQS_PER_MIN=3

ASK_ANON_TOKENS_PER_MIN=15000

ASK_USER_REQS_PER_MIN=10

ASK_USER_TOKENS_PER_MIN=50000

```



Configure CLERK_ISSUER and CLERK_JWKS_URL for the same Clerk

application used by the frontend. Set CLERK_AUTHORIZED_PARTIES

to the permitted frontend origins, including the deployed Vercel URL.



## Frontend environment



Configure privately in Vercel:



- FASTAPI_URL=https://sec-10k-backend.onrender.com

- API_KEY: same value as Render

- NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY

- CLERK_SECRET_KEY



Never expose API_KEY or CLERK_SECRET_KEY through NEXT_PUBLIC_ variables.



## Database and startup



Schema version at deployment: 0005.



Initial data transferred from the local database:

- 5 companies

- 15 filings

- 1,358 chunks with embeddings

- 343 XBRL facts



The Docker startup command runs Alembic migrations, then Uvicorn.

The backend listens on the hosting platform's PORT.



## Verification



- GET /health returned database and graph status "ok".

- Microsoft FY2025 revenue returned $281,724,000,000 with a verified source.

- Apple FY2024 supply-chain question rendered cited filing excerpts.

- Signed-in identity and EDGAR links were manually checked.



These checks are smoke tests, not a full production certification.



## Limitations



- Render Free sleeps when idle, causing cold-start delays.

- Provider free-tier quotas can interrupt requests.

- Graph checkpointing is disabled; persistent graph conversation state

  is not enabled.

- Request limits do not guarantee provider quota availability.

- Anonymous lifetime limits remain disabled pending proxy-IP verification.

- CI runs mocked backend/evaluation tests and frontend type/lint checks;

  it does not run live evaluations or browser tests.

