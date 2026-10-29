\# Deployment



\## Services



| Component | Hosting | Configuration |

|---|---|---|

| Frontend | Vercel | Next.js; root directory `frontend` |

| Backend | Render Free | Docker; root directory `backend` |

| Database | Neon | PostgreSQL with pgvector |



Frontend: https://sec-10k-agent-three.vercel.app  

Backend: https://sec-10k-backend.onrender.com



\## Backend environment



Configure secrets privately in Render:

\- DATABASE\_URL

\- GROQ\_API\_KEY

\- GEMINI\_API\_KEY

\- API\_KEY

\- SEC\_USER\_AGENT



Database URL format:



```text

postgresql+asyncpg://USER:PASSWORD@HOST/neondb?ssl=verify-full

```



Additional settings:



```dotenv

PGSSLROOTCERT=/etc/ssl/certs/ca-certificates.crt

EMBEDDING\_PROVIDER=gemini

GEMINI\_EMBEDDING\_MODEL=gemini-embedding-2

PLAN\_MODEL=openai/gpt-oss-20b

VALIDATE\_MODEL=openai/gpt-oss-120b

REFUSAL\_MODEL=openai/gpt-oss-20b

ALLOWED\_MODELS=openai/gpt-oss-20b,openai/gpt-oss-120b

LANGGRAPH\_CHECKPOINTER\_ENABLED=false

EXPOSE\_TRACE\_URLS=false

ANON\_LIFETIME\_ENABLED=false

ASK\_TOKEN\_LIMITER\_ENABLED=true

ASK\_ANON\_REQS\_PER\_MIN=3

ASK\_ANON\_TOKENS\_PER\_MIN=15000

ASK\_USER\_REQS\_PER\_MIN=10

ASK\_USER\_TOKENS\_PER\_MIN=50000

```



Configure CLERK\_ISSUER and CLERK\_JWKS\_URL for the same Clerk

application used by the frontend. Set CLERK\_AUTHORIZED\_PARTIES

to the permitted frontend origins, including the deployed Vercel URL.



\## Frontend environment



Configure privately in Vercel:



\- FASTAPI\_URL=https://sec-10k-backend.onrender.com

\- API\_KEY: same value as Render

\- NEXT\_PUBLIC\_CLERK\_PUBLISHABLE\_KEY

\- CLERK\_SECRET\_KEY



Never expose API\_KEY or CLERK\_SECRET\_KEY through NEXT\_PUBLIC\_ variables.



\## Database and startup



Schema version at deployment: 0005.



Initial data transferred from the local database:

\- 5 companies

\- 15 filings

\- 1,358 chunks with embeddings

\- 343 XBRL facts



The Docker startup command runs Alembic migrations, then Uvicorn.

The backend listens on the hosting platform's PORT.



\## Verification



\- GET /health returned database and graph status "ok".

\- Microsoft FY2025 revenue returned $281,724,000,000 with a verified source.

\- Apple FY2024 supply-chain question rendered cited filing excerpts.

\- Signed-in identity and EDGAR links were manually checked.



These checks are smoke tests, not a full production certification.



\## Limitations



\- Render Free sleeps when idle, causing cold-start delays.

\- Provider free-tier quotas can interrupt requests.

\- Graph checkpointing is disabled; persistent graph conversation state

&#x20; is not enabled.

\- Request limits do not guarantee provider quota availability.

\- Anonymous lifetime limits remain disabled pending proxy-IP verification.

\- CI runs mocked backend/evaluation tests and frontend type/lint checks;

&#x20; it does not run live evaluations or browser tests.

