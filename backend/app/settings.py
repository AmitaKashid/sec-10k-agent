from pathlib import Path
from typing import Annotated

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_DEFAULT_API_KEY = "dev-local-key"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATABASE_URL: str = "postgresql+asyncpg://sec_agent:dev_only@localhost:5432/sec_agent"
    # Integration tests connect HERE (never DATABASE_URL). When unset
    # (the common case), it's auto-derived by suffixing the dev DB name
    # with `_test` (see `_derive_test_db_url`). Override only if your
    # test DB is on a different host/credentials. The conftest contains
    # a hard tripwire that aborts the test run if this ever equals
    # DATABASE_URL — wiping the dev DB via tests must be impossible.
    TEST_DATABASE_URL: str | None = None
    DB_ECHO: bool = False

    SEC_USER_AGENT: str
    DATA_CACHE_DIR: Path = Path(__file__).parent.parent / "data_cache"

    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    EMBEDDING_PROVIDER: str = "openai"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-2"
    ANTHROPIC_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # OpenAI; Groq doesn't offer embeddings
    PLAN_MODEL: str = "gpt-5.4-mini"  # cheap tier — Plan/Locate
    VALIDATE_MODEL: str = "gpt-5.1"  # general/frontier tier — Validate
    REFUSAL_MODEL: str = "gpt-5.4-mini"  # cheap tier — refusal_composer
    ALLOWED_MODELS: Annotated[list[str], NoDecode] = [
        # OpenAI (default tiers)
        "gpt-5.4",  # judge / strongest
        "gpt-5.1",  # general / Validate
        "gpt-5.4-mini",  # cheap / Plan / Locate
        "gpt-4o",
        "gpt-4o-mini",
        # Groq (dev fallback, free tier)
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        # Anthropic
        "claude-haiku-4-5",
        "claude-sonnet-4-6",
    ]

    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    EXPOSE_TRACE_URLS: bool = True
    # Operator kill-switch for sending user-content (the /ask question +
    # final answer) to Langfuse. When False, traces + per-node spans +
    # session_id still flow (so latency dashboards keep working) but the
    # Input/Output panels stay empty and no user text leaves the
    # process. Default True preserves the dev/demo experience; flip to
    # False in prod (or any environment with privacy/compliance
    # requirements). Independent of EXPOSE_TRACE_URLS, which only gates
    # returning the trace URL to clients.
    LANGFUSE_CAPTURE_IO: bool = True

    API_KEY: str = _DEFAULT_API_KEY
    STRICT_API_KEY: bool = False
    RATE_LIMIT_PER_MINUTE: int = 60
    CORS_ORIGINS: Annotated[list[str], NoDecode] = ["http://localhost:3000"]
    # Behind a trusted proxy (Railway, Vercel, Cloudflare, etc.) the
    # raw socket peer is the proxy itself, so per-IP rate limiting
    # collapses every anon user onto one bucket. Set True in
    # production to read the leftmost entry of `X-Forwarded-For` for
    # anon identity keying. Only enable when ALL traffic reaches the
    # app via a proxy that strips/replaces inbound XFF — otherwise a
    # client can spoof the header to evade the cap.
    RATE_LIMIT_TRUST_XFF: bool = False

    # Clerk JWT verification. See docs/auth_and_rate_limit_plan.md.
    # Empty in dev means /ask runs anonymous-tier only — Bearer tokens
    # presented with auth unconfigured return 401, not a silent downgrade.
    CLERK_ISSUER: str = ""
    CLERK_JWKS_URL: str = ""
    # Allowlist of origins that may request a Clerk session token (the
    # `azp` claim — "authorized party"). When non-empty, tokens whose
    # `azp` is not in this list are rejected. This is Clerk's standard
    # hardening against cross-app token reuse if your JWKS is ever
    # shared. Leave empty in dev; in prod set to the deployed frontend
    # origin(s), comma-separated.
    CLERK_AUTHORIZED_PARTIES: Annotated[list[str], NoDecode] = []

    # Tiered rate limits for /ask. Anonymous tier is keyed by IP, gated by
    # the frontend's "sign in to continue" modal. Authenticated tier is
    # keyed by Clerk user id (sub). See docs/auth_and_rate_limit_plan.md.
    ASK_TOKEN_LIMITER_ENABLED: bool = True
    ASK_ANON_REQS_PER_MIN: int = 5
    ASK_ANON_TOKENS_PER_MIN: int = 5_000
    ASK_ANON_MAX_INPUT_TOKENS: int = 1_000
    ASK_USER_REQS_PER_MIN: int = 60
    ASK_USER_TOKENS_PER_MIN: int = 50_000
    ASK_USER_MAX_INPUT_TOKENS: int = 2_000
    # Up-front reservation for output tokens at request start. Reconciled
    # against actual LLM usage when the request completes.
    ASK_EST_OUTPUT_TOKENS: int = 1_500
    # Up-front reservation for retrieved-chunk context the validate node
    # feeds to the frontier model. Without this the reservation only
    # accounts for the question + output and routinely under-charges by
    # several thousand tokens per request.
    ASK_EST_RETRIEVAL_TOKENS: int = 3_000

    # Lifetime per-IP cap on anonymous /ask calls. Persisted in Postgres
    # (anon_usage table) so it survives server restarts and the
    # frontend's sign-in modal is the only way past it. ANON_HASH_SALT
    # must be non-empty when the limiter is enabled — raw IPs are never
    # stored. See docs/anon_hardening_plan.md.
    ANON_LIFETIME_ENABLED: bool = False  # opt-in; prod env flips to true
    ANON_LIFETIME_LIMIT: int = 10
    ANON_HASH_SALT: str = ""

    # Anonymous /ask is restricted to questions about this allowlist of
    # tickers (case-insensitive). Anything else returns 403
    # anon_ticker_not_allowed with the frontend's sign-in prompt.
    # Meta/concierge questions (no resolved ticker) pass through.
    ANON_TICKER_ALLOWLIST: Annotated[list[str], NoDecode] = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
    ]

    # Multi-turn conversation memory. When True, the LangGraph
    # AsyncPostgresSaver is wired into the graph and `thread_id` from the
    # request keys persistent state across turns. Fails open to no-memory
    # if Postgres is unreachable. See docs/conversation_memory_plan.md.
    LANGGRAPH_CHECKPOINTER_ENABLED: bool = True

    # Idle-thread TTL for the LangGraph checkpointer. Threads whose latest
    # checkpoint is older than this are eligible for pruning by
    # scripts/prune_checkpoints.py. Demo default; tune as data grows.
    CHECKPOINT_TTL_DAYS: int = 7

    @field_validator(
        "ALLOWED_MODELS",
        "CORS_ORIGINS",
        "CLERK_AUTHORIZED_PARTIES",
        "ANON_TICKER_ALLOWLIST",
        mode="before",
    )
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def _ensure_asyncpg_driver(cls, v):
        """Railway/Heroku-style Postgres URLs are `postgresql://...` and
        `postgres://...` — neither carries the `+asyncpg` driver suffix
        SQLAlchemy needs. Rewrite at ingress so the same env var works
        on Railway, locally, and in CI without per-env munging."""
        if isinstance(v, str):
            if v.startswith("postgres://"):
                v = "postgresql://" + v[len("postgres://") :]
            if v.startswith("postgresql://"):
                v = "postgresql+asyncpg://" + v[len("postgresql://") :]
        return v

    @model_validator(mode="after")
    def _derive_test_db_url(self) -> "Settings":
        """If TEST_DATABASE_URL is unset, derive it from DATABASE_URL by
        suffixing the database name with `_test`. Keeps zero-config dev
        working while making it impossible for tests to share a DB with
        dev by accident. Idempotent: an already-suffixed URL is left
        alone."""
        if self.TEST_DATABASE_URL:
            return self
        url = self.DATABASE_URL
        # postgresql+asyncpg://user:pw@host:port/dbname[?params]
        scheme_body, _, query = url.partition("?")
        head, sep, dbname = scheme_body.rpartition("/")
        if not sep or not dbname:
            # Malformed URL — leave TEST_DATABASE_URL as None and let
            # the conftest tripwire surface the problem loudly.
            return self
        if dbname.endswith("_test"):
            self.TEST_DATABASE_URL = url
        else:
            suffixed = f"{head}/{dbname}_test"
            self.TEST_DATABASE_URL = f"{suffixed}?{query}" if query else suffixed
        return self

    @model_validator(mode="after")
    def _check_api_key_strength(self) -> "Settings":
        if self.STRICT_API_KEY:
            if self.API_KEY == _DEFAULT_API_KEY:
                raise ValueError(
                    "API_KEY must be changed from the default in production. "
                    "Set a random secret of at least 32 characters."
                )
            if len(self.API_KEY) < 32:
                raise ValueError(
                    "API_KEY must be at least 32 characters in production (STRICT_API_KEY=true)."
                )
        return self

    @model_validator(mode="after")
    def _check_anon_hash_salt(self) -> "Settings":
        if self.ANON_LIFETIME_ENABLED and not self.ANON_HASH_SALT:
            raise ValueError(
                "ANON_HASH_SALT must be set when ANON_LIFETIME_ENABLED=true. "
                "Generate a random secret (e.g. `openssl rand -hex 32`) and set it "
                "via env — never commit it. Disabling: set ANON_LIFETIME_ENABLED=false."
            )
        return self

    @property
    def anon_ticker_allowlist_upper(self) -> frozenset[str]:
        return frozenset(t.upper() for t in self.ANON_TICKER_ALLOWLIST)


settings = Settings()
