# Multi-stage build keeps the runtime image small. Stage 1 installs deps
# with uv; stage 2 copies the resolved venv into a slim runtime.
FROM python:3.12-slim AS builder

# uv: fast Python package manager. Pinned to a known-good version so
# rebuilds are deterministic.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install build deps for psycopg/asyncpg native wheels (most ship
# prebuilt for linux/amd64, but build-essential is cheap insurance).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Resolve dependencies first (cached layer if pyproject/uv.lock unchanged).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Now copy source and install the project itself.
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
COPY scripts ./scripts
COPY config ./config
RUN uv sync --frozen --no-dev


# ---- runtime ----
FROM python:3.12-slim AS runtime

# libpq runtime is needed by psycopg even if the wheel is prebuilt
# (psycopg-c links against it dynamically).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

WORKDIR /app

# Bring over the resolved venv + source from the builder.
COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app

# Railway sets $PORT at runtime. We run migrations on each boot — safe
# because Alembic is idempotent and the demo runs one instance.
CMD alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
