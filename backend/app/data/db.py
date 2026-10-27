"""Async engine + session factory. Pure plumbing — see docs/lld.md §LLD 5."""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.settings import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    # Detect dead connections (Postgres restarts, managed-PG idle cuts,
    # PgBouncer drops) at checkout instead of failing the request.
    pool_pre_ping=True,
    # Force a refresh before managed providers' idle TCP timeout (often
    # ~30 min) so we never hand out a connection the server has already
    # forgotten about.
    pool_recycle=1800,
    echo=settings.DB_ECHO,
)

async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: commit on success, rollback on exception."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
