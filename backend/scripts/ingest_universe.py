"""Ingest the full 50-company universe from config/tickers.csv.

Usage:
    uv run python -m scripts.ingest_universe [--retry-failed] [--concurrency N]
    uv run python -m scripts.ingest_universe --force-reembed   # re-embed with contextual prefix
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from app.data.ingestion import main

BACKEND_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = BACKEND_ROOT / "config" / "tickers.csv"


def cli() -> None:
    p = argparse.ArgumentParser(description="Ingest full company universe")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument(
        "--force-reembed",
        action="store_true",
        help="Reset all 'embedded' filings to 'pending' so chunks are re-embedded "
        "with the new contextual prefix. Use after changing the embedding scheme.",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    async def _run() -> None:
        if args.force_reembed:
            await _reset_embedded_filings()
        await main(CSV_PATH, concurrency=args.concurrency, retry_failed=args.retry_failed)

    asyncio.run(_run())


async def _reset_embedded_filings() -> None:
    """Reset ingestion_status='embedded' → 'pending' so ingest re-embeds all chunks."""
    from sqlalchemy import update

    from app.data.db import async_session_factory
    from app.data.models import Filing

    logging.getLogger(__name__).info("--force-reembed: resetting all embedded filings to pending")
    async with async_session_factory() as session:
        await session.execute(
            update(Filing)
            .where(Filing.ingestion_status == "embedded")
            .values(ingestion_status="pending")
        )
        await session.commit()
    logging.getLogger(__name__).info("reset complete — re-ingestion will re-embed all chunks")


if __name__ == "__main__":
    cli()
