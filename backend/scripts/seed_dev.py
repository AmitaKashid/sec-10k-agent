"""Seed the dev DB with 5 mega-cap tech tickers.

Usage:
    uv run python -m scripts.seed_dev [--retry-failed]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from app.data.ingestion import main

BACKEND_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = BACKEND_ROOT / "config" / "seed_dev.csv"


def cli() -> None:
    p = argparse.ArgumentParser(description="Seed dev DB with 5 tickers")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--concurrency", type=int, default=5)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    asyncio.run(main(CSV_PATH, concurrency=args.concurrency, retry_failed=args.retry_failed))


if __name__ == "__main__":
    cli()
