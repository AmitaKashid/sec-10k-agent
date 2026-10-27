"""SEC EDGAR HTTP client. The only module allowed to call SEC.

See docs/lld.md §LLD 1 for the full contract.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime
from pathlib import Path

import httpx
from aiolimiter import AsyncLimiter
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.settings import settings

logger = logging.getLogger(__name__)


# --- Errors ---------------------------------------------------------------


class EdgarError(Exception):
    """Base for all SEC EDGAR errors."""


class EdgarNotFound(EdgarError):
    """404 — the requested resource does not exist on EDGAR."""


class EdgarAuthError(EdgarError):
    """403 — usually a missing/malformed User-Agent."""


class EdgarRateLimited(EdgarError):
    """429 after retries — caller should back off."""


class EdgarServerError(EdgarError):
    """5xx or network error after retries."""


# --- Types ----------------------------------------------------------------


class FilingMeta(BaseModel):
    accession_no: str
    form_type: str
    filing_date: date
    period_of_report: date
    primary_doc: str
    primary_doc_url: str


# --- Singletons -----------------------------------------------------------

_LIMITER = AsyncLimiter(max_rate=5, time_period=1.0)
_TICKER_MAP: dict[str, str] | None = None
_TICKER_MAP_FETCHED_AT: float = 0.0
_TICKER_MAP_TTL_SECONDS = 24 * 3600

_CACHE_DIR = Path(settings.DATA_CACHE_DIR)
_TTL_24H = 24 * 3600


def _headers() -> dict[str, str]:
    ua = settings.SEC_USER_AGENT
    if not ua or "@" not in ua:
        raise EdgarAuthError("SEC_USER_AGENT must be set to a string containing a contact email")
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}


# --- Cache helpers --------------------------------------------------------


def _cache_path(*parts: str) -> Path:
    p = _CACHE_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _is_fresh(path: Path, ttl_seconds: int | None) -> bool:
    if not path.exists():
        return False
    if ttl_seconds is None:
        return True
    return (time.time() - path.stat().st_mtime) < ttl_seconds


# --- Retry-aware GET ------------------------------------------------------


async def _request(client: httpx.AsyncClient, url: str) -> httpx.Response:
    """Issue a rate-limited GET with retries on 429/5xx/network. Raises on 404/403."""

    attempt_no = {"n": 0}

    async def attempt() -> httpx.Response:
        attempt_no["n"] += 1
        t0 = time.perf_counter()
        async with _LIMITER:
            response = await client.get(url, headers=_headers(), timeout=30.0)
        dt_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            "edgar.http",
            extra={
                "url": url,
                "status": response.status_code,
                "attempt": attempt_no["n"],
                "duration_ms": dt_ms,
                "bytes": len(response.content),
            },
        )

        if response.status_code == 404:
            raise EdgarNotFound(f"404 {url}")
        if response.status_code == 403:
            raise EdgarAuthError(f"403 {url} — check SEC_USER_AGENT")
        if response.status_code == 429:
            raise EdgarRateLimited(f"429 {url}")
        if 500 <= response.status_code < 600:
            raise EdgarServerError(f"{response.status_code} {url}")
        response.raise_for_status()
        return response

    try:
        async for retry_state in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential_jitter(initial=1.0, max=8.0),
            retry=retry_if_exception_type((EdgarRateLimited, EdgarServerError, httpx.RequestError)),
            reraise=True,
        ):
            with retry_state:
                return await attempt()
    except RetryError as e:
        logger.warning("edgar.http.giveup", extra={"url": url, "attempts": attempt_no["n"]})
        raise EdgarServerError(str(e)) from e
    raise EdgarServerError("retry loop exited without result")  # pragma: no cover


# --- Public API -----------------------------------------------------------


async def warmup() -> None:
    """Preload the ticker map. Called from FastAPI lifespan startup."""
    await _load_ticker_map()


async def _load_ticker_map() -> dict[str, str]:
    global _TICKER_MAP, _TICKER_MAP_FETCHED_AT
    if _TICKER_MAP is not None and (time.time() - _TICKER_MAP_FETCHED_AT) < _TICKER_MAP_TTL_SECONDS:
        return _TICKER_MAP

    cache_file = _cache_path("ticker_map.json")
    if _is_fresh(cache_file, _TTL_24H):
        logger.info("edgar.cache.hit", extra={"resource": "ticker_map"})
        data = json.loads(cache_file.read_text())
    else:
        logger.info("edgar.cache.miss", extra={"resource": "ticker_map"})
        async with httpx.AsyncClient() as client:
            response = await _request(client, "https://www.sec.gov/files/company_tickers.json")
        data = response.json()
        cache_file.write_text(json.dumps(data))

    # Source format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    mapping = {row["ticker"].upper(): str(row["cik_str"]).zfill(10) for row in data.values()}
    _TICKER_MAP = mapping
    _TICKER_MAP_FETCHED_AT = time.time()
    return mapping


async def get_cik_for_ticker(ticker: str) -> str | None:
    mapping = await _load_ticker_map()
    return mapping.get(ticker.upper())


async def get_submissions(cik: str, form_type: str = "10-K") -> list[FilingMeta]:
    cik = cik.zfill(10)
    cache_file = _cache_path("submissions", f"{cik}.json")

    if _is_fresh(cache_file, _TTL_24H):
        logger.info("edgar.cache.hit", extra={"resource": "submissions", "cik": cik})
        data = json.loads(cache_file.read_text())
    else:
        logger.info("edgar.cache.miss", extra={"resource": "submissions", "cik": cik})
        async with httpx.AsyncClient() as client:
            response = await _request(client, f"https://data.sec.gov/submissions/CIK{cik}.json")
        data = response.json()
        cache_file.write_text(json.dumps(data))

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_nos = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    periods = recent.get("reportDate", [])
    primary_docs = recent.get("primaryDocument", [])

    out: list[FilingMeta] = []
    for i, form in enumerate(forms):
        if form != form_type:
            continue
        accession = accession_nos[i]
        accession_no_dashes = accession.replace("-", "")
        primary_doc = primary_docs[i]
        out.append(
            FilingMeta(
                accession_no=accession,
                form_type=form,
                filing_date=datetime.strptime(filing_dates[i], "%Y-%m-%d").date(),
                period_of_report=datetime.strptime(periods[i], "%Y-%m-%d").date(),
                primary_doc=primary_doc,
                primary_doc_url=(
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{int(cik)}/{accession_no_dashes}/{primary_doc}"
                ),
            )
        )
    return out


async def get_filing_html(cik: str, accession_no: str, primary_doc: str) -> bytes:
    cik = cik.zfill(10)
    accession_no_dashes = accession_no.replace("-", "")
    cache_file = _cache_path("filings", cik, accession_no, primary_doc)

    if cache_file.exists():
        logger.info(
            "edgar.cache.hit",
            extra={"resource": "filing_html", "cik": cik, "accession_no": accession_no},
        )
        return cache_file.read_bytes()

    logger.info(
        "edgar.cache.miss",
        extra={"resource": "filing_html", "cik": cik, "accession_no": accession_no},
    )
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes}/{primary_doc}"
    async with httpx.AsyncClient() as client:
        response = await _request(client, url)
    cache_file.write_bytes(response.content)
    return response.content


async def get_company_facts(cik: str) -> dict:
    cik = cik.zfill(10)
    cache_file = _cache_path("facts", f"{cik}.json")

    if _is_fresh(cache_file, _TTL_24H):
        logger.info("edgar.cache.hit", extra={"resource": "company_facts", "cik": cik})
        return json.loads(cache_file.read_text())

    logger.info("edgar.cache.miss", extra={"resource": "company_facts", "cik": cik})
    async with httpx.AsyncClient() as client:
        response = await _request(
            client, f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        )
    data = response.json()
    cache_file.write_text(json.dumps(data))
    return data


__all__ = [
    "EdgarError",
    "EdgarNotFound",
    "EdgarAuthError",
    "EdgarRateLimited",
    "EdgarServerError",
    "FilingMeta",
    "warmup",
    "get_cik_for_ticker",
    "get_submissions",
    "get_filing_html",
    "get_company_facts",
]
