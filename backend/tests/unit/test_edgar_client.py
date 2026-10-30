"""Unit tests for app.data.edgar_client (LLD 1 test plan).

All HTTP calls are mocked with respx. No live SEC traffic.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
import tenacity

from app.data import edgar_client
from app.data.edgar_client import (
    EdgarAuthError,
    EdgarNotFound,
    EdgarRateLimited,
    EdgarServerError,
    FilingMeta,
)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK0000320193.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Redirect cache to a temp dir and reset module-level singletons."""
    from aiolimiter import AsyncLimiter

    monkeypatch.setattr(edgar_client, "_LIMITER", AsyncLimiter(max_rate=5, time_period=1.0))
    monkeypatch.setattr(edgar_client, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(edgar_client, "_TICKER_MAP", None)
    monkeypatch.setattr(edgar_client, "_TICKER_MAP_FETCHED_AT", 0.0)
    # Disable tenacity's exponential backoff so retry tests run fast.
    monkeypatch.setattr(edgar_client, "wait_exponential_jitter", lambda **_: tenacity.wait_none())


def _ticker_payload() -> dict:
    return {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
    }


def _submissions_payload() -> dict:
    return {
        "filings": {
            "recent": {
                "form": ["10-K", "10-Q", "10-K"],
                "accessionNumber": [
                    "0000320193-24-000123",
                    "0000320193-24-000050",
                    "0000320193-23-000106",
                ],
                "filingDate": ["2024-11-01", "2024-08-01", "2023-11-03"],
                "reportDate": ["2024-09-28", "2024-06-29", "2023-09-30"],
                "primaryDocument": ["aapl-20240928.htm", "aapl-q3.htm", "aapl-20230930.htm"],
            }
        }
    }


# --- get_cik_for_ticker ---------------------------------------------------


@respx.mock
async def test_get_cik_for_ticker_hit():
    respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=_ticker_payload()))
    cik = await edgar_client.get_cik_for_ticker("AAPL")
    assert cik == "0000320193"


@respx.mock
async def test_get_cik_for_ticker_lowercase_normalized():
    respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=_ticker_payload()))
    assert await edgar_client.get_cik_for_ticker("aapl") == "0000320193"


@respx.mock
async def test_get_cik_for_ticker_miss_returns_none():
    respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=_ticker_payload()))
    assert await edgar_client.get_cik_for_ticker("NOPE") is None


@respx.mock
async def test_get_cik_uses_cache_on_second_call():
    route = respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=_ticker_payload()))
    await edgar_client.get_cik_for_ticker("AAPL")
    # Reset in-memory cache so the second call must hit either disk or HTTP.
    edgar_client._TICKER_MAP = None
    edgar_client._TICKER_MAP_FETCHED_AT = 0.0
    await edgar_client.get_cik_for_ticker("AAPL")
    assert route.call_count == 1  # second call served from disk cache


# --- get_submissions ------------------------------------------------------


@respx.mock
async def test_get_submissions_filters_by_form_type():
    respx.get(SUBMISSIONS_URL).mock(return_value=httpx.Response(200, json=_submissions_payload()))
    filings = await edgar_client.get_submissions("0000320193", form_type="10-K")
    assert len(filings) == 2
    assert all(isinstance(f, FilingMeta) for f in filings)
    assert filings[0].accession_no == "0000320193-24-000123"
    assert filings[0].primary_doc == "aapl-20240928.htm"
    assert (
        filings[0].primary_doc_url
        == "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    )


@respx.mock
async def test_get_submissions_zero_pads_cik():
    respx.get(SUBMISSIONS_URL).mock(return_value=httpx.Response(200, json=_submissions_payload()))
    # Pass non-padded CIK; client must pad to 10 digits.
    filings = await edgar_client.get_submissions("320193")
    assert len(filings) == 2


# --- get_filing_html ------------------------------------------------------


@respx.mock
async def test_get_filing_html_fetches_and_caches():
    url = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    body = b"<html>10-K body</html>"
    route = respx.get(url).mock(return_value=httpx.Response(200, content=body))

    out1 = await edgar_client.get_filing_html(
        "0000320193", "0000320193-24-000123", "aapl-20240928.htm"
    )
    out2 = await edgar_client.get_filing_html(
        "0000320193", "0000320193-24-000123", "aapl-20240928.htm"
    )
    assert out1 == body
    assert out2 == body
    assert route.call_count == 1  # second call from cache


# --- get_company_facts ----------------------------------------------------


@respx.mock
async def test_get_company_facts_fetch_and_cache():
    payload = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": []}}}}}
    route = respx.get(FACTS_URL).mock(return_value=httpx.Response(200, json=payload))

    facts = await edgar_client.get_company_facts("0000320193")
    assert facts == payload
    # Second call uses the on-disk cache.
    facts2 = await edgar_client.get_company_facts("0000320193")
    assert facts2 == payload
    assert route.call_count == 1


# --- Error mapping --------------------------------------------------------


@respx.mock
async def test_404_raises_edgar_not_found():
    respx.get(SUBMISSIONS_URL).mock(return_value=httpx.Response(404))
    with pytest.raises(EdgarNotFound):
        await edgar_client.get_submissions("0000320193")


@respx.mock
async def test_403_raises_edgar_auth_error():
    respx.get(SUBMISSIONS_URL).mock(return_value=httpx.Response(403))
    with pytest.raises(EdgarAuthError):
        await edgar_client.get_submissions("0000320193")


def test_missing_user_agent_raises_auth_error(monkeypatch):
    monkeypatch.setattr(edgar_client.settings, "SEC_USER_AGENT", "no-email-here")
    with pytest.raises(EdgarAuthError):
        edgar_client._headers()


# --- Retry behavior -------------------------------------------------------


@respx.mock
async def test_429_retried_then_succeeds():
    respx.get(SUBMISSIONS_URL).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(200, json=_submissions_payload()),
        ]
    )
    filings = await edgar_client.get_submissions("0000320193")
    assert len(filings) == 2


@respx.mock
async def test_persistent_5xx_raises_server_error():
    respx.get(SUBMISSIONS_URL).mock(return_value=httpx.Response(503))
    with pytest.raises(EdgarServerError):
        await edgar_client.get_submissions("0000320193")


@respx.mock
async def test_persistent_429_raises_rate_limited_after_retries():
    # tenacity reraises the last exception (reraise=True), so callers see EdgarRateLimited.
    respx.get(SUBMISSIONS_URL).mock(return_value=httpx.Response(429))
    with pytest.raises(EdgarRateLimited):
        await edgar_client.get_submissions("0000320193")


# --- FilingMeta sanity ----------------------------------------------------


def test_filing_meta_round_trip():
    payload = _submissions_payload()
    # Construct a FilingMeta directly to ensure the schema is stable.
    fm = FilingMeta(
        accession_no=payload["filings"]["recent"]["accessionNumber"][0],
        form_type="10-K",
        filing_date="2024-11-01",
        period_of_report="2024-09-28",
        primary_doc="aapl-20240928.htm",
        primary_doc_url="https://example.test/aapl-20240928.htm",
    )
    assert fm.accession_no.startswith("0000320193")
    # Pydantic dump round-trips.
    assert json.loads(fm.model_dump_json())["form_type"] == "10-K"
