"""Anon-tier ticker allowlist gate. See docs/anon_hardening_plan.md §7.

Runs immediately after `plan`, before any retrieval or model work. For
anonymous callers, refuses any question whose resolved ticker is not in
`settings.ANON_TICKER_ALLOWLIST`. Meta / no-ticker questions pass
through (concierge handles them separately, and they don't reveal
restricted data anyway).

Pure Python — no LLM, no DB. Cheap enough to run on every request.
"""

from __future__ import annotations

import logging

from app.agent.state import AgentState
from app.settings import settings

logger = logging.getLogger(__name__)


async def anon_gate(state: AgentState) -> dict:
    if state.tier != "anon":
        return {}
    ticker = (state.ticker or "").upper()
    if not ticker:
        # Concierge / meta / no-ticker questions pass through — they
        # don't reveal restricted data and refusing them would be
        # confusing UX ("what can you do?" should always work).
        return {}
    allowlist = settings.anon_ticker_allowlist_upper
    if ticker in allowlist:
        return {}
    logger.info(
        "anon_gate.refused",
        extra={"ticker": ticker, "allowlist_size": len(allowlist)},
    )
    return {
        "question_type": "REFUSE",
        "refusal_code": "ANON_TICKER_NOT_ALLOWED",
        "refusal_reason": (
            f"Sign in to ask about {ticker}. The free preview covers "
            f"{', '.join(sorted(allowlist))}."
        ),
    }


__all__ = ["anon_gate"]
