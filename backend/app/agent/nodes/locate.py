"""Locate node — confirms the requested Item section was actually parsed.

For Type A (numerical) this is a no-op pass: XBRL doesn't depend on a
narrative section. For Type B we check that at least one chunk exists
for (filing, item).
"""

from __future__ import annotations

from sqlalchemy import select

from app.agent.state import AgentState, RefusalCode
from app.data.db import async_session_factory
from app.data.models import Chunk


async def locate(state: AgentState) -> dict:
    if state.question_type == "REFUSE":
        return {}

    if state.question_type == "A_numerical":
        return {
            "located": True,
            "trace_events": state.trace_events + [{"node": "locate", "skipped_for_type_a": True}],
        }

    # Type B
    if not state.filing or not state.item:
        return _refuse(state, "Locate is missing filing or item context.", "INTERNAL_ERROR")

    async with async_session_factory() as session:
        present = (
            await session.execute(
                select(Chunk.id)
                .where(
                    Chunk.accession_no == state.filing.accession_no,
                    Chunk.item == state.item,
                )
                .limit(1)
            )
        ).scalar_one_or_none()

    if present is None:
        return _refuse(
            state,
            f"Item {state.item} wasn't successfully parsed for "
            f"{state.ticker} fiscal year {state.fiscal_year}.",
            "LOCATE_FAILED",
        )

    return {
        "located": True,
        "trace_events": state.trace_events
        + [{"node": "locate", "item": state.item, "located": True}],
    }


def _refuse(state: AgentState, reason: str, code: RefusalCode) -> dict:
    return {
        "question_type": "REFUSE",
        "refusal_reason": reason,
        "refusal_code": code,
        "trace_events": state.trace_events
        + [{"node": "locate", "refused": True, "reason": reason, "refusal_code": code}],
    }
