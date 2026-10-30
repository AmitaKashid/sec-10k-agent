"""Opt-in Groq end-to-end smoke test against the isolated test database.

Run with scripts/run_tests.py --suite live. Real API quota is consumed.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import GRAPH
from app.agent.state import AgentState
from tests.integration.conftest import SKIP_REASON, db_reachable

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not db_reachable(), reason=SKIP_REASON),
]


@pytest.fixture(autouse=True)
async def _require_groq(agent_session: AsyncSession):
    """`agent_session` seeds AAPL inline; gate on the configured Groq key."""
    from app.settings import settings
    if not settings.GROQ_API_KEY:
        pytest.skip("GROQ_API_KEY not set")



async def test_aapl_revenue_live_end_to_end():
    """Groq plans the question; exact XBRL formatting and citation must pass."""
    state = AgentState(
        question="What was Apple's revenue in fiscal year 2024?",
        request_id="live-rev-1",
        plan_model_override="openai/gpt-oss-20b",
        validate_model_override="openai/gpt-oss-120b",
    )
    out = await GRAPH.ainvoke(state)
    final = AgentState.model_validate(out)

    # Sanity: pipeline reached the end
    assert final.question_type == "A_numerical", f"plan misclassified: {final.question_type}"
    assert final.cik == "0000320193"
    assert final.fiscal_year == 2024
    assert final.xbrl_fact is not None
    assert final.xbrl_fact.value == Decimal("391035000000")

    assert final.verified is True, final.refusal_reason_final or final.refusal_reason
    assert final.final_answer is not None
    assert "391,035,000,000" in final.final_answer
    assert final.claims[0].source_ref.value == Decimal("391035000000")
