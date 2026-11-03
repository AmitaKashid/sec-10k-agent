"""Score the visible answer independently of retrieval."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.metrics import Expected, score_row


@pytest.mark.parametrize(("answer", "correct"), [
    ("Revenue was $110 million [1].", False),
    ("Revenue was $100 million [1].", True),
    ("FY2025 revenue was $100,000,000 [1].", True),
    ("Revenue was $100 million or $110 million.", False),
    ("Revenue was not disclosed.", False),
    (None, False),
])
def test_displayed_numeric_answer(answer, correct):
    expected = Expected(id="revenue", question="What was revenue?",
                        pipeline="xbrl", value="100000000", tolerance_pct=0)
    state = {"final_answer": answer, "draft_answer": "Revenue was $100 million.",
             "xbrl_fact": {"value": "100000000"}}
    assert score_row(expected, state).numeric_correct is correct
