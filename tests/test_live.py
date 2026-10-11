"""Golden registry cases against live SEC EDGAR data. Run with: pytest -m live"""

from pathlib import Path

import pytest
import yaml

from earnings_analyst.evals import run_tool_case
from earnings_analyst.session import AnalystSession
from earnings_analyst.tools.metric_tools import build

CASES = yaml.safe_load((Path(__file__).parents[1] / "evals" / "cases.yaml").read_text())["tool_cases"]


@pytest.fixture(scope="module")
def live():
    session = AnalystSession("live tests")
    return {t.name: t for t in build(session)}, session


@pytest.mark.live
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_golden_case(case, live):
    result = run_tool_case(case, *live)
    assert result["passed"], result
