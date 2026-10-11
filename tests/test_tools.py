import json
from pathlib import Path

from earnings_analyst import grounding
from earnings_analyst.tools import build_tools
from earnings_analyst.tools.python_tool import execute


def tools_for(session):
    return {t.name: t for t in build_tools(session)}


def call(tools, name, **kwargs):
    return json.loads(tools[name].invoke(kwargs))


def test_toolset_matches_the_brief(session):
    assert set(tools_for(session)) == {
        "calculate_metric", "get_metric_history", "get_filing_section", "get_segment_data",
        "get_8k_earnings_release", "run_python", "save_brief",
    }


def test_calculate_metric_has_provenance_and_comparison(session):
    compact = call(tools_for(session), "calculate_metric", ticker="ACME", metric="gross_margin", compare_to="prior_year")
    assert compact["ok"] and compact["period"] == "Q3 FY2025" and compact["display"] == "40.0%"
    assert compact["citation"].startswith("10-Q, Q3 FY2025, filed ") and "provenance" not in compact
    assert compact["comparison"]["period"] == "Q3 FY2024" and compact["comparison"]["change"] == "+0.0 pp"
    out = session.ledger.metrics[-1]  # the full result kept for the brief's JSON
    prov = out["provenance"]
    assert prov["form"] == "10-Q" and prov["fiscal_period"] == "Q3 FY2025" and prov["filing_date"]
    assert out["citation"].startswith("10-Q, Q3 FY2025, filed ")
    assert all(i["provenance"]["filing_date"] for i in out["inputs"])
    assert out["comparison"]["period"]["label"] == "Q3 FY2024" and out["comparison"]["change_pp"] == 0
    assert session.ledger.metrics[-1]["metric"] == "gross_margin"


def test_calculate_metric_aliases_and_unknown(session):
    tools = tools_for(session)
    assert call(tools, "calculate_metric", ticker="ACME", metric="revenue growth")["metric"] == "revenue_growth_yoy"
    err = call(tools, "calculate_metric", ticker="ACME", metric="ebitda_margn")
    assert err["ok"] is False and err["error"] == "unknown_metric" and "run_python" in err["message"]
    assert session.ledger.errors and session.ledger.errors[-1]["error"] == "unknown_metric"


def test_missing_data_is_never_empty(session):
    err = call(tools_for(session), "calculate_metric", ticker="ACME", metric="inventory_turnover")
    assert err == {**err, "ok": False, "error": "missing_data"} and err["message"]


def test_metric_history(session):
    out = call(tools_for(session), "get_metric_history", ticker="ACME", metric="revenue_growth_yoy", periods=8)
    assert out["ok"] and len(out["points"]) == 8
    assert out["points"][-1]["period"] == "Q3 FY2025" and out["points"][-1]["citation"]
    assert out["points"][0]["value"] is None and out["points"][0]["error"]  # no prior-year data for the first one


def test_run_python_executes_and_logs(session):
    out = call(tools_for(session), "run_python", code="print(round(37/12, 3))", purpose="test ratio",
               metric_name="test_ratio")
    assert out["ok"] and out["stdout"].strip() == "3.083"
    rows = [json.loads(line) for line in Path(session.python_log_path).read_text().splitlines()]
    assert rows[-1]["metric_name"] == "test_ratio" and rows[-1]["code"].startswith("print")


def test_run_python_timeout_and_isolation():
    assert execute("while True: pass", timeout=1)["error"] == "timeout"
    out = execute("import os; print(sorted(k for k in os.environ if 'KEY' in k))")
    assert out["stdout"].strip() == "[]"


def test_save_brief_writes_markdown_and_json(session):
    tools = tools_for(session)
    call(tools, "calculate_metric", ticker="ACME", metric="revenue_growth_yoy")
    md = ("# Acme (ACME) - Q3 FY2025 earnings brief\n\nRevenue grew 10.0% year over year "
          "[10-Q, Q3 FY2025, filed 2025-11-04], reaching $132.\n")
    out = call(tools, "save_brief", markdown=md, ticker="ACME", period_label="Q3 FY2025")
    assert out["ok"] and out["warnings"] == []
    text = Path(out["path"]).read_text()
    assert "not investment advice" in text and "```json" in text
    data = json.loads(Path(out["json_path"]).read_text())
    assert data["metrics"][0]["metric"] == "revenue_growth_yoy"
    assert data["metrics"][0]["inputs"][0]["provenance"]["form"] == "10-Q"
    assert data["grounding"]["ungrounded"] == []


def test_save_brief_flags_ungrounded_numbers_and_advice(session):
    tools = tools_for(session)
    call(tools, "calculate_metric", ticker="ACME", metric="revenue_growth_yoy")
    md = "# Brief\nRevenue grew 12.7% and margins hit 55.5%. We recommend buying the shares; price target $300."
    out = call(tools, "save_brief", markdown=md, ticker="ACME", period_label="Q3 FY2025")
    joined = " ".join(out["warnings"])
    assert "12.7%" in joined and "55.5%" in joined and "investment-advice" in joined


def test_grounding_matches_displayed_precision():
    pool = [11213000000.0, 42.7894, -4.3, 0.48]
    report = grounding.check(
        "Revenue was $11.2 billion, gross margin 42.8%, down 4.3%, EPS $0.48 [10-Q, Q1 FY2027, filed 2026-10-02].",
        pool, [])
    assert report.checked == 4 and report.ungrounded == []
    report = grounding.check("Wholesale revenue was $6.8 billion.", pool, ["NIKE Brand wholesale revenues were $6.8 billion"])
    assert report.grounded_text == ["$6.8 billion"]


def test_ledger_keeps_each_company(session):
    tools = tools_for(session)
    call(tools, "calculate_metric", ticker="ACME", metric="gross_margin")
    session.ledger.metrics[0]["ticker"] = "OTHER"  # pretend the first result came from another company
    call(tools, "calculate_metric", ticker="ACME", metric="gross_margin")
    call(tools, "calculate_metric", ticker="ACME", metric="gross_margin")  # a repeat replaces, not appends
    assert [m["ticker"] for m in session.ledger.metrics] == ["OTHER", "ACME"]
