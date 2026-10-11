from earnings_analyst.accuracy import contexts, period_hints, resolve_facts, score_case

GOLDENS = {
    "g23": {"ticker": "AAPL", "metric": "gross_margin", "period": "FY2023", "expected": 44.1311},
    "g22": {"ticker": "AAPL", "metric": "gross_margin", "period": "FY2022", "expected": 43.3096},
    "q1": {"ticker": "AAPL", "metric": "gross_margin", "period": "Q1 FY2024", "expected": 45.875},
    "q2": {"ticker": "AAPL", "metric": "gross_margin", "period": "Q2 FY2024", "expected": 46.5781},
    "ko": {"ticker": "KO", "metric": "operating_margin", "period": "FY2023", "expected": 24.7213},
    "pep": {"ticker": "PEP", "metric": "operating_margin", "period": "FY2023", "expected": 13.1036},
    "rg": {"ticker": "AAPL", "metric": "revenue_growth_yoy", "period": "FY2023", "expected": -2.8005},
}


def score(facts_spec, brief):
    case = {"id": "t", "question": "q", "facts": facts_spec}
    return score_case(case, resolve_facts(case, GOLDENS), brief, tool_numbers=[])


def statuses(res):
    return [r["status"] for r in res["results"]]


def test_two_periods_and_change_in_a_table():
    brief = """# Apple
| Metric | FY2023 | FY2022 | Change |
|---|---|---|---|
| Gross margin | **44.1%** | 43.3% | +0.8 pp |
"""
    res = score([{"golden": "g23", "period_hint": True}, {"golden": "g22", "period_hint": True},
                 {"change_of": ["g23", "g22"]}], brief)
    assert statuses(res) == ["correct", "correct", "correct"] and res["score"] == 1.0


def test_prose_wrong_and_missing():
    brief = "Apple's gross margin was 41.0% in fiscal 2023. Revenue was flat."
    res = score([{"golden": "g23"}, {"golden": "ko", "aliases": ["Coca-Cola"]}], brief)
    assert statuses(res) == ["wrong", "missing"]
    assert res["results"][0]["candidates"] == ["41.0%"]


def test_sign_is_ignored_and_precision_respected():
    assert statuses(score([{"golden": "rg"}], "Revenue fell 2.8% in FY2023."))[0] == "correct"
    assert statuses(score([{"golden": "rg"}], "Revenue fell 2.6% in FY2023."))[0] == "wrong"


def test_time_series_rows_need_the_right_period():
    brief = """| Quarter | Gross margin |
|---|---|
| Q1 FY2024 | 46.6% |
| Q2 FY2024 | 45.9% |
"""
    # Values are swapped between quarters: both must be marked wrong even though each number exists.
    res = score([{"golden": "q1", "period_hint": True}, {"golden": "q2", "period_hint": True}], brief)
    assert statuses(res) == ["wrong", "wrong"]


def test_cross_company_and_usd_scales():
    brief = "Coca-Cola's operating margin was 24.7%, well above PepsiCo at 13.1%."
    res = score([{"golden": "ko", "aliases": ["Coca-Cola"]}, {"golden": "pep", "aliases": ["PepsiCo"]}], brief)
    assert statuses(res) == ["correct", "correct"]
    brief = "| Segment | Revenue |\n|---|---|\n| Americas | $162.6 B |\n| Europe | $94,294 million |"
    res = score([{"value": 162560000000, "unit": "usd", "aliases": ["Americas"]},
                 {"value": 94294000000, "unit": "usd", "aliases": ["Europe"]}], brief)
    assert statuses(res) == ["correct", "correct"]


def test_helpers():
    assert period_hints("Q2 FY2024") == ["Q2", "second quarter", "2Q"]
    assert "2022" in period_hints("FY2022")
    assert contexts("| a | b |\n|---|---|\n| x | 1 |")[-1] == ("| a | b | | x | 1 |", "| x | 1 |")
