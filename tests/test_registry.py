import pytest

from earnings_analyst.metrics.registry import CONCEPTS, METRIC_NAMES, REGISTRY, evaluate, total_debt
from earnings_analyst.provenance import DataError


def test_registry_covers_the_brief():
    expected = {
        "revenue_growth_yoy", "revenue_growth_qoq", "gross_margin", "operating_margin", "net_margin",
        "eps_growth_yoy", "free_cash_flow", "fcf_margin", "current_ratio", "debt_to_equity",
        "interest_coverage", "roe", "roa", "dso", "inventory_turnover",
    }
    assert expected <= set(METRIC_NAMES)
    for m in REGISTRY.values():
        assert m.formula and m.description and m.inputs
        assert all(i.field in CONCEPTS for i in m.inputs)


def test_revenue_growth_yoy(facts):
    mv = evaluate(facts, REGISTRY["revenue_growth_yoy"], facts.resolve_period("Q3 FY2025"))
    assert mv.value == pytest.approx((132 - 120) / 120 * 100)
    assert {r.key for r in mv.inputs} == {"revenue", "revenue@prior_year"}


def test_revenue_growth_qoq_across_derived_q4(facts):
    mv = evaluate(facts, REGISTRY["revenue_growth_qoq"], facts.resolve_period("Q1 FY2025"))
    assert mv.value == pytest.approx((115 - 130) / 130 * 100)
    assert any("derived" in n for n in mv.notes)


def test_gross_margin_falls_back_to_revenue_minus_cost(facts):
    mv = evaluate(facts, REGISTRY["gross_margin"], facts.resolve_period("Q3 FY2025"))
    assert mv.value == pytest.approx(40.0)
    assert any("revenue minus cost" in n for n in mv.notes)


def test_fcf_on_derived_quarter(facts):
    mv = evaluate(facts, REGISTRY["free_cash_flow"], facts.resolve_period("Q3 FY2025"))
    assert mv.value == pytest.approx(132 * 0.2 - 132 * 0.05)


def test_qoq_not_allowed_for_annual(facts):
    with pytest.raises(DataError) as exc:
        evaluate(facts, REGISTRY["revenue_growth_qoq"], facts.resolve_period("FY2024", "annual"))
    assert exc.value.code == "unsupported_period_type"


def test_missing_inventory_is_structured(facts):
    with pytest.raises(DataError) as exc:
        evaluate(facts, REGISTRY["inventory_turnover"], facts.resolve_period("latest"))
    err = exc.value.to_dict()
    assert err["ok"] is False and err["error"] == "missing_data"
    assert "inventory not reported" in err["message"]
    assert "InventoryNet" in err["concepts_tried"]["inventory"]


def test_missing_operating_income_without_expenses(facts):
    with pytest.raises(DataError) as exc:
        evaluate(facts, REGISTRY["operating_margin"], facts.resolve_period("latest"))
    assert "operating income" in exc.value.message


def test_roe_uses_average_equity_and_annualises(facts):
    p = facts.resolve_period("Q1 FY2025")
    mv = evaluate(facts, REGISTRY["roe"], p)
    assert mv.value == pytest.approx(15.0 * 365 / p.days / 400 * 100)
    assert any("annualised" in n for n in mv.notes)


def test_total_debt_components():
    debt, notes = total_debt({"long_term_debt_noncurrent": 100.0, "debt_current": 10.0, "short_term_borrowings": 5.0})
    assert debt == 115.0 and "short-term" in notes[0]
    debt, _ = total_debt({"long_term_debt_total": 80.0})
    assert debt == 80.0


def _add(raw, concept, unit, start, end, val, accn, form, filed, fy, fp):
    raw["facts"]["us-gaap"].setdefault(concept, {"units": {}})["units"].setdefault(unit, []).append(
        {"start": start, "end": end, "val": val, "accn": accn, "form": form, "filed": filed, "fy": fy, "fp": fp})


def test_growth_uses_one_filing_so_splits_are_not_mixed(raw_facts):
    from earnings_analyst.sec.xbrl import CompanyFacts

    # FY2024 10-K (pre-split) reports FY2024 EPS 4.60 and FY2023 EPS 3.90 (fixture values).
    # A later filing re-reports FY2024 EPS split-adjusted (0.46) but not FY2023.
    _add(raw_facts, "EarningsPerShareDiluted", "USD/shares", "2024-01-01", "2024-12-31", 0.46,
         "0000000001-26-000004", "10-K", "2026-02-15", 2025, "FY")
    facts = CompanyFacts(raw_facts)
    mv = evaluate(facts, REGISTRY["eps_growth_yoy"], facts.resolve_period("FY2024", "annual"))
    assert mv.value == pytest.approx((4.60 - 3.90) / 3.90 * 100)
    assert {r.fact.accession for r in mv.inputs} == {"0000000001-24-000004"}


def test_annual_ratios_are_not_rescaled_for_long_fiscal_years(facts):
    p = facts.resolve_period("FY2024", "annual")
    mv = evaluate(facts, REGISTRY["dso"], p)
    assert mv.value == pytest.approx(60.0 / 460 * 365)
