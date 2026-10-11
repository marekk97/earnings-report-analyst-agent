from datetime import date

import pytest

from earnings_analyst.provenance import DataError
from earnings_analyst.sec.xbrl import CompanyFacts


def test_fiscal_calendar_labels(facts):
    q = [p.label for p in facts.periods("quarter")]
    assert q == ["Q4 FY2023", "Q1 FY2024", "Q2 FY2024", "Q3 FY2024", "Q4 FY2024", "Q1 FY2025", "Q2 FY2025", "Q3 FY2025"]
    assert [p.label for p in facts.periods("annual")] == ["FY2023", "FY2024"]
    latest = facts.latest("quarter")
    assert latest.end == date(2025, 9, 30) and latest.start == date(2025, 7, 1) and latest.days == 92


@pytest.mark.parametrize("spec,label", [
    ("latest", "Q3 FY2025"), ("Q2 FY2025", "Q2 FY2025"), ("q2 2025", "Q2 FY2025"), ("2025Q2", "Q2 FY2025"),
    ("FY2024 Q4", "Q4 FY2024"), ("2025-06-30", "Q2 FY2025"), ("FY2024", "FY2024"), ("FY24", "FY2024"),
])
def test_resolve_period(facts, spec, label):
    ptype = "annual" if spec.startswith("FY") and "Q" not in spec else "quarter"
    assert facts.resolve_period(spec, ptype).label == label


def test_unknown_period_lists_alternatives(facts):
    with pytest.raises(DataError) as exc:
        facts.resolve_period("Q1 FY2019")
    assert exc.value.code == "period_not_found"
    assert "Q3 FY2025" in exc.value.details["available_periods"]


def test_shift(facts):
    p = facts.resolve_period("Q3 FY2025")
    assert facts.shift(p, "prior_year").label == "Q3 FY2024"
    assert facts.shift(p, "prior_quarter").label == "Q2 FY2025"


def test_direct_quarter(facts):
    p = facts.resolve_period("Q2 FY2025")
    f = facts.lookup(["Revenues"], p, "duration")
    assert f.value == 121 and f.derived is None and f.form == "10-Q"


def test_q4_derived_from_annual_minus_nine_months(facts):
    p = facts.resolve_period("Q4 FY2024")
    f = facts.lookup(["Revenues"], p, "duration")
    assert f.value == pytest.approx(130)
    assert "derived" in f.derived and f.form == "10-K"


def test_cash_flow_quarter_derived_from_ytd(facts):
    p = facts.resolve_period("Q2 FY2025")
    f = facts.lookup(["NetCashProvidedByUsedInOperatingActivities"], p, "duration")
    assert f.value == pytest.approx(121 * 0.2)
    assert f.derived


def test_per_share_values_are_not_derived(facts):
    p = facts.resolve_period("Q4 FY2024")
    assert facts.lookup(["EarningsPerShareDiluted"], p, "duration", unit="USD/shares", additive=False) is None


def test_restated_value_wins(facts):
    p = facts.resolve_period("Q1 FY2024")
    f = facts.lookup(["NetIncomeLoss"], p, "duration")
    assert f.value == 12.0 and f.filed > "2025-01-01"
    prov = facts.provenance(f, p)
    assert prov.fiscal_period == "Q1 FY2024" and "comparative" in prov.note


def test_concept_fallback_order(raw_facts):
    us = raw_facts["facts"]["us-gaap"]
    us["RevenueFromContractWithCustomerExcludingAssessedTax"] = us.pop("Revenues")
    facts = CompanyFacts(raw_facts)
    p = facts.resolve_period("latest")
    f = facts.lookup(["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"], p, "duration")
    assert f.concept == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_instant_lookup(facts):
    p = facts.resolve_period("Q3 FY2025")
    assert facts.lookup(["LiabilitiesCurrent"], p, "instant").value == 100.0
    assert facts.lookup(["InventoryNet"], p, "instant") is None


def test_ifrs_filer_is_a_structured_error():
    with pytest.raises(DataError) as exc:
        CompanyFacts({"cik": 2, "entityName": "Foreign plc", "facts": {"ifrs-full": {}}})
    assert exc.value.code == "no_us_gaap_facts"
