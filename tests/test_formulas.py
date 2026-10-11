import pytest

from earnings_analyst.metrics import formulas as F


def test_growth_and_negative_base():
    assert F.growth_pct(110, 100) == pytest.approx(10.0)
    assert F.growth_pct(383285, 394328) == pytest.approx(-2.8005, abs=1e-4)
    # From a loss of 10 to a loss of 5 is an improvement.
    assert F.growth_pct(-5, -10) == pytest.approx(50.0)


def test_margins_and_ratios():
    assert F.margin_pct(169148, 383285) == pytest.approx(44.1311, abs=1e-4)
    assert F.ratio(143566, 145308) == pytest.approx(0.98801, abs=1e-5)
    assert F.interest_coverage(109433, -2935) == pytest.approx(37.2855, abs=1e-4)


def test_free_cash_flow_uses_capex_magnitude():
    assert F.free_cash_flow(118548, 44477) == 74071
    assert F.free_cash_flow(118548, -44477) == 74071


def test_annualised_returns_and_turnover():
    assert F.return_on_pct(25, 400, 91) == pytest.approx(25 * 365 / 91 / 400 * 100)
    assert F.return_on_pct(100, 400, 365) == pytest.approx(25.0)
    assert F.average(None, 10) == 10
    assert F.average(8, 12) == 10
    assert F.inventory_turnover(600, 150, 365) == pytest.approx(4.0)
    assert F.days_sales_outstanding(60, 120, 91) == pytest.approx(45.5)


@pytest.mark.parametrize("fn,args", [
    (F.growth_pct, (1, 0)), (F.margin_pct, (1, 0)), (F.ratio, (1, 0)), (F.interest_coverage, (1, 0)),
])
def test_division_by_zero_is_a_formula_error(fn, args):
    with pytest.raises(F.FormulaError):
        fn(*args)
