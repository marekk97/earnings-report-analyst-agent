"""Pure, unit-tested financial formulas. No I/O, no XBRL knowledge: numbers in, number out.

Conventions
* Percentages are returned as percent values (12.5 means 12.5%).
* Growth uses the absolute value of the base so a move from -10 to -5 reads as +50% (an improvement).
* Flow-over-stock ratios (ROE, ROA, inventory turnover) on a quarter are annualised to 365 days.
"""

from __future__ import annotations


class FormulaError(ValueError):
    """Input that makes a formula meaningless (e.g. division by zero)."""


def _nonzero(x: float, what: str) -> float:
    if x == 0:
        raise FormulaError(f"{what} is zero")
    return x


def growth_pct(current: float, prior: float) -> float:
    return (current - prior) / abs(_nonzero(prior, "prior-period value")) * 100.0


def margin_pct(numerator: float, revenue: float) -> float:
    return numerator / _nonzero(revenue, "revenue") * 100.0


def gross_profit(revenue: float, cost_of_revenue: float) -> float:
    return revenue - cost_of_revenue


def free_cash_flow(operating_cash_flow: float, capital_expenditures: float) -> float:
    # Capex is tagged as a positive "payments to acquire" amount; subtract its magnitude.
    return operating_cash_flow - abs(capital_expenditures)


def ratio(numerator: float, denominator: float, what: str = "denominator") -> float:
    return numerator / _nonzero(denominator, what)


def average(opening: float | None, closing: float) -> float:
    return closing if opening is None else (opening + closing) / 2.0


def annualize(value: float, days: int) -> float:
    return value * 365.0 / _nonzero(days, "period length")


def return_on_pct(income: float, avg_base: float, days: int) -> float:
    """ROE / ROA: annualised income over the average balance."""
    return annualize(income, days) / _nonzero(avg_base, "average balance") * 100.0


def days_sales_outstanding(receivables: float, revenue: float, days: int) -> float:
    return receivables / _nonzero(revenue, "revenue") * days


def inventory_turnover(cost_of_revenue: float, avg_inventory: float, days: int) -> float:
    return annualize(cost_of_revenue, days) / _nonzero(avg_inventory, "average inventory")


def interest_coverage(operating_income: float, interest_expense: float) -> float:
    return operating_income / abs(_nonzero(interest_expense, "interest expense"))
