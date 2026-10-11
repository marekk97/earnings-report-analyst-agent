"""The metric registry: each metric maps to a formula and the XBRL fields it needs.

The model chooses *which* metrics to compute; this module alone decides *how* they are computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from ..provenance import DataError
from ..sec.xbrl import CompanyFacts, Fact, Period
from . import formulas as F

Offset = Literal["current", "prior_year", "prior_quarter", "opening"]
Unit = Literal["percent", "ratio", "usd", "days", "times"]

# XBRL concept fallbacks, in priority order. Companies change tags over time, so a period uses the
# first concept that has a value for it.
CONCEPTS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenuesNetOfInterestExpense",
    ),
    "cost_of_revenue": (
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
        "CostOfServices",
    ),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "operating_expenses": ("OperatingExpenses",),
    "sga": ("SellingGeneralAndAdministrativeExpense",),
    # The "excluding acquired in-process" tag, when present, is the income-statement R&D line; some filers
    # (e.g. J&J) use the plain tag only for a minor item.
    "research_and_development": (
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost",
    ),
    "net_income": ("NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"),
    "eps_diluted": ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"),
    "operating_cash_flow": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "capital_expenditures": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsForCapitalImprovements",
    ),
    "current_assets": ("AssetsCurrent",),
    "current_liabilities": ("LiabilitiesCurrent",),
    "long_term_debt_noncurrent": ("LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"),
    "long_term_debt_total": ("LongTermDebt", "DebtInstrumentCarryingAmount"),
    "debt_current": ("LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent", "DebtCurrent"),
    "short_term_borrowings": ("ShortTermBorrowings", "CommercialPaper"),
    "stockholders_equity": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "total_assets": ("Assets",),
    "interest_expense": (
        "InterestExpense",
        "InterestExpenseNonoperating",
        "InterestExpenseDebt",
        "InterestAndDebtExpense",
        "InterestExpenseNet",
    ),
    "accounts_receivable": ("AccountsReceivableNetCurrent", "ReceivablesNetCurrent", "AccountsReceivableNet"),
    "inventory": ("InventoryNet", "InventoryFinishedGoodsNetOfReserves", "InventoryFinishedGoods", "InventoryGross"),
}

PER_SHARE = {"eps_diluted"}
INSTANTS = {
    "current_assets", "current_liabilities", "long_term_debt_noncurrent", "long_term_debt_total", "debt_current",
    "short_term_borrowings", "stockholders_equity", "total_assets", "accounts_receivable", "inventory",
}


@dataclass(frozen=True)
class Input:
    field: str
    offset: Offset = "current"
    required: bool = True

    @property
    def key(self) -> str:
        return self.field if self.offset == "current" else f"{self.field}@{self.offset}"

    @property
    def kind(self) -> Literal["duration", "instant"]:
        return "instant" if self.field in INSTANTS else "duration"


Values = dict[str, float | None]
Compute = Callable[[Values, Period], tuple[float, list[str]]]


@dataclass(frozen=True)
class Metric:
    name: str
    description: str
    unit: Unit
    formula: str
    inputs: tuple[Input, ...]
    compute: Compute
    period_types: tuple[str, ...] = ("quarter", "annual")
    notes: tuple[str, ...] = field(default_factory=tuple)


def _need(v: Values, *names: str) -> list[float]:
    missing = [n for n in names if v.get(n) is None]
    if missing:
        raise MissingInputs(missing)
    return [v[n] for n in names]  # type: ignore[misc]


class MissingInputs(Exception):
    def __init__(self, fields: list[str]):
        super().__init__(", ".join(fields))
        self.fields = fields


# ---------------------------------------------------------------------- compute functions


def _growth(field_: str, offset: Offset) -> Compute:
    def compute(v: Values, p: Period) -> tuple[float, list[str]]:
        cur, prior = _need(v, field_, f"{field_}@{offset}")
        notes = []
        if prior < 0:
            notes.append("prior-period value was negative; growth is measured against its absolute value")
        return F.growth_pct(cur, prior), notes
    return compute


def _gross_profit(v: Values) -> tuple[float, list[str]]:
    if v.get("gross_profit") is not None:
        return v["gross_profit"], []  # type: ignore[return-value]
    if v.get("revenue") is not None and v.get("cost_of_revenue") is not None:
        return F.gross_profit(v["revenue"], v["cost_of_revenue"]), ["gross profit computed as revenue minus cost of revenue"]
    raise MissingInputs(["gross_profit", "cost_of_revenue"])


def _gross_margin(v: Values, p: Period) -> tuple[float, list[str]]:
    (rev,) = _need(v, "revenue")
    gp, notes = _gross_profit(v)
    return F.margin_pct(gp, rev), notes


def _operating_income(v: Values) -> tuple[float, list[str]]:
    """Tagged operating income, else gross profit less operating expenses (flagged as an approximation)."""
    if v.get("operating_income") is not None:
        return v["operating_income"], []  # type: ignore[return-value]
    # Only approximate for companies that present a gross-profit line themselves. Without one (pharma, energy,
    # banks...), revenue - cost of goods - SG&A leaves out large cost lines and badly overstates the margin.
    if v.get("gross_profit") is None:
        raise MissingInputs(["operating_income"])
    gp, notes = v["gross_profit"], []
    if v.get("operating_expenses") is not None:
        return gp - v["operating_expenses"], notes + [  # type: ignore[operator]
            "operating income not tagged; approximated as gross profit minus total operating expenses"]
    if v.get("sga") is not None:
        rd = v.get("research_and_development") or 0.0
        return gp - v["sga"] - rd, notes + [  # type: ignore[operator]
            "operating income not tagged; approximated as gross profit minus SG&A"
            + (" and R&D" if rd else "") + " (excludes other operating items)"]
    raise MissingInputs(["operating_income"])


OPERATING_INCOME_INPUTS = (
    Input("operating_income", required=False), Input("revenue"), Input("gross_profit", required=False),
    Input("cost_of_revenue", required=False), Input("operating_expenses", required=False),
    Input("sga", required=False), Input("research_and_development", required=False),
)


def _operating_margin(v: Values, p: Period) -> tuple[float, list[str]]:
    (rev,) = _need(v, "revenue")
    oi, notes = _operating_income(v)
    return F.margin_pct(oi, rev), notes


def _margin(field_: str) -> Compute:
    def compute(v: Values, p: Period) -> tuple[float, list[str]]:
        num, rev = _need(v, field_, "revenue")
        return F.margin_pct(num, rev), []
    return compute


def _fcf(v: Values, p: Period) -> tuple[float, list[str]]:
    cfo, capex = _need(v, "operating_cash_flow", "capital_expenditures")
    return F.free_cash_flow(cfo, capex), []


def _fcf_margin(v: Values, p: Period) -> tuple[float, list[str]]:
    cfo, capex, rev = _need(v, "operating_cash_flow", "capital_expenditures", "revenue")
    return F.margin_pct(F.free_cash_flow(cfo, capex), rev), []


def _current_ratio(v: Values, p: Period) -> tuple[float, list[str]]:
    ca, cl = _need(v, "current_assets", "current_liabilities")
    return F.ratio(ca, cl, "current liabilities"), []


def total_debt(v: Values) -> tuple[float, list[str]]:
    components: list[tuple[str, float]] = []
    if v.get("long_term_debt_noncurrent") is not None:
        components.append(("long-term debt (non-current)", v["long_term_debt_noncurrent"]))  # type: ignore[arg-type]
        if v.get("debt_current") is not None:
            components.append(("current portion of debt", v["debt_current"]))  # type: ignore[arg-type]
    elif v.get("long_term_debt_total") is not None:
        components.append(("long-term debt incl. current portion", v["long_term_debt_total"]))  # type: ignore[arg-type]
    elif v.get("debt_current") is not None:
        components.append(("current debt", v["debt_current"]))  # type: ignore[arg-type]
    if v.get("short_term_borrowings") is not None:
        components.append(("short-term borrowings / commercial paper", v["short_term_borrowings"]))  # type: ignore[arg-type]
    if not components:
        raise MissingInputs(["debt"])
    return sum(x for _, x in components), ["total debt = " + " + ".join(name for name, _ in components)]


def _debt_to_equity(v: Values, p: Period) -> tuple[float, list[str]]:
    (equity,) = _need(v, "stockholders_equity")
    debt, notes = total_debt(v)
    if equity < 0:
        notes.append("stockholders' equity is negative, so debt-to-equity is not meaningful")
    return F.ratio(debt, equity, "stockholders' equity"), notes


def _interest_coverage(v: Values, p: Period) -> tuple[float, list[str]]:
    (ie,) = _need(v, "interest_expense")
    oi, notes = _operating_income(v)
    return F.interest_coverage(oi, ie), notes + ["EBIT approximated by operating income"]


def _return_on(income_field: str, base_field: str) -> Compute:
    def compute(v: Values, p: Period) -> tuple[float, list[str]]:
        income, closing = _need(v, income_field, base_field)
        opening = v.get(f"{base_field}@opening")
        notes = []
        if opening is None:
            notes.append(f"opening {base_field.replace('_', ' ')} unavailable; used period-end balance")
        if p.type == "quarter":
            notes.append("quarterly figure annualised (x365/days)")
        return F.return_on_pct(income, F.average(opening, closing), _flow_days(p)), notes
    return compute


def _flow_days(p: Period) -> int:
    """Days a flow figure covers: a fiscal year counts as 365 (52/53-week years are not rescaled); quarters use
    their actual length so they annualise correctly."""
    return p.days if p.type == "quarter" else 365


def _dso(v: Values, p: Period) -> tuple[float, list[str]]:
    ar, rev = _need(v, "accounts_receivable", "revenue")
    days = _flow_days(p)
    return F.days_sales_outstanding(ar, rev, days), [f"period-end receivables over {days}-day revenue"]


def _inventory_turnover(v: Values, p: Period) -> tuple[float, list[str]]:
    cogs, closing = _need(v, "cost_of_revenue", "inventory")
    opening = v.get("inventory@opening")
    notes = [] if opening is not None else ["opening inventory unavailable; used period-end balance"]
    if p.type == "quarter":
        notes.append("quarterly cost of revenue annualised (x365/days)")
    return F.inventory_turnover(cogs, F.average(opening, closing), _flow_days(p)), notes


I = Input
REGISTRY: dict[str, Metric] = {
    m.name: m
    for m in [
        Metric("revenue_growth_yoy", "Revenue growth versus the same period a year earlier", "percent",
               "(revenue - revenue_prior_year) / |revenue_prior_year|",
               (I("revenue"), I("revenue", "prior_year")), _growth("revenue", "prior_year")),
        Metric("revenue_growth_qoq", "Revenue growth versus the immediately preceding quarter", "percent",
               "(revenue - revenue_prior_quarter) / |revenue_prior_quarter|",
               (I("revenue"), I("revenue", "prior_quarter")), _growth("revenue", "prior_quarter"),
               period_types=("quarter",)),
        Metric("gross_margin", "Gross profit as a share of revenue", "percent",
               "gross_profit / revenue (gross_profit = revenue - cost_of_revenue if not tagged)",
               (I("revenue"), I("gross_profit", required=False), I("cost_of_revenue", required=False)), _gross_margin),
        Metric("operating_margin", "Operating income as a share of revenue", "percent",
               "operating_income / revenue (operating income = gross profit - operating expenses if not tagged)",
               OPERATING_INCOME_INPUTS, _operating_margin),
        Metric("net_margin", "Net income as a share of revenue", "percent",
               "net_income / revenue", (I("net_income"), I("revenue")), _margin("net_income")),
        Metric("eps_growth_yoy", "Diluted EPS growth versus the same period a year earlier", "percent",
               "(eps_diluted - eps_diluted_prior_year) / |eps_diluted_prior_year|",
               (I("eps_diluted"), I("eps_diluted", "prior_year")), _growth("eps_diluted", "prior_year")),
        Metric("eps_growth_qoq", "Diluted EPS growth versus the preceding quarter", "percent",
               "(eps_diluted - eps_diluted_prior_quarter) / |eps_diluted_prior_quarter|",
               (I("eps_diluted"), I("eps_diluted", "prior_quarter")), _growth("eps_diluted", "prior_quarter"),
               period_types=("quarter",)),
        Metric("free_cash_flow", "Operating cash flow minus capital expenditures", "usd",
               "operating_cash_flow - capital_expenditures",
               (I("operating_cash_flow"), I("capital_expenditures")), _fcf),
        Metric("fcf_margin", "Free cash flow as a share of revenue", "percent",
               "(operating_cash_flow - capital_expenditures) / revenue",
               (I("operating_cash_flow"), I("capital_expenditures"), I("revenue")), _fcf_margin),
        Metric("current_ratio", "Current assets divided by current liabilities", "ratio",
               "current_assets / current_liabilities", (I("current_assets"), I("current_liabilities")), _current_ratio),
        Metric("debt_to_equity", "Total debt divided by stockholders' equity", "ratio",
               "(long_term_debt + current_debt + short_term_borrowings) / stockholders_equity",
               (I("stockholders_equity"), I("long_term_debt_noncurrent", required=False),
                I("long_term_debt_total", required=False), I("debt_current", required=False),
                I("short_term_borrowings", required=False)), _debt_to_equity),
        Metric("interest_coverage", "Operating income divided by interest expense", "times",
               "operating_income / |interest_expense|", (*OPERATING_INCOME_INPUTS, I("interest_expense")), _interest_coverage),
        Metric("roe", "Return on average stockholders' equity (annualised for quarters)", "percent",
               "net_income * 365/days / avg(equity_opening, equity_closing)",
               (I("net_income"), I("stockholders_equity"), I("stockholders_equity", "opening", required=False)),
               _return_on("net_income", "stockholders_equity")),
        Metric("roa", "Return on average total assets (annualised for quarters)", "percent",
               "net_income * 365/days / avg(assets_opening, assets_closing)",
               (I("net_income"), I("total_assets"), I("total_assets", "opening", required=False)),
               _return_on("net_income", "total_assets")),
        Metric("dso", "Days sales outstanding", "days",
               "accounts_receivable / revenue * days_in_period", (I("accounts_receivable"), I("revenue")), _dso),
        Metric("inventory_turnover", "Cost of revenue over average inventory (annualised for quarters)", "times",
               "cost_of_revenue * 365/days / avg(inventory_opening, inventory_closing)",
               (I("cost_of_revenue"), I("inventory"), I("inventory", "opening", required=False)), _inventory_turnover),
    ]
}

METRIC_NAMES = tuple(REGISTRY)


# ---------------------------------------------------------------------- evaluation


@dataclass
class ResolvedInput:
    key: str
    fact: Fact
    period: Period


@dataclass
class MetricValue:
    metric: Metric
    period: Period
    value: float
    inputs: list[ResolvedInput]
    notes: list[str]


def _target_period(facts: CompanyFacts, period: Period, offset: Offset) -> Period:
    if offset == "current":
        return period
    if offset == "opening":
        return facts.shift(period, "prior_quarter" if period.type == "quarter" else "prior_year")
    return facts.shift(period, offset)


def evaluate(facts: CompanyFacts, metric: Metric, period: Period) -> MetricValue:
    if period.type not in metric.period_types:
        raise DataError(
            "unsupported_period_type",
            f"{metric.name} is only defined for {', '.join(metric.period_types)} periods.",
        )
    values: Values = {}
    resolved: list[ResolvedInput] = []
    missing: list[str] = []
    tried: dict[str, list[str]] = {}
    targets: dict[str, Period | None] = {}
    for inp in metric.inputs:
        try:
            targets[inp.key] = _target_period(facts, period, inp.offset)
        except DataError:
            targets[inp.key] = None

    # A field needed for several periods (growth, opening balances) is read from one filing when possible.
    same_filing: dict[str, tuple[str, str]] = {}
    by_field: dict[str, list[Input]] = {}
    for inp in metric.inputs:
        if targets[inp.key] is not None:
            by_field.setdefault(inp.field, []).append(inp)
    for field_name, inps in by_field.items():
        if len(inps) > 1:
            hit = facts.common_filing(CONCEPTS[field_name], [targets[i.key] for i in inps], inps[0].kind,
                                      unit="USD/shares" if field_name in PER_SHARE else "USD")
            if hit:
                same_filing[field_name] = hit

    for inp in metric.inputs:
        target = targets[inp.key]
        if target is None:
            if inp.required:
                missing.append(f"{inp.field} ({inp.offset.replace('_', ' ')} period not in filings)")
            values[inp.key] = None
            continue
        unit = "USD/shares" if inp.field in PER_SHARE else "USD"
        fact = None
        if inp.field in same_filing:
            concept, accession = same_filing[inp.field]
            fact = facts.lookup([concept], target, inp.kind, unit=unit, accession=accession)
        if fact is None:
            fact = facts.lookup(CONCEPTS[inp.field], target, inp.kind, unit=unit, additive=inp.field not in PER_SHARE)
        values[inp.key] = fact.value if fact else None
        if fact:
            resolved.append(ResolvedInput(inp.key, fact, target))
        elif inp.required:
            missing.append(inp.field if target is period else f"{inp.field} ({target.label})")
            tried[inp.field] = list(CONCEPTS[inp.field])

    def missing_error(fields: list[str]) -> DataError:
        hint = None
        if any(f.startswith("eps_diluted") for f in fields) and period.type == "quarter":
            hint = "Per-share values are not derived from year-to-date totals; Q4 EPS is often only reported for the full year."
        names = ", ".join(f.replace("_", " ") for f in fields)
        return DataError(
            "missing_data",
            f"{names} not reported by {facts.name} for {period.label}; cannot compute {metric.name}.",
            metric=metric.name,
            period=period.label,
            missing=fields,
            concepts_tried=tried or {f: list(CONCEPTS[f]) for f in fields if f in CONCEPTS} or None,
            hint=hint,
        )

    if missing:
        raise missing_error(missing)
    try:
        value, notes = metric.compute(values, period)
    except MissingInputs as exc:
        raise missing_error(exc.fields) from None
    except F.FormulaError as exc:
        raise DataError("undefined_metric", f"{metric.name} is undefined for {period.label}: {exc}.", metric=metric.name) from None
    notes = list(notes) + [r.fact.derived for r in resolved if r.fact.derived]
    for r in resolved:
        if r.period is not period and r.fact.start and r.period.type == "quarter" and abs(r.period.days - period.days) > 7:
            notes.append(f"periods differ in length: {period.label} is {period.days} days, {r.period.label} is "
                         f"{r.period.days} days (e.g. a 16-week vs 12-week fiscal quarter); compare with care")
            break
    return MetricValue(metric, period, value, resolved, notes)
