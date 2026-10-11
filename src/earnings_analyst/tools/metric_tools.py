"""calculate_metric and get_metric_history: deterministic metrics from XBRL facts, with provenance."""

from __future__ import annotations

import difflib
from typing import Literal

from langchain_core.tools import BaseTool, tool

from ..metrics.registry import METRIC_NAMES, REGISTRY, Metric, MetricValue, evaluate
from ..provenance import DataError
from ..session import AnalystSession
from ..sec.xbrl import CompanyFacts
from .common import fmt_input, fmt_value, tool_guard

ALIASES = {
    "revenue_growth": "revenue_growth_yoy",
    "sales_growth": "revenue_growth_yoy",
    "eps_growth": "eps_growth_yoy",
    "fcf": "free_cash_flow",
    "debt_equity": "debt_to_equity",
    "d/e": "debt_to_equity",
    "days_sales_outstanding": "dso",
    "return_on_equity": "roe",
    "return_on_assets": "roa",
}

METRIC_HELP = "; ".join(f"{m.name} ({m.description})" for m in REGISTRY.values())


def resolve_metric(name: str) -> Metric:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    key = ALIASES.get(key, key)
    if key in REGISTRY:
        return REGISTRY[key]
    raise DataError(
        "unknown_metric",
        f"'{name}' is not in the metric registry. Use run_python for one-off calculations.",
        valid_metrics=list(METRIC_NAMES),
        closest=difflib.get_close_matches(key, METRIC_NAMES, n=3, cutoff=0.4) or None,
    )


def serialise(mv: MetricValue, facts: CompanyFacts) -> dict:
    inputs = []
    for r in mv.inputs:
        field = r.key.split("@")[0]
        inputs.append(
            {
                "name": r.key,
                "value": r.fact.value,
                "display": fmt_input(r.fact.value, field),
                "provenance": facts.provenance(r.fact, r.period).to_dict(),
            }
        )
    current = [i for i in inputs if "@" not in i["name"]] or inputs
    primary = current[0]["provenance"] if current else None
    return {
        "period": mv.period.to_dict(),
        "value": round(mv.value, 6),
        "display": fmt_value(mv.value, mv.metric.unit),
        "inputs": inputs,
        "notes": mv.notes,
        "provenance": primary,
        "citation": f"{primary['form']}, {primary['fiscal_period']}, filed {primary['filing_date']}" if primary else None,
    }


def for_model(full: dict) -> dict:
    """What the model sees: values, display strings, periods and one citation per period.

    The full result (every input's accession number, URL, XBRL concept, dates...) stays in the session ledger,
    from which save_brief builds the brief's JSON block and the grounding check takes its numbers.
    """
    def slim(part: dict) -> dict:
        return {
            "period": part["period"]["label"],
            "value": part["value"],
            "display": part["display"],
            "inputs": {i["name"]: i["display"] for i in part.get("inputs", [])},
            "citation": part.get("citation"),
        }

    out = {"ok": True, "ticker": full["ticker"], "metric": full["metric"], "unit": full["unit"], **slim(full)}
    if full.get("notes"):
        out["notes"] = full["notes"]
    comp = full.get("comparison")
    if comp and comp.get("ok", True):
        c = slim(comp)
        c.pop("inputs")
        out["comparison"] = {**c, "change": comp.get("change_display")}
        if comp.get("meaning"):
            out["comparison"]["meaning"] = comp["meaning"]
    elif comp:
        out["comparison"] = {"error": comp.get("message")}
    return out


def build(session: AnalystSession) -> list[BaseTool]:
    edgar = session.edgar

    @tool("calculate_metric")
    @tool_guard(session, "calculate_metric")
    def calculate_metric(
        ticker: str,
        metric: str,
        period: str = "latest",
        period_type: Literal["quarter", "annual"] = "quarter",
        compare_to: Literal["prior_year", "prior_quarter"] = "prior_year",
    ) -> dict:
        """Compute ONE registered financial metric from SEC XBRL data, with provenance for every input.

        Call exactly once per metric the question needs (the year-ago comparison is included by default, so never
        call the same metric twice). Values are deterministic; quote them as returned.

        Args:
            ticker: Company ticker (e.g. "NKE") or name.
            metric: One of: revenue_growth_yoy, revenue_growth_qoq, gross_margin, operating_margin, net_margin,
                eps_growth_yoy, eps_growth_qoq, free_cash_flow, fcf_margin, current_ratio, debt_to_equity,
                interest_coverage, roe, roa, dso, inventory_turnover.
            period: "latest", a fiscal label like "Q2 FY2025" or "FY2024", or a period-end date "2025-11-30".
            period_type: "quarter" or "annual" (used when period is "latest").
            compare_to: Basis of the comparison that is always included: "prior_year" (default) or "prior_quarter".
                For growth metrics the comparison is the growth rate of the earlier period (e.g. FY2023 revenue
                growth when asking for FY2024), so one call answers "compare growth in FY2024 with FY2023".
        """
        company = edgar.resolve_company(ticker)
        facts = edgar.facts(company)
        m = resolve_metric(metric)
        p = facts.resolve_period(period, period_type)
        current = evaluate(facts, m, p)
        out = {
            "ok": True,
            "company": facts.name,
            "ticker": company.ticker,
            "metric": m.name,
            "description": m.description,
            "unit": m.unit,
            "formula": m.formula,
            **serialise(current, facts),
        }
        is_growth = "growth" in m.name
        if compare_to not in ("prior_year", "prior_quarter"):
            compare_to = "prior_year"  # tolerate legacy/invalid values from the model
        if m.name.endswith("_qoq"):
            compare_to = "prior_quarter"
        if True:
            try:
                prior_period = facts.shift(p, compare_to)
                prior = serialise(evaluate(facts, m, prior_period), facts)
                change = current.value - prior["value"]
                comp = {"basis": compare_to, **prior}
                if is_growth:
                    comp["meaning"] = f"the growth rate one period earlier ({prior_period.label}); change = acceleration"
                if m.unit == "percent":
                    comp["change_pp"] = round(change, 4)
                    comp["change_display"] = f"{change:+.1f} pp"
                else:
                    comp["change"] = round(change, 6)
                    comp["change_display"] = fmt_value(change, m.unit)
                    if prior["value"]:
                        pct = change / abs(prior["value"]) * 100
                        comp["change_pct"] = round(pct, 4)
                        comp["change_display"] += f" ({pct:+.1f}%)"
                out["comparison"] = comp
            except DataError as exc:
                out["comparison"] = exc.to_dict()
        # One ledger entry per company, metric and period; a repeated call replaces the earlier result.
        key = (company.ticker, m.name, p.label)
        session.ledger.metrics[:] = [
            x for x in session.ledger.metrics if (x["ticker"], x["metric"], x["period"]["label"]) != key
        ] + [out]
        return for_model(out)

    @tool("get_metric_history")
    @tool_guard(session, "get_metric_history")
    def get_metric_history(
        ticker: str,
        metric: str,
        periods: int = 8,
        period_type: Literal["quarter", "annual"] = "quarter",
        end_period: str = "latest",
    ) -> dict:
        """Return one registered metric for a run of consecutive periods ending at `end_period`, oldest first.

        Use for trend questions ("has gross margin been improving?") and for "each quarter of fiscal 2024"
        (end_period="Q4 FY2024", periods=4) or "fiscal 2022-2024" (end_period="FY2024", periods=3,
        period_type="annual"). Each point carries its fiscal period label and citation; quote the labels as returned.
        Periods where data is missing are listed with the reason instead of being dropped.

        Args:
            ticker: Company ticker (e.g. "NKE") or name.
            metric: A registered metric name (same list as calculate_metric).
            periods: Number of periods, 2-12.
            period_type: "quarter" or "annual".
            end_period: Last period of the run: "latest" (default), "Q4 FY2024", "FY2024" or a period-end date.
        """
        company = edgar.resolve_company(ticker)
        facts = edgar.facts(company)
        m = resolve_metric(metric)
        n = max(2, min(int(periods), 12))
        all_periods = facts.periods(period_type)
        last = facts.resolve_period(end_period, period_type)
        if last.type != period_type:  # e.g. end_period="FY2024" with period_type="quarter": use its Q4
            last = facts.period_at(last.end, period_type) or last
        upto = [x for x in all_periods if x.end <= last.end]
        points = []
        for p in upto[-n:]:
            try:
                mv = evaluate(facts, m, p)
                s = serialise(mv, facts)
                points.append({"period": p.label, "end": p.end.isoformat(), "value": s["value"], "display": s["display"],
                               "citation": s["citation"], **({"notes": mv.notes} if mv.notes else {})})
            except DataError as exc:
                points.append({"period": p.label, "end": p.end.isoformat(), "value": None, "error": exc.message})
        if not any(pt["value"] is not None for pt in points):
            raise DataError("missing_data", f"No {m.name} values could be computed for {facts.name}.", periods=points)
        out = {"ok": True, "company": facts.name, "ticker": company.ticker, "metric": m.name, "unit": m.unit,
               "formula": m.formula, "points": points}
        session.ledger.histories.append(out)
        return out

    return [calculate_metric, get_metric_history]
