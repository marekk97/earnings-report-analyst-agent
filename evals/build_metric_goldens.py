"""Builds evals/metric_goldens.yaml: expected metric values computed INDEPENDENTLY of the agent's XBRL pipeline.

Why a separate implementation: the agent's pipeline (sec/xbrl.py) merges every filing, de-duplicates by period,
prefers the latest-filed (restated) value and derives missing quarters. This script does none of that. It opens
ONE filing (the 10-K for the fiscal year, or the 10-Q for the quarter) and reads the figures exactly as that
filing reported them: current period = latest end date in the filing, comparison = the period one year earlier
in the same filing. If both implementations agree, the golden is trustworthy; if not, the case is reported so a
human can decide (usually a later restatement, sometimes a pipeline bug).

Only the XBRL concept names are shared with the pipeline (metrics.registry.CONCEPTS); the formulas are written
out here again on purpose.

Usage:  python evals/build_metric_goldens.py [--fy 2023] [--out evals/metric_goldens.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import yaml

from earnings_analyst.metrics.registry import CONCEPTS, REGISTRY, evaluate
from earnings_analyst.provenance import DataError
from earnings_analyst.sec.filings import EDGAR

COMPANIES = [
    # tech / platforms
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "ORCL", "ADBE", "CSCO", "INTC",
    # consumer / retail
    "NKE", "AMZN", "WMT", "COST", "HD", "LOW", "TGT", "SBUX", "MCD", "KO", "PEP", "PG",
    # healthcare
    "JNJ", "PFE", "UNH",
    # industrial / energy / media
    "CAT", "BA", "CVX", "DIS", "TSLA",
    # Exxon by CIK: in 2026 the XOM ticker moved to a new holding company (CIK 2115436) with almost no history.
    "34088",
]
ANNUAL_METRICS = [
    "revenue_growth_yoy", "gross_margin", "operating_margin", "net_margin", "eps_growth_yoy",
    "free_cash_flow", "fcf_margin", "current_ratio", "roa", "roe", "dso", "inventory_turnover",
]
QUARTER_METRICS = ["revenue_growth_yoy", "gross_margin", "net_margin"]
TOLERANCE = {"percent": 0.05, "ratio": 0.005, "days": 0.2, "times": 0.02, "usd": 5_000_000}


def _d(s: str) -> date:
    return date.fromisoformat(s)


def _back(d: date, years: int) -> date:
    """Same date `years` earlier (29 Feb -> 28 Feb); lookups allow +-10 days anyway."""
    try:
        return d.replace(year=d.year - years)
    except ValueError:
        return d.replace(year=d.year - years, day=28)


class FilingView:
    """All us-gaap facts of a single filing (one accession number)."""

    def __init__(self, raw: dict, accession: str):
        self.accession = accession
        self.facts: dict[str, dict[str, list[dict]]] = {}
        for concept, node in raw["facts"].get("us-gaap", {}).items():
            for unit, arr in node["units"].items():
                rows = [f for f in arr if f["accn"] == accession]
                if rows:
                    self.facts.setdefault(concept, {})[unit] = rows
        ends = [_d(f["end"]) for c in self.facts.values() for u in c.values() for f in u if f.get("start")]
        self.period_end = max(ends)
        meta = next(f for c in self.facts.values() for u in c.values() for f in u)
        self.form, self.filed, self.fy, self.fp = meta["form"], meta["filed"], meta["fy"], meta["fp"]

    def duration(self, field: str, years_back: int, kind: str, unit: str = "USD") -> float | None:
        lo, hi = (350, 380) if kind == "annual" else (80, 119)
        target = _back(self.period_end, years_back)
        for concept in CONCEPTS[field]:
            for f in self.facts.get(concept, {}).get(unit, []):
                if not f.get("start"):
                    continue
                days = (_d(f["end"]) - _d(f["start"])).days + 1
                if lo <= days <= hi and abs((_d(f["end"]) - target).days) <= 10:
                    return float(f["val"])
        return None

    def instant(self, field: str, years_back: int) -> float | None:
        target = _back(self.period_end, years_back)
        for concept in CONCEPTS[field]:
            for f in self.facts.get(concept, {}).get("USD", []):
                if not f.get("start") and abs((_d(f["end"]) - target).days) <= 10:
                    return float(f["val"])
        return None


class Q4View:
    """Fiscal Q4 read independently: full year from the 10-K minus nine months year-to-date from the Q3 10-Q."""

    def __init__(self, annual: FilingView, q3: FilingView):
        self.annual, self.q3 = annual, q3
        self.period_end, self.filed, self.form = annual.period_end, annual.filed, "10-K - Q3 10-Q"

    def duration(self, field: str, years_back: int, kind: str, unit: str = "USD") -> float | None:
        if unit != "USD":
            return None  # per-share values do not subtract
        fy = self.annual.duration(field, years_back, "annual")
        target = _back(self.q3.period_end, years_back)
        nine = None
        for concept in CONCEPTS[field]:
            for f in self.q3.facts.get(concept, {}).get("USD", []):
                if f.get("start") and 250 <= (_d(f["end"]) - _d(f["start"])).days + 1 <= 290 \
                        and abs((_d(f["end"]) - target).days) <= 10:
                    nine = float(f["val"])
                    break
            if nine is not None:
                break
        return None if fy is None or nine is None else fy - nine

    def instant(self, field: str, years_back: int) -> float | None:
        return None


def independent(view: FilingView, metric: str, kind: str) -> float | None:
    """Formulas written out independently of metrics/formulas.py."""
    d = lambda f, back=0: view.duration(f, back, kind)  # noqa: E731
    days = 365  # a fiscal year's flows are used as-is (no rescaling of 52/53-week years)
    rev = d("revenue")
    if metric == "revenue_growth_yoy":
        prev = d("revenue", 1)
        return None if not (rev and prev) else (rev / prev - 1) * 100
    if metric == "gross_margin":
        gp = d("gross_profit")
        if gp is None and rev is not None and d("cost_of_revenue") is not None:
            gp = rev - d("cost_of_revenue")
        return None if gp is None or not rev else gp / rev * 100
    if metric == "operating_margin":
        oi = d("operating_income")  # only the tagged figure: the pipeline's approximation is not golden material
        return None if oi is None or not rev else oi / rev * 100
    if metric == "net_margin":
        ni = d("net_income")
        return None if ni is None or not rev else ni / rev * 100
    if metric == "eps_growth_yoy":
        cur, prev = view.duration("eps_diluted", 0, kind, "USD/shares"), view.duration("eps_diluted", 1, kind, "USD/shares")
        return None if cur is None or not prev else (cur - prev) / abs(prev) * 100
    if metric in ("free_cash_flow", "fcf_margin"):
        cfo, capex = d("operating_cash_flow"), d("capital_expenditures")
        if cfo is None or capex is None:
            return None
        fcf = cfo - abs(capex)
        if metric == "free_cash_flow":
            return fcf
        return None if not rev else fcf / rev * 100
    if metric == "current_ratio":
        ca, cl = view.instant("current_assets", 0), view.instant("current_liabilities", 0)
        return None if ca is None or not cl else ca / cl
    if metric in ("roa", "roe"):
        field = "total_assets" if metric == "roa" else "stockholders_equity"
        ni, close, open_ = d("net_income"), view.instant(field, 0), view.instant(field, 1)
        if ni is None or close is None or open_ is None:
            return None
        return ni * 365 / days / ((close + open_) / 2) * 100
    if metric == "dso":
        ar = view.instant("accounts_receivable", 0)
        return None if ar is None or not rev else ar / rev * days
    if metric == "inventory_turnover":
        cogs, close, open_ = d("cost_of_revenue"), view.instant("inventory", 0), view.instant("inventory", 1)
        if cogs is None or close is None or open_ is None:
            return None
        return cogs * 365 / days / ((close + open_) / 2)
    return None


def find_filing(raw: dict, form: str, fy: int, fp: str) -> str | None:
    accns = {
        f["accn"]
        for node in raw["facts"].get("us-gaap", {}).values()
        for arr in node["units"].values()
        for f in arr
        if f.get("form") == form and f.get("fy") == fy and f.get("fp") == fp
    }
    return sorted(accns)[0] if len(accns) >= 1 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fy", default="2022,2023,2024", help="fiscal years for annual cases")
    ap.add_argument("--quarter-fy", type=int, default=2024, help="fiscal year whose four quarters are covered")
    ap.add_argument("--out", default="evals/metric_goldens.yaml")
    args = ap.parse_args()

    edgar = EDGAR()
    cases, disagreements, skipped = [], [], []
    for ticker in COMPANIES:
        company = edgar.resolve_company(ticker)
        ticker = company.ticker if not ticker.isdigit() else f"CIK{ticker}"
        facts = edgar.facts(company)
        raw = facts.raw
        jobs = [("annual", "10-K", int(fy), "FY", ANNUAL_METRICS) for fy in args.fy.split(",")]
        jobs += [("quarter", "10-Q", args.quarter_fy, fp, QUARTER_METRICS) for fp in ("Q1", "Q2", "Q3", "Q4")]
        for kind, form, fy, fp, metrics in jobs:
            if fp == "Q4":
                a, q3 = find_filing(raw, "10-K", fy, "FY"), find_filing(raw, "10-Q", fy, "Q3")
                if not (a and q3):
                    skipped.append(f"{ticker} Q4 fy{fy}: 10-K or Q3 10-Q not found")
                    continue
                view, accn = Q4View(FilingView(raw, a), FilingView(raw, q3)), f"{a} minus {q3}"
            else:
                accn = find_filing(raw, form, fy, fp)
                if not accn:
                    skipped.append(f"{ticker} {form} fy{fy} {fp}: filing not found")
                    continue
                view = FilingView(raw, accn)
            period_label = f"FY{fy}" if kind == "annual" else f"{fp} FY{fy}"
            for metric in metrics:
                expected = independent(view, metric, kind)
                case_id = f"{ticker.lower()}_{period_label.replace(' ', '').lower()}_{metric}"
                try:
                    p = facts.resolve_period(period_label, kind)
                    got = evaluate(facts, REGISTRY[metric], p).value
                except DataError as exc:
                    got = None
                    pipeline_err = exc.code
                else:
                    pipeline_err = None
                if expected is None:
                    # Not computable from the filing itself: only keep it if the pipeline also says data is missing.
                    if pipeline_err == "missing_data":
                        cases.append({"id": case_id, "category": "missing_data", "ticker": ticker, "metric": metric,
                                      "period": period_label, "period_type": kind, "expect_error": "missing_data",
                                      "source": f"{form} {accn}"})
                    else:
                        skipped.append(f"{case_id}: filing lacks inputs (pipeline: {got if got is not None else pipeline_err})")
                    continue
                tol = TOLERANCE[REGISTRY[metric].unit]
                case = {"id": case_id, "category": f"{kind}_metric", "ticker": ticker, "metric": metric,
                        "period": period_label, "period_type": kind, "expected": round(expected, 4),
                        "tolerance": tol, "source": f"{view.form} {accn} filed {view.filed}"}
                if got is None or abs(got - expected) > tol:
                    disagreements.append({**case, "pipeline": got if got is not None else pipeline_err})
                else:
                    cases.append(case)
        print(f"{ticker}: done", file=sys.stderr)

    header = (
        "# GENERATED by evals/build_metric_goldens.py - expected values read independently from each company's\n"
        f"# own 10-Ks (FY{args.fy}) and 10-Qs (Q1-Q3 FY{args.quarter_fy}; Q4 = 10-K minus Q3 year-to-date).\n"
        "# Cases where the independent reading and the agent's pipeline disagreed are NOT included\n"
        "# (see the disagreements printed by the build and saved in metric_goldens.build.json).\n"
    )
    Path(args.out).write_text(header + yaml.safe_dump({"tool_cases": cases}, sort_keys=False, width=140))
    report = {"cases": len(cases), "disagreements": disagreements, "skipped": skipped}
    Path(args.out).with_suffix(".build.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n{len(cases)} cases written to {args.out}; {len(disagreements)} disagreements; {len(skipped)} skipped",
          file=sys.stderr)
    for d in disagreements:
        print(f"  DISAGREE {d['id']}: independent={d['expected']} pipeline={d['pipeline']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
