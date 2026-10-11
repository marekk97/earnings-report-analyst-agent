"""get_segment_data: revenue and operating income by business segment, geography or product line."""

from __future__ import annotations

from datetime import date
from typing import Literal

from langchain_core.tools import BaseTool, tool

from ..metrics import formulas as F
from ..provenance import DataError
from ..sec.segments import parse_segments, summarise
from ..sec.xbrl import _days, _is_annual, _is_quarter
from ..session import AnalystSession
from .common import fmt_usd, tool_guard


def _pick(parsed: dict, period_type: str) -> dict | None:
    want = _is_quarter if period_type == "quarter" else _is_annual
    for row in summarise(parsed):
        if want(_days(date.fromisoformat(row["start"]), date.fromisoformat(row["end"]))):
            return row
    return None


def build(session: AnalystSession) -> list[BaseTool]:
    edgar = session.edgar

    @tool("get_segment_data")
    @tool_guard(session, "get_segment_data")
    def get_segment_data(
        ticker: str,
        dimension: Literal["segment", "geography", "product"] = "segment",
        period: str = "latest",
        period_type: Literal["quarter", "annual"] = "quarter",
    ) -> dict:
        """Break out revenue and operating income by reportable segment, geography or product line.

        Data comes from the XBRL of the 10-Q/10-K for the period, with the prior-year comparison from the same
        filing and year-over-year growth computed per member.

        Args:
            ticker: Company ticker or name.
            dimension: "segment" (reportable segments), "geography" or "product".
            period: "latest", a fiscal label like "Q2 FY2025" / "FY2024", or a period-end date.
            period_type: "quarter" or "annual" (used when period is "latest").
        """
        company = edgar.resolve_company(ticker)
        facts = edgar.facts(company)
        p = facts.resolve_period(period, period_type)
        form = "10-K" if (p.type == "annual" or p.fiscal_quarter == 4) else "10-Q"
        filing = edgar.periodic_filing(company, form, p.end)
        xml = edgar.client.get_text(edgar.xbrl_instance_url(filing))
        parsed = parse_segments(xml, dimension, p.end.isoformat())
        # A fiscal Q4 lives in the 10-K, which usually tags only full-year segment figures.
        current = _pick(parsed, p.type)
        if current is None:
            raise DataError(
                "segments_not_reported",
                f"{form} for {p.label} tags {dimension} data only for other period lengths "
                f"(10-Ks usually report segments for the full year; try period_type='annual').",
            )
        notes = []
        prior = None
        try:
            prior_p = facts.shift(p, "prior_year")
            prior = _pick(parse_segments(xml, dimension, prior_p.end.isoformat()), p.type)
        except DataError:
            notes.append("prior-year segment comparatives not found in this filing")
        prior_by = {m["xbrl_member"]: m for m in (prior["members"] if prior else [])}
        total_rev = sum(m["revenue"] or 0 for m in current["members"] if not m.get("is_subtotal"))
        members = []
        for m in current["members"]:
            row = {
                "name": m["member"],
                "revenue": m["revenue"],
                "revenue_display": fmt_usd(m["revenue"]) if m["revenue"] is not None else None,
                "share_of_listed_revenue_pct": round(m["revenue"] / total_rev * 100, 2)
                if m["revenue"] and total_rev and not m.get("is_subtotal") else None,
                "operating_income": m["operating_income"],
                "operating_income_display": fmt_usd(m["operating_income"]) if m["operating_income"] is not None else None,
                "profit_concept": m["profit_concept"],
            }
            if m.get("is_subtotal"):
                row["is_subtotal"] = True  # sum of other listed members; not an additional segment
            pm = prior_by.get(m["xbrl_member"])
            if pm:
                row["revenue_prior_year"] = pm["revenue"]
                row["operating_income_prior_year"] = pm["operating_income"]
                if m["revenue"] is not None and pm["revenue"]:
                    row["revenue_growth_yoy_pct"] = round(F.growth_pct(m["revenue"], pm["revenue"]), 2)
                if m["operating_income"] is not None and pm["operating_income"]:
                    row["operating_income_growth_yoy_pct"] = round(F.growth_pct(m["operating_income"], pm["operating_income"]), 2)
            members.append(row)
        if not any(m["operating_income"] is not None for m in members):
            notes.append("segment profit is not tagged in a recognised concept for this dimension")
        fiscal = p.label
        prov = {
            "form": filing.form,
            "fiscal_period": fiscal,
            "filing_date": filing.filing_date,
            "accession": filing.accession,
            "url": filing.index_url,
            "xbrl_axis": parsed["axis"],
        }
        session.ledger.add_source({"kind": f"segment_data:{dimension}", **prov})
        out = {
            "ok": True,
            "company": facts.name,
            "dimension": dimension,
            "period": {"label": fiscal, "start": current["start"], "end": current["end"]},
            "members": members,
            "notes": notes,
            "provenance": prov,
            "citation": f"{filing.form}, {fiscal}, filed {filing.filing_date}",
        }
        session.ledger.segments.append(out)
        # The model gets display strings and growth; raw values and XBRL details stay in the ledger.
        keep = ("name", "revenue_display", "revenue_growth_yoy_pct", "share_of_listed_revenue_pct",
                "operating_income_display", "operating_income_growth_yoy_pct", "is_subtotal")
        return {**{k: v for k, v in out.items() if k not in ("members", "provenance")},
                "members": [{k: m[k] for k in keep if m.get(k) is not None} for m in members]}

    return [get_segment_data]
