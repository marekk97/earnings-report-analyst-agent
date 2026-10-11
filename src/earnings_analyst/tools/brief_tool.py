"""save_brief: writes the Markdown brief plus a machine-readable JSON block built from the tool ledger.

The JSON is assembled from what the tools actually returned, not from the model's prose, so downstream
users (and the eval harness) can check every figure against its source filing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from .. import grounding
from ..session import AnalystSession, utcnow
from .common import tool_guard

ADVICE_RE = re.compile(
    r"\b(we|i) (recommend|suggest) (buying|selling|holding)|\b(strong )?(buy|sell|hold) (rating|recommendation)|"
    r"\bprice target\b|\bshould (buy|sell|short)\b|\b(undervalued|overvalued)\b",
    re.I,
)
DISCLAIMER = (
    "*Generated automatically from SEC EDGAR filings. Figures are computed from XBRL data and cited to their "
    "source filing. This is not investment advice.*"
)


# Models often emit non-breaking hyphens and narrow spaces; normalise them so citations stay greppable.
_UNICODE_FIXES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2212": "-",
                                "\u00a0": " ", "\u202f": " ", "\u2009": " "})


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-") or "brief"


def tool_numbers(session: AnalystSession) -> list[float]:
    led = session.ledger
    nums = list(grounding.numbers_from_obj([led.metrics, led.histories, led.segments]))
    for call in led.python_calls:
        nums.extend(q.value for q in grounding.extract_numbers(call.get("stdout", ""), require_unit=False))
    return nums


def compact_metric(m: dict) -> dict:
    out = {k: m.get(k) for k in ("metric", "unit", "formula", "period", "value", "display", "citation", "provenance", "notes")}
    out["inputs"] = [{"name": i["name"], "value": i["value"], "provenance": i["provenance"]} for i in m.get("inputs", [])]
    comp = m.get("comparison")
    if comp:
        out["comparison"] = {
            k: comp.get(k)
            for k in ("basis", "period", "value", "display", "change_pp", "change", "change_pct", "change_display", "citation", "error", "message")
            if comp.get(k) is not None
        }
    return out


def build_payload(session: AnalystSession, ticker: str, period_label: str, report: grounding.GroundingReport) -> dict:
    led = session.ledger
    company = next((m.get("company") for m in led.metrics), None)
    return {
        "schema": "earnings-brief/v1",
        "run_id": session.run_id,
        "generated_at": utcnow(),
        "question": session.question,
        "company": company,
        "ticker": ticker.upper(),
        "period": period_label,
        "metrics": [compact_metric(m) for m in led.metrics],
        "metric_histories": [
            {"metric": h["metric"], "unit": h["unit"], "points": h["points"]} for h in led.histories
        ],
        "segments": [
            {"dimension": s["dimension"], "period": s["period"], "citation": s["citation"],
             "members": [{k: v for k, v in mm.items() if not k.endswith("_display")} for mm in s["members"]]}
            for s in led.segments
        ],
        "sources": led.sources,
        "python_fallbacks": [
            {"metric_name": c.get("metric_name"), "purpose": c.get("purpose"), "ok": c.get("ok")} for c in led.python_calls
        ],
        "data_gaps": [
            {"tool": e["tool"], "error": e.get("error"), "message": e.get("message")} for e in led.errors
        ],
        "grounding": report.to_dict(),
    }


def build(session: AnalystSession) -> list[BaseTool]:
    @tool("save_brief")
    @tool_guard(session, "save_brief")
    def save_brief(markdown: str, ticker: str, period_label: str, title: str = "") -> dict:
        """Write the final one-page Markdown brief. A JSON block with every computed metric, its inputs and
        their source filings is generated from your tool results and appended automatically.

        Returns the file path plus warnings: numbers in the brief that do not match any tool result, or
        language that reads like investment advice. If there are warnings, fix the brief and call save_brief again.

        Args:
            markdown: The complete brief in Markdown (title, bottom line, metrics table, drivers, outlook, caveats).
            ticker: Company ticker.
            period_label: Fiscal period the brief covers, e.g. "Q1 FY2027".
            title: Optional title; defaults to the first Markdown heading.
        """
        body = markdown.strip().translate(_UNICODE_FIXES)
        if title and not body.startswith("#"):
            body = f"# {title}\n\n{body}"
        if "not investment advice" not in body.lower():
            body += "\n\n---\n" + DISCLAIMER

        report = grounding.check(body, tool_numbers(session), session.ledger.text_snippets)
        payload = build_payload(session, ticker, period_label, report)

        out_dir = Path(session.settings.brief_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{_slug(ticker.upper())}_{_slug(period_label)}_{session.run_id}"
        md_path = out_dir / f"{stem}.md"
        json_path = out_dir / f"{stem}.json"
        json_text = json.dumps(payload, indent=2, default=str)
        md_path.write_text(f"{body}\n\n## Data\n\n```json\n{json_text}\n```\n", encoding="utf-8")
        json_path.write_text(json_text + "\n", encoding="utf-8")
        session.ledger.brief_path = str(md_path)

        warnings = []
        if report.ungrounded:
            warnings.append(
                "These numbers do not match any tool result or fetched filing text: "
                + ", ".join(report.ungrounded[:15])
                + ". Use the exact values returned by the tools (or remove them)."
            )
        from ..evals import gives_advice

        if gives_advice(body):
            warnings.append("The brief contains investment-advice language (buy/sell/price target/valuation calls). Remove it.")
        if not session.ledger.metrics:
            warnings.append("No calculate_metric results were recorded; the brief has no computed metrics.")
        return {
            "ok": True,
            "path": str(md_path),
            "json_path": str(json_path),
            "metrics_recorded": len(session.ledger.metrics),
            "sources_recorded": len(session.ledger.sources),
            "grounding": report.to_dict(),
            "warnings": warnings,
        }

    return [save_brief]
