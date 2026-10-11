"""Metric-accuracy suite: are the numbers the agent states in its brief correct?

For every expected fact the scorer looks for a table row or sentence in the brief that mentions the fact (an alias
such as "gross margin", "Americas" or "PepsiCo", plus the period when several periods are in play) and checks
whether that row/sentence contains a number equal to the expected value at the precision the brief displays it.

Each fact gets one of three outcomes:
  correct    - stated with the right value
  wrong      - mentioned, but none of the numbers next to it is right (the candidates are recorded)
  missing    - not mentioned at all
Separately, `computed` records whether any tool result held the right value, which tells calculation errors
(tool wrong) apart from reporting errors (tool right, brief wrong or silent).
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

from . import grounding
from .config import get_settings
from .metrics.registry import REGISTRY

METRIC_ALIASES = {
    "revenue_growth_yoy": ["revenue", "sales", "growth"],
    "revenue_growth_qoq": ["revenue", "sales", "sequential"],
    "gross_margin": ["gross margin"],
    "operating_margin": ["operating margin"],
    "net_margin": ["net margin", "net income margin", "net profit margin", "profit margin"],
    "eps_growth_yoy": ["eps", "earnings per share"],
    "free_cash_flow": ["free cash flow", "fcf"],
    "fcf_margin": ["fcf margin", "free cash flow margin", "free-cash-flow margin", "fcf"],
    "current_ratio": ["current ratio"],
    "debt_to_equity": ["debt-to-equity", "debt to equity", "d/e"],
    "interest_coverage": ["interest coverage"],
    "roe": ["return on equity", "roe"],
    "roa": ["return on assets", "roa"],
    "dso": ["days sales outstanding", "dso", "receivable"],
    "inventory_turnover": ["inventory turnover", "turnover"],
}
UNIT_TOLERANCE = {"percent": 0.06, "pp": 0.06, "ratio": 0.006, "days": 0.06, "times": 0.006}
QUARTER_WORDS = {1: "first", 2: "second", 3: "third", 4: "fourth"}


# ---------------------------------------------------------------------- dataset


def resolve_facts(case: dict, goldens: dict[str, dict]) -> list[dict]:
    """Turn a case's fact specs into {label, value, unit, aliases, hints}."""
    out = []
    for spec in case["facts"]:
        if "golden" in spec or "change_of" in spec:
            refs = [spec["golden"]] if "golden" in spec else spec["change_of"]
            g = [goldens[r] for r in refs]
            metric = g[0]["metric"]
            unit = REGISTRY[metric].unit
            if "change_of" in spec:
                value = g[0]["expected"] - g[1]["expected"]
                unit = "pp" if unit == "percent" else unit
                label = f"{metric} change {g[1]['period']}->{g[0]['period']}"
            else:
                value = g[0]["expected"]
                label = f"{g[0]['ticker']} {metric} {g[0]['period']}"
            aliases = spec.get("aliases") or METRIC_ALIASES.get(metric, [metric.replace("_", " ")])
            hints = period_hints(g[0]["period"]) if spec.get("period_hint") else []
        else:
            value, unit = float(spec["value"]), spec["unit"]
            aliases = spec["aliases"]
            label = f"{aliases[0]} ({spec.get('source', 'literal')})"
            hints = period_hints(spec["period"]) if spec.get("period_hint") and spec.get("period") else []
        out.append({"label": label, "value": float(value), "unit": unit, "aliases": aliases, "hints": hints})
    return out


def period_hints(label: str) -> list[str]:
    """Ways a brief may name a period: 'Q2 FY2024' -> Q2, second quarter, 2Q; 'FY2022' -> 2022, FY22."""
    m = re.match(r"Q([1-4]) FY(\d{4})", label)
    if m:
        q = int(m.group(1))
        return [f"Q{q}", f"{QUARTER_WORDS[q]} quarter", f"{q}Q"]
    m = re.match(r"FY(\d{4})", label)
    if m:
        y = m.group(1)
        return [y, f"FY{y[2:]}", f"FY {y}"]
    return [label]


# ---------------------------------------------------------------------- scoring


def contexts(brief: str) -> list[tuple[str, str]]:
    """(where to look for aliases/hints, where to look for numbers): table rows carry their header row."""
    body = brief.split("\n## Data\n")[0].replace("**", "").replace("__", "")
    out: list[tuple[str, str]] = []
    header = ""
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("|"):
            if re.match(r"^\|[\s:|-]+\|?$", s):
                continue  # separator row
            if not header:
                header = s
                out.append((s, s))
            else:
                out.append((f"{header} {s}", s))
            continue
        header = ""
        for sentence in re.split(r"(?<=[.!?;])\s+", s):
            if sentence:
                out.append((sentence, sentence))
    return out


def _contains(text: str, needles: list[str]) -> bool:
    """Whole-word match; a numeric needle such as "2023" only needs non-digit neighbours (so "FY2023" counts)."""
    low = re.sub(r"[-‐‑–]", " ", text.lower())  # "Gross-margin" counts as "gross margin"
    for n in needles:
        n = re.sub(r"[-‐‑–]", " ", n.lower())
        before = r"(?<!\d)" if n[0].isdigit() else r"(?<![a-z])"
        after = r"(?!\d)" if n[-1].isdigit() else r"(?![a-z])"
        if re.search(before + re.escape(n) + after, low):
            return True
    return False


def _tolerance(fact: dict) -> float:
    if fact["unit"] == "usd":
        return 0.006 * abs(fact["value"])
    return UNIT_TOLERANCE.get(fact["unit"], 0.01)


def score_fact(fact: dict, ctxs: list[tuple[str, str]], tool_numbers: list[float]) -> dict:
    tol = _tolerance(fact)
    target = abs(fact["value"])
    candidates: list[str] = []
    mentioned = False
    for where, numbers_text in ctxs:
        if not _contains(where, fact["aliases"]):
            continue
        if fact["hints"] and not _contains(where, fact["hints"]):
            continue
        mentioned = True
        for q in grounding.extract_numbers(numbers_text, require_unit=False):
            if abs(abs(q.value) - target) <= max(tol, q.tolerance):
                return {**fact, "status": "correct", "stated": q.text,
                        "computed": _in_tools(fact, tool_numbers)}
            candidates.append(q.text)
    return {**fact, "status": "wrong" if mentioned else "missing", "candidates": candidates[:8],
            "computed": _in_tools(fact, tool_numbers)}


def _in_tools(fact: dict, tool_numbers: list[float]) -> bool:
    tol = _tolerance(fact)
    return any(abs(abs(x) - abs(fact["value"])) <= tol for x in tool_numbers)


def score_case(case: dict, facts: list[dict], brief: str, tool_numbers: list[float]) -> dict:
    ctxs = contexts(brief)
    results = [score_fact(f, ctxs, tool_numbers) for f in facts]
    correct = sum(r["status"] == "correct" for r in results)
    return {
        "id": case["id"],
        "category": case.get("category", "uncategorised"),
        "question": case["question"],
        "facts": len(results),
        "correct": correct,
        "score": round(correct / len(results), 3) if results else 0.0,
        "computed_correct": sum(r["computed"] for r in results),
        "results": results,
    }


# ---------------------------------------------------------------------- suite


def run_accuracy_suite(cases: list[dict], goldens: dict[str, dict], model: str | None) -> dict:
    from .agent import run
    from .tools.brief_tool import tool_numbers

    settings = get_settings()
    if model:
        settings = replace(settings, model=model)
    out = []
    for case in cases:
        print(f"  running {case['id']} ...", file=sys.stderr, flush=True)
        facts = resolve_facts(case, goldens)
        started = time.monotonic()
        try:
            res = run(case["question"], settings=settings)
            brief = Path(res.brief_path).read_text(encoding="utf-8") if res.brief_path else ""
            scored = score_case(case, facts, brief, tool_numbers(res.session))
            scored.update({"brief_path": res.brief_path, "log_path": res.log_path, "model_calls": res.model_calls,
                           "seconds": res.seconds})
            if not brief:
                scored["error"] = "no brief saved"
        except Exception as exc:  # a crash scores zero for every fact
            scored = {"id": case["id"], "category": case.get("category"), "question": case["question"],
                      "facts": len(facts), "correct": 0, "score": 0.0, "computed_correct": 0,
                      "results": [{**f, "status": "missing", "computed": False} for f in facts],
                      "error": repr(exc), "seconds": round(time.monotonic() - started, 1)}
        out.append(scored)
    total = sum(c["facts"] for c in out)
    return {
        "suite": "accuracy",
        "model": settings.model,
        "cases": out,
        "score": round(sum(c["correct"] for c in out) / total, 3) if total else 0.0,  # share of facts stated correctly
        "case_score": round(mean(c["score"] for c in out), 3) if out else 0.0,
        "computed_score": round(sum(c["computed_correct"] for c in out) / total, 3) if total else 0.0,
    }


def report(res: dict, prev: dict | None) -> None:
    prev_cases = {c["id"]: c for c in (prev or {}).get("cases", [])}
    print(f"\n== accuracy suite ({res['model']}) - {len(res['cases'])} questions, "
          f"{sum(c['facts'] for c in res['cases'])} facts ==")
    for c in res["cases"]:
        was = prev_cases.get(c["id"], {}).get("score")
        delta = f" ({c['score'] - was:+.2f})" if was is not None else ""
        print(f"  {c['score']:.2f}{delta}  [{c['category']}] {c['id']}  {c['correct']}/{c['facts']} correct"
              + (f"  ERROR: {c['error']}" if c.get("error") else ""))
        for r in c["results"]:
            if r["status"] != "correct":
                where = "tools had it" if r["computed"] else "tools did NOT have it"
                seen = f", saw {r.get('candidates')}" if r.get("candidates") else ""
                print(f"        {r['status'].upper():7s} {r['label']}: expected {r['value']:,.4g} {r['unit']} ({where}{seen})")
    groups: dict[str, list[dict]] = {}
    for c in res["cases"]:
        groups.setdefault(c["category"], []).append(c)
    print("  by category (facts stated correctly):")
    for cat, cs in sorted(groups.items()):
        n = sum(c["facts"] for c in cs)
        ok = sum(c["correct"] for c in cs)
        print(f"    {cat:15s} {ok / n:.2f}  ({ok}/{n} facts, {len(cs)} questions)")
    statuses = [r["status"] for c in res["cases"] for r in c["results"]]
    print(f"  facts: {statuses.count('correct')} correct, {statuses.count('wrong')} wrong, {statuses.count('missing')} missing")
    print(f"  computed by tools: {res['computed_score']:.2f}   stated in brief: {res['score']:.3f}"
          + (f" (previous {prev['score']:.3f})" if prev else ""))
