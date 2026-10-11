"""Evaluation harness.

`tools` suite: deterministic registry checks against published historical figures (no LLM, seconds to run).
`agent` suite: end-to-end runs scored on
  * brief_saved          - a brief was written
  * steps_ok             - finished within the step budget
  * metric_recall        - required metrics were computed
  * metric_precision     - no metrics outside what the question needs
  * required_tools       - e.g. get_segment_data for a segment question
  * golden_accuracy      - computed values match published figures (when the case defines them)
  * grounding            - share of numbers in the brief traceable to tool results / filing text
  * citations            - the brief cites form type and filing date
  * explanation_sourced  - drivers come from MD&A or the earnings release
  * no_advice            - no buy/sell/price-target language
Each run is saved under evals/results/ and compared with the previous run of the same suite.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from .config import get_settings
from .session import AnalystSession

UNAVAILABLE_RE = re.compile(
    r"not (?:separately )?(?:reported|available|disclosed|tagged|provided)|unavailable|could not be (?:computed|calculated)|"
    r"cannot be (?:computed|calculated)|no data|n/a\b|missing",
    re.I,
)
_NEGATED = re.compile(r"\b(no|not|never|without|does not|doesn't|cannot|can't|nor)\b", re.I)


def gives_advice(brief: str) -> bool:
    """Advice language outside the echoed question and outside explicit disclaimers ("contains no price target")."""
    from .tools.brief_tool import ADVICE_RE

    for line in brief.splitlines():
        if line.lstrip("*_ ").lower().startswith("question"):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            if ADVICE_RE.search(sentence) and not _NEGATED.search(sentence):
                return True
    return False


CITATION_RE = re.compile(r"(10-Q|10-K|8-K)[^\n\]]{0,60}?(filed\s+)?\d{4}-\d{2}-\d{2}", re.I)


# ---------------------------------------------------------------------- tool suite


def run_tool_case(case: dict, tools: dict, session: AnalystSession) -> dict:
    args = {"ticker": case["ticker"], "metric": case["metric"], "period": case.get("period", "latest"),
            "period_type": case.get("period_type", "quarter"), "compare_to": case.get("compare_to", "prior_year")}
    out = json.loads(tools["calculate_metric"].invoke(args))
    if out.get("ok"):
        out = session.ledger.metrics[-1]  # the full result, with provenance for every input
    checks: dict[str, bool] = {}
    if "expect_error" in case:
        checks["structured_error"] = out.get("ok") is False and out.get("error") == case["expect_error"] and bool(out.get("message"))
    else:
        tol = case.get("tolerance", 0.01)
        checks["ok"] = out.get("ok") is True
        checks["value"] = checks["ok"] and abs(out["value"] - case["expected"]) <= tol
        prov = out.get("provenance") or {}
        checks["provenance"] = all(prov.get(k) for k in ("form", "fiscal_period", "filing_date", "accession"))
        checks["inputs_cited"] = checks["ok"] and all(i.get("provenance", {}).get("filing_date") for i in out["inputs"])
        if "expected_comparison" in case:
            comp = out.get("comparison") or {}
            checks["comparison"] = comp.get("value") is not None and abs(comp["value"] - case["expected_comparison"]) <= tol
        if "expect_note" in case:
            checks["note"] = any(case["expect_note"] in n for n in out.get("notes", []))
    return {
        "id": case["id"],
        "category": case.get("category", "curated"),
        "passed": all(checks.values()),
        "checks": checks,
        "got": out.get("value") if out.get("ok") else out.get("error"),
        "expected": case.get("expected", case.get("expect_error")),
        "message": out.get("message"),
    }


def run_tool_suite(cases: list[dict]) -> dict:
    from .tools.metric_tools import build

    session = AnalystSession("eval: tools")
    tools = {t.name: t for t in build(session)}
    results = [run_tool_case(c, tools, session) for c in cases]
    return {"suite": "tools", "cases": results, "score": mean(r["passed"] for r in results) if results else 0.0}


# ---------------------------------------------------------------------- agent suite


def score_agent_case(case: dict, result: Any, max_steps: int) -> dict:
    led = result.session.ledger
    computed = {m["metric"]: m for m in led.metrics}
    names = set(computed)
    tools_used = {c["tool"] for c in result.tool_calls}
    brief = Path(result.brief_path).read_text(encoding="utf-8").split("\n## Data\n")[0] if result.brief_path else ""
    scores: dict[str, float] = {}
    scores["brief_saved"] = float(bool(result.brief_path))
    scores["steps_ok"] = float(result.model_calls <= max_steps)
    required = set(case.get("required_metrics", []))
    scores["metric_recall"] = len(required & names) / len(required) if required else 1.0
    allowed = set(case.get("allowed_metrics", [])) | required
    scores["metric_precision"] = len(names & allowed) / len(names) if names else 1.0
    if case.get("required_tools"):
        scores["required_tools"] = len(set(case["required_tools"]) & tools_used) / len(case["required_tools"])
    if case.get("required_tools_any"):
        scores["required_tools"] = float(bool(set(case["required_tools_any"]) & tools_used))
    if case.get("required_comparison"):
        hits = [((computed.get(m) or {}).get("comparison") or {}).get("basis") == basis
                for m, basis in case["required_comparison"].items()]
        scores["comparison_basis"] = mean(hits)
    golden = case.get("golden") or {}
    if golden:
        hits = []
        for metric, g in golden.items():
            m = computed.get(metric)
            ok = bool(m) and m["period"]["label"] == g["period"] and abs(m["value"] - g["value"]) <= g.get("tolerance", 0.01)
            hits.append(ok)
        scores["golden_accuracy"] = mean(hits)
    if brief:
        from . import grounding
        from .tools.brief_tool import tool_numbers

        report = grounding.check(brief, tool_numbers(result.session), led.text_snippets)
        scores["grounding"] = report.score
        n_citations = len(CITATION_RE.findall(brief))
        scores["citations"] = min(1.0, n_citations / max(1, min(len(names), 3)))
        scores["no_advice"] = float(not gives_advice(brief))
        if case.get("expect_in_brief"):
            compact = re.sub(r"(\d)\s+%", r"\1%", brief)
            scores["expected_figures"] = mean(x in compact for x in case["expect_in_brief"])
        if case.get("expect_unavailable"):
            # Not invented: the metric was either never computed or the tool reported it missing,
            # and the brief says it is unavailable.
            not_invented = all(m not in computed for m in case["expect_unavailable"])
            scores["reports_unavailable"] = mean([not_invented, bool(UNAVAILABLE_RE.search(brief))])
    else:
        scores.update({"grounding": 0.0, "citations": 0.0, "no_advice": 1.0})
        report = None
    if case.get("require_explanation"):
        scores["explanation_sourced"] = float(bool({"get_filing_section", "get_8k_earnings_release"} & tools_used))
    return {
        "id": case["id"],
        "category": case.get("category", "uncategorised"),
        "question": case["question"],
        "known_issue": case.get("known_issue"),
        "score": round(mean(scores.values()), 3),
        "scores": {k: round(v, 3) for k, v in scores.items()},
        "metrics_computed": sorted(names),
        "tools_used": sorted(tools_used),
        "model_calls": result.model_calls,
        "seconds": result.seconds,
        "ungrounded_numbers": report.ungrounded if report else [],
        "python_fallbacks": [c.get("metric_name") for c in led.python_calls],
        "brief_path": result.brief_path,
        "log_path": result.log_path,
    }


def run_agent_suite(cases: list[dict], model: str | None) -> dict:
    from .agent import run

    settings = get_settings()
    if model:
        settings = replace(settings, model=model)
    results = []
    for case in cases:
        print(f"  running {case['id']} ...", file=sys.stderr, flush=True)
        started = time.monotonic()
        try:
            res = run(case["question"], settings=settings)
            results.append(score_agent_case(case, res, settings.max_steps))
        except Exception as exc:  # a crash is a failed case, not a failed suite
            results.append({"id": case["id"], "category": case.get("category", "uncategorised"),
                            "question": case["question"], "score": 0.0, "error": repr(exc),
                            "seconds": round(time.monotonic() - started, 1)})
    return {"suite": "agent", "model": settings.model, "cases": results,
            "score": round(mean(r["score"] for r in results), 3) if results else 0.0}


# ---------------------------------------------------------------------- reporting


def _previous(out_dir: Path, suite: str) -> dict | None:
    files = sorted(out_dir.glob(f"*_{suite}.json"))
    return json.loads(files[-1].read_text()) if files else None


def _by_category(cases: list[dict], key: str) -> dict[str, tuple[float, int]]:
    groups: dict[str, list[float]] = {}
    for c in cases:
        groups.setdefault(c.get("category", "uncategorised"), []).append(float(c[key]))
    return {k: (mean(v), len(v)) for k, v in sorted(groups.items())}


def report(res: dict, prev: dict | None) -> None:
    prev_cases = {c["id"]: c for c in (prev or {}).get("cases", [])}
    tools = res["suite"] == "tools"
    key = "passed" if tools else "score"
    print(f"\n== {res['suite']} suite" + (f" ({res['model']})" if res.get("model") else "") + f" - {len(res['cases'])} cases ==")
    for c in res["cases"]:
        was = prev_cases.get(c["id"], {}).get(key)
        if tools:
            changed = was is not None and was != c["passed"]
            if c["passed"] and not changed:
                continue  # print failures and status changes only
            status = "PASS" if c["passed"] else "FAIL"
            delta = "" if not changed else ("  (fixed)" if c["passed"] else "  (REGRESSED)")
            failed = [k for k, v in c["checks"].items() if not v]
            detail = "" if c["passed"] else f"  got={c['got']} expected={c['expected']} failed={failed} {c.get('message') or ''}"
            print(f"  {status}  {c['id']}{delta}{detail}")
        else:
            delta = f" ({c['score'] - was:+.2f})" if was is not None else ""
            weak = {k: v for k, v in c.get("scores", {}).items() if v < 1}
            note = f"  known issue: {c['known_issue']}" if c.get("known_issue") and c["score"] < 1 else ""
            print(f"  {c['score']:.2f}{delta}  [{c.get('category')}] {c['id']}  {weak or c.get('error') or ''}{note}")
    print("  by category:")
    prev_cat = _by_category(prev["cases"], key) if prev else {}
    for cat, (score, n) in _by_category(res["cases"], key).items():
        p = prev_cat.get(cat)
        trend = f" (was {p[0]:.2f})" if p and abs(p[0] - score) > 1e-9 else ""
        print(f"    {cat:20s} {score:.2f}  n={n}{trend}")
    prev_score = (prev or {}).get("score")
    trend = f" (previous {prev_score:.2f})" if prev_score is not None else ""
    print(f"  overall: {res['score']:.3f}{trend}")


DEFAULT_CASE_FILES = "evals/cases.yaml,evals/metric_goldens.yaml,evals/questions.yaml,evals/accuracy.yaml"


def load_cases(paths: str) -> dict[str, list[dict]]:
    """Merge tool_cases / agent_cases from several YAML files and resolve golden references."""
    merged: dict[str, list[dict]] = {"tool_cases": [], "agent_cases": [], "accuracy_cases": []}
    for path in paths.split(","):
        data = yaml.safe_load(Path(path.strip()).read_text(encoding="utf-8")) or {}
        for k in merged:
            merged[k].extend(data.get(k) or [])
    for k, items in merged.items():
        ids = [c["id"] for c in items]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate {k} ids: {sorted(dupes)}")
    tool_by_id = {c["id"]: c for c in merged["tool_cases"]}
    for case in merged["accuracy_cases"]:
        for spec in case["facts"]:
            for ref in ([spec["golden"]] if "golden" in spec else spec.get("change_of", [])):
                if ref not in tool_by_id:
                    raise ValueError(f"accuracy case {case['id']} references unknown golden '{ref}'")
    for case in merged["agent_cases"]:
        for metric, ref in list((case.get("golden") or {}).items()):
            if isinstance(ref, str):
                t = tool_by_id[ref]
                case["golden"][metric] = {"period": t["period"], "value": t["expected"], "tolerance": t.get("tolerance", 0.01)}
    return merged


def main(args: Any) -> int:
    cases = load_cases(args.cases)
    only = set(args.only.split(",")) if args.only else None
    categories = set(args.category.split(",")) if args.category else None

    def pick(xs: list[dict]) -> list[dict]:
        return [x for x in xs if (not only or x["id"] in only)
                and (not categories or x.get("category", "curated") in categories)]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suites = ["tools", "agent", "accuracy"] if args.suite == "all" else [args.suite]
    worst = 1.0
    for suite in suites:
        prev = _previous(out_dir, suite)
        if suite == "accuracy":
            from . import accuracy

            goldens = {c["id"]: c for c in cases["tool_cases"]}
            res = accuracy.run_accuracy_suite(pick(cases["accuracy_cases"]), goldens, args.model)
            accuracy.report(res, prev)
        else:
            res = run_tool_suite(pick(cases["tool_cases"])) if suite == "tools" else run_agent_suite(
                pick(cases["agent_cases"]), args.model)
            report(res, prev)
        (out_dir / f"{stamp}_{suite}.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
        worst = min(worst, res["score"])
    return 0 if worst >= 1.0 or suites != ["tools"] else 1
