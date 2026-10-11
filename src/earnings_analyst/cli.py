"""Command-line interface.

  earnings-analyst ask "How did Nike's latest quarter compare to a year ago?"
  earnings-analyst metric NKE gross_margin --compare prior_year
  earnings-analyst fallbacks
  earnings-analyst eval --suite tools
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .config import get_settings


def cmd_ask(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from .agent import run

    settings = get_settings()
    if args.model:
        settings = replace(settings, model=args.model)
    if args.max_steps:
        settings = replace(settings, max_steps=args.max_steps)
    print(f"Model: {settings.model} @ {settings.ollama_host} (max {settings.max_steps} steps)", file=sys.stderr)
    result = run(args.question, settings=settings)
    print(f"Run {result.run_id}: {result.model_calls} model calls, {len(result.tool_calls)} tool calls, "
          f"{result.seconds}s. Log: {result.log_path}", file=sys.stderr)
    if result.brief_path:
        brief = Path(result.brief_path).read_text(encoding="utf-8")
        print(brief if args.show_json else brief.split("\n## Data\n")[0])
        print(f"\nSaved: {result.brief_path}", file=sys.stderr)
        return 0
    print(result.answer)
    print("\nNo brief was saved (step limit reached or the model stopped early).", file=sys.stderr)
    return 1


def cmd_metric(args: argparse.Namespace) -> int:
    from .session import AnalystSession
    from .tools.metric_tools import build

    session = AnalystSession(f"cli metric {args.ticker} {args.metric}")
    tools = {t.name: t for t in build(session)}
    if args.history:
        out = tools["get_metric_history"].invoke(
            {"ticker": args.ticker, "metric": args.metric, "periods": args.history, "period_type": args.period_type})
    else:
        out = tools["calculate_metric"].invoke({
            "ticker": args.ticker, "metric": args.metric, "period": args.period,
            "period_type": args.period_type, "compare_to": args.compare,
        })
    result = json.loads(out)
    if args.full and result.get("ok") and not args.history:
        result = session.ledger.metrics[-1]  # with full provenance for every input
    print(json.dumps(result, indent=1))
    return 0 if result.get("ok", True) else 1


def cmd_fallbacks(args: argparse.Namespace) -> int:
    path = get_settings().log_dir / "run_python.jsonl"
    if not path.exists():
        print("No run_python calls logged yet.")
        return 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = Counter((r.get("metric_name") or "(unnamed)") for r in rows)
    print(f"{len(rows)} run_python fallback calls logged in {path}\n")
    print("Most requested calculations (candidates for the metric registry):")
    for name, n in counts.most_common(args.top):
        example = next(r["purpose"] for r in rows if (r.get("metric_name") or "(unnamed)") == name)
        print(f"  {n:4d}  {name:30s} e.g. {example[:80]}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evals import main as eval_main

    return eval_main(args)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="earnings-analyst", description="Sourced earnings briefs from SEC filings.")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("ask", help="Ask a question; the agent writes a brief.")
    a.add_argument("question")
    a.add_argument("--model", help="Override EARNINGS_MODEL")
    a.add_argument("--max-steps", type=int)
    a.add_argument("--show-json", action="store_true", help="Also print the JSON data block")
    a.set_defaults(fn=cmd_ask)

    m = sub.add_parser("metric", help="Compute one registry metric directly (no LLM).")
    m.add_argument("ticker")
    m.add_argument("metric")
    m.add_argument("--period", default="latest")
    m.add_argument("--period-type", choices=["quarter", "annual"], default="quarter")
    m.add_argument("--compare", choices=["prior_year", "prior_quarter"], default="prior_year")
    m.add_argument("--history", type=int, metavar="N", help="Show the last N periods instead")
    m.add_argument("--full", action="store_true", help="Show full provenance (what goes into the brief's JSON)")
    m.set_defaults(fn=cmd_metric)

    f = sub.add_parser("fallbacks", help="Summarise logged run_python fallbacks.")
    f.add_argument("--top", type=int, default=15)
    f.set_defaults(fn=cmd_fallbacks)

    e = sub.add_parser("eval", help="Run the evaluation harness.")
    e.add_argument("--suite", choices=["tools", "agent", "accuracy", "all"], default="tools")
    e.add_argument("--cases", default="evals/cases.yaml,evals/metric_goldens.yaml,evals/questions.yaml,evals/accuracy.yaml",
                   help="Comma-separated YAML files with tool_cases / agent_cases")
    e.add_argument("--only", help="Comma-separated case ids")
    e.add_argument("--category", help="Comma-separated categories, e.g. single_metric,missing_data")
    e.add_argument("--model", help="Override EARNINGS_MODEL for agent cases")
    e.add_argument("--out", default="evals/results")
    e.set_defaults(fn=cmd_eval)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
