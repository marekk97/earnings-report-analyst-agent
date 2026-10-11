# Earnings Report Analyst Agent

A LangChain agent that turns a company's SEC filings into a sourced, one-page earnings brief.

Ask a question like *"How did Nike's latest quarter compare to a year ago?"*. The agent picks the metrics
the question needs and computes them with tested formulas from SEC XBRL data. It explains the results
with management's own words from the MD&A or the earnings release, and saves a Markdown brief plus a JSON block.
Every number is cited to the filing it came from.

```
User question → Agent loop (open model via Ollama, max 10 steps, every call logged)
                   ⇅ tool calls
                Metric tools · Text tools · run_python fallback
                   ↓ reads
                SEC EDGAR (XBRL facts, 10-Q, 10-K, 8-K)
Agent → save_brief → briefs/<TICKER>_<period>_<run>.md + .json
```

**The model plans and explains; the tools fetch and calculate.** The model decides *which* metrics to compute.
The code decides *how* they are computed.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # set SEC_USER_AGENT to "AppName your-email@domain.com"
```

**Model: pick one**

| | Setup | `.env` |
|---|---|---|
| Local (free) | `ollama pull qwen3.5:9b` (any tool-calling model works) | `OLLAMA_HOST=http://localhost:11434`, `EARNINGS_MODEL=qwen3.5:9b` |
| Ollama Cloud | Create a key at ollama.com/settings/keys | `OLLAMA_HOST=https://ollama.com`, `OLLAMA_API_KEY=...`, `EARNINGS_MODEL=gpt-oss:120b` |

Local models need a GPU or Apple Silicon to be practical. On a CPU-only machine, use Ollama Cloud.
On local models that support it, set `EARNINGS_REASONING=false` to skip "thinking" and get faster runs.

```bash
earnings-analyst ask "How did Nike's latest quarter compare to a year ago?"
earnings-analyst ask "Which of Nike's regions drove the revenue decline last quarter?" --show-json

# Tools without an LLM (useful for checking numbers)
earnings-analyst metric NKE gross_margin --compare prior_year
earnings-analyst metric NKE gross_margin --full      # full provenance, as stored for the brief's JSON
earnings-analyst metric AAPL revenue_growth_yoy --period FY2023
earnings-analyst metric NKE gross_margin --history 8

earnings-analyst fallbacks        # which run_python calculations are requested most
earnings-analyst eval --suite tools
```

## Toolset

| Tool | Purpose |
|---|---|
| `calculate_metric` | Computes one registered metric, optionally vs. the prior quarter or year |
| `get_metric_history` | One metric over the last 8 quarters (or N years) |
| `get_filing_section` | One section of a 10-Q/10-K (MD&A, Risk Factors, ...), with optional keyword focus |
| `get_segment_data` | Revenue and operating income by segment, geography or product line (from the filing's XBRL) |
| `get_8k_earnings_release` | The earnings press release (8-K Item 2.02, EX-99.1), plus extracted guidance sentences |
| `run_python` | Logged fallback for calculations not in the registry |
| `save_brief` | Writes the Markdown brief. The JSON block is built from the tool results, not from the model's text |

### Metric registry

`revenue_growth_yoy`, `revenue_growth_qoq`, `gross_margin`, `operating_margin`, `net_margin`, `eps_growth_yoy`,
`eps_growth_qoq`, `free_cash_flow`, `fcf_margin`, `current_ratio`, `debt_to_equity`, `interest_coverage`, `roe`,
`roa`, `dso`, `inventory_turnover`.

Each metric maps to a formula in [metrics/formulas.py](src/earnings_analyst/metrics/formulas.py) and to the XBRL concepts it needs in
[metrics/registry.py](src/earnings_analyst/metrics/registry.py). Concepts have fallback lists because companies change tags over time.
To add a metric, add a formula, add a `Metric(...)` entry, and add a golden case to `evals/cases.yaml`.

## Design rules (and how they're enforced)

- **Provenance on every result.** Each value, and each input to it, carries the form type, fiscal period,
  filing date, accession number, XBRL concept and filing URL. Text tools cite the filing and section.
- **Structured errors, never empty results.** Missing data returns, for example,
  `{"ok": false, "error": "missing_data", "message": "inventory not reported by ... for Q3 FY2025", "concepts_tried": {...}}`.
- **Every `run_python` call is logged** to `logs/run_python.jsonl` with its stated purpose and a suggested
  metric name. `earnings-analyst fallbacks` ranks them, which shows what to add to the registry next.
- **All calls are logged.** Every model call (content, tool calls, token usage) and every tool call (args and
  output) goes to `logs/runs/<run_id>.jsonl`.
- **10-step budget.** `ModelCallLimitMiddleware` caps model calls. A small middleware warns the model on its
  last step so it saves the brief instead of running out of steps.
- **Grounding check.** `save_brief` matches every number in the brief against the tool results and the
  filing text that was fetched, at the precision displayed. It returns warnings for numbers it can't match
  and for investment-advice language, so the model can fix the brief.

### How the XBRL data is normalised ([sec/xbrl.py](src/earnings_analyst/sec/xbrl.py))

- **Fiscal calendar.** Periods are labelled from the company's own filings (Nike's quarter ending
  31 Aug 2026 is `Q1 FY2027`).
- **Derived quarters.** Cash-flow statements are year-to-date, and 10-Ks report only full years. Missing quarters are
  therefore derived: Q2 = 6M − 3M, and Q4 = FY − 9M. The derivation appears in the metric's `notes`. Per-share values (EPS)
  are never derived this way.
- **Restatements win.** When several filings report the same period, the most recently filed value is used. A
  year-ago comparison is usually cited to the same 10-Q as the current quarter.
- **Operating income fallback.** Some companies (e.g. Nike) don't tag `OperatingIncomeLoss`. Operating income
  is then approximated as gross profit minus operating expenses, and labelled as an approximation.

## Evaluation harness

```bash
earnings-analyst eval --suite tools                     # ~1,400 metric checks, no LLM, ~1-2 min (exit 1 on any failure)
earnings-analyst eval --suite agent                     # 45 questions end-to-end, scored on behaviour
earnings-analyst eval --suite accuracy                  # 39 questions end-to-end, scored ONLY on the numbers stated
earnings-analyst eval --suite agent --category single_metric,missing_data
earnings-analyst eval --suite agent --only q_nvda_eps_split --model gpt-oss:120b
```

| File | Contents |
|---|---|
| [evals/cases.yaml](evals/cases.yaml) | Hand-curated metric checks: published figures, derived quarters, structured errors, restatements, a stock split |
| [evals/metric_goldens.yaml](evals/metric_goldens.yaml) | **Generated.** 30 companies x 12 metrics x FY2022-FY2024, plus 3 metrics for each quarter of FY2024: annual, quarterly and missing-data cases |
| [evals/questions.yaml](evals/questions.yaml) | 45 agent questions in 14 categories, see below |
| [evals/accuracy.yaml](evals/accuracy.yaml) | 39 questions / 101 expected facts for the metric-accuracy suite, see below |

**How the metric goldens are made:** [evals/build_metric_goldens.py](evals/build_metric_goldens.py) computes each expected value
*independently* of the agent's pipeline. It opens the single 10-K/10-Q and reads the figures as that filing reported them, with
no de-duplication across filings, no derived quarters, and the formulas written out again. A case is kept only when both
implementations agree. Disagreements are printed for a human to resolve. When the dataset was built, they exposed two
pipeline bugs (a stock split mixed into EPS growth, and 52/53-week fiscal years being rescaled) and genuine later
restatements at Meta, Pfizer, Tesla and Target. Three of those, plus the split, are now curated cases. As an outside check, 12 revenue-growth and margin figures were confirmed against the
companies' published annual reports. Rebuild with `python evals/build_metric_goldens.py`.

**Question categories:** single_metric, annual_comparison, quarter_yoy, quarter_qoq, cash_flow, balance_sheet, efficiency,
segments, trend, guidance, missing_data, fallback (run_python), advice, company_resolution. Questions about past fiscal
years reference a golden case by id (`golden: {gross_margin: cost_fy2023_gross_margin}`), so expected numbers live in one place.
"Latest quarter" questions change as companies file, so they are scored on behaviour rather than exact values.

**Metric-accuracy suite (`--suite accuracy`):** end-to-end questions scored on a single thing: whether the numbers
in the brief are right. The question types are single metric, several metrics, two periods (both values plus the
change), a specific quarter, time series (four quarters / three years), by region, segment or product, and
company vs company. Each question lists its expected facts, which reference the independently built goldens (segment
figures are literal values checked against the companies' annual reports). For each fact, the scorer finds the table
row or sentence that mentions it (and its period, when several periods appear) and checks the number at the precision
displayed. A fact is reported as `correct`, `wrong` (mentioned with the wrong number, with the candidates shown) or
`missing`. It also notes whether the tools had computed the right value, which separates calculation errors from
reporting errors. The headline score is the share of facts stated correctly, overall and per question type.

**Agent scores (per question):** `brief_saved`, `steps_ok`, `metric_recall`, `metric_precision`, `required_tools`,
`comparison_basis`, `golden_accuracy` (value and period), `grounding`, `citations`, `explanation_sourced`,
`expected_figures`, `reports_unavailable` (the metric was not invented and the brief says it is unavailable) and `no_advice`.

Results go to `evals/results/`. Each run prints a per-category summary and is compared with the previous run, with
regressions flagged.

## Tests

```bash
pytest            # offline: synthetic XBRL fixtures, plus the agent loop driven by a scripted chat model
```

## Layout

```
src/earnings_analyst/
  agent.py            create_agent + Ollama model, call logger, step budget
  prompts.py          system prompt (workflow, rules, brief format)
  sec/                EDGAR client (rate-limited, cached), XBRL normalisation, filings/sections, segments
  metrics/            formulas (pure) + registry (metric → formula + XBRL concepts)
  tools/              the seven tools
  grounding.py        numbers-in-brief vs. tool-results check
  evals.py, cli.py
evals/cases.yaml      golden tool cases + agent questions
```

## Scope

- **Supported:** US-GAAP filers (10-Q/10-K/8-K), using free SEC data only.
- **Not supported:** IFRS filers (20-F/40-F) return a structured error. After a holding-company reorganisation the
  ticker can point to a new registrant with little history (e.g. XOM since 2026). Pass the predecessor's CIK instead
  (`CIK34088`). Investment advice, price targets, real-time
  market data, consensus estimates and call transcripts are out of scope.
- **`run_python`:** runs in a separate isolated process with timeouts and resource limits. This prevents accidents, but it is not a
  security sandbox.
