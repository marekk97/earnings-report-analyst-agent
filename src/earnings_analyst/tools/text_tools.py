"""get_filing_section and get_8k_earnings_release: management's own words, with provenance."""

from __future__ import annotations

from datetime import date
from typing import Literal

from langchain_core.tools import BaseTool, tool

from ..provenance import DataError
from ..sec.filings import Company, Filing, extract_section, focus_text, guidance_excerpts, normalize_section
from ..sec.xbrl import CompanyFacts
from ..session import AnalystSession
from .common import tool_guard

# Filing text is the largest part of every prompt (it is re-sent on each later step), so pages are kept short;
# use `focus` keywords to get the relevant paragraphs instead of more text.
DEFAULT_CHARS = 6000
MAX_CHARS_CAP = 8000


def _fiscal_label(facts: CompanyFacts, end: date, form: str) -> str:
    p = facts.period_at(end, "annual" if form.startswith("10-K") else "quarter")
    return p.label if p else f"period ended {end}"


def _label_before(facts: CompanyFacts, filed: date) -> str:
    """Fiscal label of the latest quarter that ended before an 8-K was filed."""
    quarters = [p for p in facts.periods("quarter") if p.end < filed and (filed - p.end).days <= 100]
    return quarters[-1].label if quarters else "unknown period"


def build(session: AnalystSession) -> list[BaseTool]:
    edgar = session.edgar

    def pick_periodic(company: Company, facts: CompanyFacts, form: str, period: str) -> Filing:
        if period.strip().lower() in ("", "latest"):
            if form == "auto":
                filings = edgar.filings(company, ("10-Q", "10-K"))
                if not filings:
                    raise DataError("filing_not_found", f"No 10-Q or 10-K filings found for {company.name}.")
                return filings[0]
            return edgar.periodic_filing(company, form, None)
        is_annual = period.strip().upper().startswith("FY") and "Q" not in period.upper()
        p = facts.resolve_period(period, "annual" if is_annual else "quarter")
        if form == "auto":
            form = "10-K" if (p.type == "annual" or p.fiscal_quarter == 4) else "10-Q"
        return edgar.periodic_filing(company, form, p.end)

    @tool("get_filing_section")
    @tool_guard(session, "get_filing_section")
    def get_filing_section(
        ticker: str,
        section: Literal["mdna", "risk_factors", "business", "market_risk", "legal_proceedings"] = "mdna",
        form: Literal["auto", "10-Q", "10-K"] = "auto",
        period: str = "latest",
        focus: str | None = None,
        max_chars: int = DEFAULT_CHARS,
        offset: int = 0,
    ) -> dict:
        """Pull one section of a 10-Q or 10-K, e.g. MD&A ("mdna") or Risk Factors, to explain the numbers.

        Use `focus` with a few keywords (e.g. "gross margin, pricing, inventory") to get only the most relevant
        paragraphs; otherwise text is returned in pages of max_chars starting at `offset`.

        Args:
            ticker: Company ticker or name.
            section: mdna, risk_factors, business (10-K only), market_risk or legal_proceedings.
            form: "auto" picks the filing for the period (10-K for fiscal Q4/full years, else 10-Q).
            period: "latest", a fiscal label like "Q2 FY2025" / "FY2024", or a period-end date.
            focus: Optional comma-separated keywords to select relevant paragraphs.
            max_chars: Maximum characters to return (default 6000, up to 8000).
            offset: Character offset for paging when focus is not used.
        """
        company = edgar.resolve_company(ticker)
        facts = edgar.facts(company)
        filing = pick_periodic(company, facts, form, period)
        key = normalize_section(section)
        text = extract_section(edgar.document_text(filing.primary_url), filing.form, key)
        chunk, truncated = focus_text(text, focus, min(int(max_chars), MAX_CHARS_CAP), int(offset))
        fiscal = _fiscal_label(facts, date.fromisoformat(filing.report_date), filing.form)
        prov = {
            "form": filing.form,
            "fiscal_period": fiscal,
            "filing_date": filing.filing_date,
            "period_end": filing.report_date,
            "accession": filing.accession,
            "url": filing.primary_url,
        }
        session.ledger.add_source({"kind": "filing_section", "section": key, **prov})
        session.ledger.text_snippets.append(chunk)
        out = {
            "ok": True,
            "company": company.name,
            "section": key,
            "provenance": prov,
            "citation": f"{filing.form} {key.upper()}, {fiscal}, filed {filing.filing_date}",
            "total_chars": len(text),
            "text": chunk,
        }
        if truncated and not focus:
            out["next_offset"] = int(offset) + len(chunk)
        return out

    @tool("get_8k_earnings_release")
    @tool_guard(session, "get_8k_earnings_release")
    def get_8k_earnings_release(
        ticker: str,
        period: str = "latest",
        focus: str | None = None,
        max_chars: int = DEFAULT_CHARS,
    ) -> dict:
        """Retrieve the earnings press release (8-K Item 2.02, Exhibit 99.1), including any guidance / outlook.

        Returns release text plus `guidance_excerpts`: sentences that look like forward-looking guidance.

        Args:
            ticker: Company ticker or name.
            period: "latest" or the fiscal period the release reports on (e.g. "Q1 FY2027", "FY2025").
            focus: Optional comma-separated keywords to select relevant paragraphs (e.g. "outlook, gross margin").
            max_chars: Maximum characters of release text to return (default 6000, up to 8000).
        """
        company = edgar.resolve_company(ticker)
        facts = edgar.facts(company)
        end = None
        if period.strip().lower() not in ("", "latest"):
            is_annual = period.strip().upper().startswith("FY") and "Q" not in period.upper()
            end = facts.resolve_period(period, "annual" if is_annual else "quarter").end
        filing = edgar.earnings_8k(company, end)
        docs = edgar.filing_documents(filing)
        exhibit = next((d for d in docs if d[0].upper() == "EX-99.1"), None) or next(
            (d for d in docs if d[0].upper().startswith("EX-99")), None
        )
        if not exhibit:
            raise DataError("release_not_found", f"8-K {filing.accession} has no Exhibit 99 press release attached.")
        text = edgar.document_text(exhibit[1])
        chunk, _ = focus_text(text, focus, min(int(max_chars), MAX_CHARS_CAP))
        fiscal = _label_before(facts, date.fromisoformat(filing.filing_date))
        prov = {
            "form": f"8-K ({exhibit[0]})",
            "fiscal_period": fiscal,
            "filing_date": filing.filing_date,
            "accession": filing.accession,
            "url": exhibit[1],
        }
        session.ledger.add_source({"kind": "earnings_release", **prov})
        guidance = guidance_excerpts(text)
        session.ledger.text_snippets.append(chunk + "\n" + "\n".join(guidance))
        return {
            "ok": True,
            "company": company.name,
            "provenance": prov,
            "citation": f"8-K earnings release, {fiscal}, filed {filing.filing_date}",
            "total_chars": len(text),
            "guidance_excerpts": guidance or ["No guidance language detected in the release."],
            "text": chunk,
        }

    return [get_filing_section, get_8k_earnings_release]
