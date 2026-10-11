"""Company lookup, filing lists and filing documents (10-Q / 10-K sections, 8-K exhibits, XBRL instances)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from ..provenance import DataError, filing_index_url
from .client import SECClient
from .xbrl import CompanyFacts

ARCHIVES = "https://www.sec.gov/Archives/edgar/data"


@dataclass(frozen=True)
class Company:
    cik: int
    ticker: str
    name: str

    @property
    def cik10(self) -> str:
        return f"{self.cik:010d}"


@dataclass(frozen=True)
class Filing:
    cik: int
    accession: str
    form: str
    filing_date: str
    report_date: str
    primary_document: str
    items: str = ""

    @property
    def folder(self) -> str:
        return f"{ARCHIVES}/{self.cik}/{self.accession.replace('-', '')}"

    @property
    def primary_url(self) -> str:
        return f"{self.folder}/{self.primary_document}"

    @property
    def index_url(self) -> str:
        return filing_index_url(self.cik, self.accession)


class EDGAR:
    """High-level access to one company's EDGAR data, shared by all tools."""

    def __init__(self, client: SECClient | None = None):
        self.client = client or SECClient()
        self._facts: dict[int, CompanyFacts] = {}

    # ------------------------------------------------------------------ companies

    def resolve_company(self, query: str) -> Company:
        q = query.strip()
        if q.upper().startswith("CIK") and q[3:].strip().isdigit():
            q = q[3:].strip()
        if not q:
            raise DataError("unknown_company", "No company given.")
        table = self.client.get_json("https://www.sec.gov/files/company_tickers.json")
        rows = list(table.values())
        up = q.upper().replace(".", "-")
        for r in rows:
            if r["ticker"].upper() == up:
                return Company(int(r["cik_str"]), r["ticker"].upper(), r["title"])
        if q.isdigit():
            for r in rows:
                if int(r["cik_str"]) == int(q):
                    return Company(int(r["cik_str"]), r["ticker"].upper(), r["title"])
            # Registrants without a current ticker (e.g. predecessors of reorganised companies) by CIK.
            try:
                sub = self.client.get_json(f"https://data.sec.gov/submissions/CIK{int(q):010d}.json")
            except DataError:
                sub = None
            if sub and sub.get("name"):
                ticker = (sub.get("tickers") or [f"CIK{int(q)}"])[0]
                return Company(int(q), ticker.upper(), sub["name"])
        norm = re.sub(r"[^a-z0-9 ]", "", q.lower())
        norm = re.sub(r"\b(inc|corp|corporation|co|company|ltd|plc|the)\b", "", norm).strip()
        matches = [r for r in rows if norm and norm in re.sub(r"[^a-z0-9 ]", "", r["title"].lower())]
        if matches:
            # The SEC file is ordered roughly by market value, so the first hit is usually the intended one.
            r = matches[0]
            return Company(int(r["cik_str"]), r["ticker"].upper(), r["title"])
        raise DataError("unknown_company", f"No SEC registrant matches '{query}'. Try the ticker symbol.")

    def facts(self, company: Company) -> CompanyFacts:
        if company.cik not in self._facts:
            raw = self.client.get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{company.cik10}.json")
            self._facts[company.cik] = CompanyFacts(raw, ticker=company.ticker)
        return self._facts[company.cik]

    # ------------------------------------------------------------------ filings

    def filings(self, company: Company, forms: tuple[str, ...]) -> list[Filing]:
        sub = self.client.get_json(f"https://data.sec.gov/submissions/CIK{company.cik10}.json")
        r = sub["filings"]["recent"]
        out = []
        for i, form in enumerate(r["form"]):
            if form in forms:
                out.append(
                    Filing(
                        cik=company.cik,
                        accession=r["accessionNumber"][i],
                        form=form,
                        filing_date=r["filingDate"][i],
                        report_date=r["reportDate"][i] or r["filingDate"][i],
                        primary_document=r["primaryDocument"][i],
                        items=r.get("items", [""] * len(r["form"]))[i] or "",
                    )
                )
        return sorted(out, key=lambda f: f.filing_date, reverse=True)

    def periodic_filing(self, company: Company, form: str, period_end: date | None) -> Filing:
        """The 10-Q/10-K for a period end (or the latest one)."""
        candidates = self.filings(company, (form,))
        if not candidates:
            raise DataError("filing_not_found", f"No {form} filings found for {company.name}.")
        if period_end is None:
            return candidates[0]
        for f in candidates:
            if abs((date.fromisoformat(f.report_date) - period_end).days) <= 5:
                return f
        raise DataError("filing_not_found", f"No {form} found for the period ending {period_end} for {company.name}.")

    def earnings_8k(self, company: Company, period_end: date | None) -> Filing:
        """The Item 2.02 (Results of Operations) 8-K filed soonest after a period end, or the latest one."""
        candidates = [f for f in self.filings(company, ("8-K",)) if "2.02" in f.items]
        if not candidates:
            raise DataError("release_not_found", f"No 8-K earnings releases (Item 2.02) found for {company.name}.")
        if period_end is None:
            return candidates[0]
        after = [
            f for f in candidates
            if period_end <= date.fromisoformat(f.filing_date) <= period_end + timedelta(days=100)
        ]
        if not after:
            raise DataError("release_not_found", f"No earnings 8-K filed within 100 days after {period_end} for {company.name}.")
        return min(after, key=lambda f: f.filing_date)

    def filing_documents(self, filing: Filing) -> list[tuple[str, str]]:
        """(document type, url) pairs from the filing index page."""
        html = self.client.get_text(filing.index_url)
        soup = BeautifulSoup(html, "lxml")
        docs = []
        for row in soup.select("table.tableFile tr"):
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            link = cells[2].find("a")
            if not link:
                continue
            href = link["href"].replace("/ix?doc=", "")
            docs.append((cells[3].get_text(strip=True), "https://www.sec.gov" + href))
        return docs

    def xbrl_instance_url(self, filing: Filing) -> str:
        idx = self.client.get_json(f"{filing.folder}/index.json")
        names = [i["name"] for i in idx["directory"]["item"]]
        for n in names:
            if n.endswith("_htm.xml"):
                return f"{filing.folder}/{n}"
        for n in names:
            if n.endswith(".xml") and not re.search(r"_(cal|def|lab|pre)\.xml$|FilingSummary", n):
                return f"{filing.folder}/{n}"
        raise DataError("xbrl_instance_not_found", f"No XBRL instance document in {filing.form} {filing.accession}.")

    def document_text(self, url: str) -> str:
        return html_to_text(self.client.get_text(url))


# ---------------------------------------------------------------------- text extraction


warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        tag.decompose()
    blocks = []
    for el in soup.find_all(["p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br"]):
        el.append("\n")
    for td in soup.find_all(["td", "th"]):
        td.append(" | ")
    text = soup.get_text()
    text = text.replace("\xa0", " ").replace("​", "")
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip(" |")
        line = re.sub(r"(\s*\|\s*)+", " | ", line).strip()
        if line:
            blocks.append(line)
    out = "\n".join(blocks)
    return re.sub(r"\n{3,}", "\n\n", out)


_APOS = r"[’'`]?"
SECTION_PATTERNS: dict[str, dict[str, tuple[str, str]]] = {
    "10-K": {
        "business": (r"item\s*1\.?\s*[-–—:]?\s*business\b", r"item\s*1a\.?\s*[-–—:]?\s*risk\s+factors"),
        "risk_factors": (r"item\s*1a\.?\s*[-–—:]?\s*risk\s+factors", r"item\s*1b\.?|item\s*1c\.?|item\s*2\.?\s*[-–—:]?\s*properties"),
        "mdna": (
            rf"item\s*7\.?\s*[-–—:]?\s*management{_APOS}s?\s+discussion",
            r"item\s*7a\.?|item\s*8\.?\s*[-–—:]?\s*financial\s+statements",
        ),
        "market_risk": (r"item\s*7a\.?", r"item\s*8\.?"),
        "legal_proceedings": (r"item\s*3\.?\s*[-–—:]?\s*legal\s+proceedings", r"item\s*4\.?"),
    },
    "10-Q": {
        "mdna": (
            rf"item\s*2\.?\s*[-–—:]?\s*management{_APOS}s?\s+discussion",
            r"item\s*3\.?\s*[-–—:]?\s*quantitative|item\s*4\.?\s*[-–—:]?\s*controls",
        ),
        "risk_factors": (r"item\s*1a\.?\s*[-–—:]?\s*risk\s+factors", r"item\s*2\.?\s*[-–—:]?\s*unregistered|item\s*3\.?\s*[-–—:]?\s*defaults|item\s*5\.?|item\s*6\.?"),
        "market_risk": (r"item\s*3\.?\s*[-–—:]?\s*quantitative", r"item\s*4\.?"),
        "legal_proceedings": (r"item\s*1\.?\s*[-–—:]?\s*legal\s+proceedings", r"item\s*1a\.?|item\s*2\.?"),
    },
}
SECTION_ALIASES = {
    "md&a": "mdna", "mda": "mdna", "management_discussion": "mdna", "managements_discussion_and_analysis": "mdna",
    "risk": "risk_factors", "risks": "risk_factors", "riskfactors": "risk_factors",
    "market risk": "market_risk", "legal": "legal_proceedings",
}


def normalize_section(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    key = SECTION_ALIASES.get(key, SECTION_ALIASES.get(key.replace("_", " "), key))
    return key.replace(" ", "_")


def extract_section(text: str, form: str, section: str) -> str:
    """Return the body of an Item section, skipping table-of-contents entries.

    Every occurrence of the heading is a candidate start; each runs to the next end heading. The
    table of contents produces tiny candidates, so the longest candidate is the real section.
    """
    base_form = "10-K" if form.upper().startswith("10-K") else "10-Q"
    patterns = SECTION_PATTERNS[base_form].get(section)
    if not patterns:
        raise DataError(
            "unknown_section",
            f"Section '{section}' is not supported for {base_form}.",
            supported=sorted(SECTION_PATTERNS[base_form]),
        )
    start_re = re.compile(r"(?im)^\s*(?:part\s+i+\s*[-–—,.]?\s*)?" + patterns[0])
    end_re = re.compile(r"(?im)^\s*(?:part\s+i+\s*[-–—,.]?\s*)?(?:" + patterns[1] + ")")
    best = ""
    for m in start_re.finditer(text):
        end = end_re.search(text, m.end())
        chunk = text[m.start(): end.start() if end else len(text)]
        if len(chunk) > len(best):
            best = chunk
    if len(best) < 500:
        raise DataError("section_not_found", f"Could not locate the {section} section in this {base_form}.")
    return best.strip()


def focus_text(text: str, focus: str | None, max_chars: int, offset: int = 0) -> tuple[str, bool]:
    """Trim long text. With `focus` keywords, keep the most relevant paragraphs in document order."""
    if focus:
        terms = [t for t in re.split(r"[,\s]+", focus.lower()) if len(t) > 2]
        paras = [p for p in re.split(r"\n+", text) if p.strip()]
        scored = []
        for i, p in enumerate(paras):
            low = p.lower()
            score = sum(low.count(t) for t in terms)
            if score:
                scored.append((score, i))
        keep: set[int] = set()
        total = 0
        for score, i in sorted(scored, key=lambda s: (-s[0], s[1])):
            for j in (i - 1, i, i + 1):  # keep a little context around each hit
                if 0 <= j < len(paras) and j not in keep:
                    if total + len(paras[j]) > max_chars:
                        break
                    keep.add(j)
                    total += len(paras[j]) + 1
            if total >= max_chars:
                break
        if keep:
            return "\n".join(paras[j] for j in sorted(keep)), len(keep) < len(paras)
    chunk = text[offset: offset + max_chars]
    return chunk, offset + max_chars < len(text)


GUIDANCE_RE = re.compile(
    r"\b(outlook|guidance|expects?|expected|anticipates?|forecasts?|full[- ]year|next quarter|fiscal (?:year )?20\d\d (?:revenue|sales)|"
    r"projects?|targets?|reaffirm\w*|rais\w+ (?:its|our|the) (?:outlook|guidance)|lower\w* (?:its|our|the) (?:outlook|guidance))\b",
    re.I,
)


def guidance_excerpts(text: str, limit: int = 8) -> list[str]:
    """Sentences that look like forward-looking guidance (skipping the safe-harbor boilerplate)."""
    out = []
    for para in text.split("\n"):
        low = para.lower()
        if "forward-looking statements" in low or "safe harbor" in low or "private securities litigation" in low:
            continue
        if GUIDANCE_RE.search(para) and 40 < len(para) < 1200:
            out.append(para.strip())
        if len(out) >= limit:
            break
    return out

