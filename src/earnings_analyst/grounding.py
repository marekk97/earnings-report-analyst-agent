"""Checks that every number quoted in a brief can be traced to a tool result.

Numbers are matched at the precision the brief displays them, e.g. "42.8%" matches a computed 42.7864
and "$11.2 billion" matches 11,213,000,000. Numbers that only appear in fetched filing text (management
commentary) count as text-grounded; anything else is reported as ungrounded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

SCALES = {
    "trillion": 1e12, "tn": 1e12, "t": 1e12,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "million": 1e6, "mn": 1e6, "mm": 1e6, "m": 1e6,
    "thousand": 1e3, "k": 1e3,
}

# Things that look numeric but are identifiers or dates, not figures.
_NOISE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{10}-\d{2}-\d{6}\b|\b(?:10-Q|10-K|8-K|20-F|EX-99\.\d|Exhibit\s+99\.\d|Item\s+\d+\.\d+)\b|"
    r"\bQ[1-4]\b|\bFY\s?'?\d{2,4}\b|\b(?:19|20)\d{2}\b(?![.,]\d)|\[\^?\d+\]|\b\d+(?:st|nd|rd|th)\b",
    re.I,
)
_NUM = re.compile(
    r"(?P<dollar>\$)?\s?(?P<neg>[-−(])?\s?(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?P<dec>\.\d+)?\)?"
    r"(?:\s?(?P<unit>%|percent|pp|percentage points?|bps|basis points|x\b|×|days?\b|"
    r"trillion|billion|million|thousand|tn\b|bn\b|mn\b|mm\b|[TBMK]\b))?",
    re.I,
)


@dataclass
class Quoted:
    text: str
    value: float
    tolerance: float
    is_percent: bool


@dataclass
class GroundingReport:
    checked: int = 0
    grounded_tools: list[str] = field(default_factory=list)
    grounded_text: list[str] = field(default_factory=list)
    ungrounded: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return 1.0 if not self.checked else (len(self.grounded_tools) + len(self.grounded_text)) / self.checked

    def to_dict(self) -> dict:
        return {
            "numbers_checked": self.checked,
            "grounded_in_tool_results": len(self.grounded_tools),
            "grounded_in_filing_text": len(self.grounded_text),
            "ungrounded": self.ungrounded,
            "score": round(self.score, 3),
        }


def extract_numbers(text: str, require_unit: bool = True) -> list[Quoted]:
    clean = _NOISE.sub(" ", text)
    out = []
    for m in _NUM.finditer(clean):
        unit = (m.group("unit") or "").lower()
        dec = m.group("dec") or ""
        if require_unit and not (m.group("dollar") or unit or dec):
            continue  # bare integers are counts, ranks, footnotes...
        try:
            base = float(m.group("num").replace(",", "") + dec)
        except ValueError:
            continue
        decimals = len(dec) - 1 if dec else 0
        scale = SCALES.get(unit, 1.0)
        if unit in ("bps", "basis points"):
            base, decimals = base / 100, decimals + 2
        tol = 0.5 * 10 ** (-decimals) * scale
        # A whole-number figure such as "$5 billion" or "4%" is often rounded hard; allow one more unit.
        if decimals == 0 and scale > 1:
            tol = max(tol, 0.5 * scale)
        out.append(Quoted(m.group(0).strip(), base * scale, tol * 1.0001 + 1e-9, unit in ("%", "percent", "pp", "percentage point", "percentage points", "bps", "basis points")))
    return out


def numbers_from_obj(obj) -> Iterable[float]:
    """All numeric leaves of a JSON-like object."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from numbers_from_obj(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from numbers_from_obj(v)


def _matches(q: Quoted, pool: list[float]) -> bool:
    target = abs(q.value)
    for p in pool:
        if abs(abs(p) - target) <= q.tolerance:
            return True
    return False


def check(markdown: str, tool_numbers: Iterable[float], text_sources: Iterable[str]) -> GroundingReport:
    pool = sorted({abs(x) for x in tool_numbers})
    text_pool: list[float] | None = None
    report = GroundingReport()
    body = markdown.split("```json")[0]
    for q in extract_numbers(body):
        report.checked += 1
        if _matches(q, pool):
            report.grounded_tools.append(q.text)
            continue
        if text_pool is None:
            text_pool = sorted({abs(n.value) for t in text_sources for n in extract_numbers(t, require_unit=False)})
        if _matches(q, text_pool):
            report.grounded_text.append(q.text)
        else:
            report.ungrounded.append(q.text)
    return report
