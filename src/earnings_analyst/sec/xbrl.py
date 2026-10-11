"""Normalises the SEC `companyfacts` XBRL API into clean quarterly / annual / point-in-time series.

The raw API lists every value a company ever tagged, once per filing that repeated it. This module:
* keeps only 10-Q / 10-K facts and, when a period was reported several times, the most recently
  filed value (so restatements win and comparatives come from the same filing as the current period);
* classifies durations as quarters, fiscal years or year-to-date figures;
* derives missing quarters from cumulative figures (Q2 = 6M YTD − Q1, Q4 = FY − 9M YTD), which is how
  cash-flow statements and fourth quarters are usually reported, and records how it did so;
* labels periods with the company's own fiscal calendar (e.g. "Q1 FY2027" for Nike's quarter ending
  31 Aug 2026), using the fiscal year/period metadata of the filing whose primary period ends on that date.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Iterable, Literal

from ..provenance import DataError, Provenance, filing_index_url

PeriodType = Literal["quarter", "annual"]

# 12- to 16-week quarters: retailers such as Costco use 12/12/12/16-week fiscal quarters.
QUARTER_DAYS = (80, 119)
ANNUAL_DAYS = (357, 371)
ACCEPTED_FORMS = ("10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT", "10-QT")


def _d(s: str | None) -> date | None:
    return date.fromisoformat(s) if s else None


def _days(start: date, end: date) -> int:
    return (end - start).days + 1


def _is_quarter(days: int) -> bool:
    return QUARTER_DAYS[0] <= days <= QUARTER_DAYS[1]


def _is_annual(days: int) -> bool:
    return ANNUAL_DAYS[0] <= days <= ANNUAL_DAYS[1]


@dataclass(frozen=True)
class Fact:
    concept: str
    value: float
    unit: str
    end: date
    start: date | None
    accession: str
    form: str
    filed: str
    derived: str | None = None  # human-readable derivation note when not reported directly

    @property
    def days(self) -> int | None:
        return _days(self.start, self.end) if self.start else None


@dataclass(frozen=True)
class Period:
    type: PeriodType
    end: date
    label: str
    start: date | None = None
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None  # 1-4 for quarters

    @property
    def days(self) -> int:
        if self.start:
            return _days(self.start, self.end)
        return 91 if self.type == "quarter" else 365

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "type": self.type,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat(),
        }


@dataclass(frozen=True)
class _FilingInfo:
    accession: str
    form: str
    filed: str
    fy: int | None
    fp: str | None
    period_end: date


class CompanyFacts:
    def __init__(self, raw: dict, ticker: str | None = None):
        self.raw = raw
        self.cik = int(raw.get("cik", 0))
        self.name = raw.get("entityName", "")
        self.ticker = (ticker or "").upper()
        self._gaap: dict = raw.get("facts", {}).get("us-gaap", {})
        if not self._gaap:
            raise DataError(
                "no_us_gaap_facts",
                f"{self.name or self.ticker} has no US-GAAP XBRL facts (foreign filers using IFRS are not supported in v1).",
            )
        self._dedup_cache: dict[tuple[str, str], list[Fact]] = {}
        self._all_cache: dict[tuple[str, str], list[Fact]] = {}
        self._filings = self._index_filings()
        self._quarters, self._annuals = self._canonical_periods()

    # ------------------------------------------------------------------ indexing

    def _iter_raw(self, concept: str, unit: str) -> Iterable[dict]:
        node = self._gaap.get(concept)
        if not node:
            return []
        return [f for f in node.get("units", {}).get(unit, []) if f.get("form") in ACCEPTED_FORMS]

    def _index_filings(self) -> dict[str, _FilingInfo]:
        """Map each 10-Q/10-K accession to its primary period end.

        The primary end is the latest end of a *year-long* figure in a 10-K or a *quarter-long* figure in a 10-Q.
        (The latest end of any figure is unreliable: some 10-Ks tag a few values dated after the year end.)
        """
        best: dict[str, dict] = {}       # latest fact of the filing's own period length
        fallback: dict[str, dict] = {}   # latest fact of any length
        for node in self._gaap.values():
            for facts in node.get("units", {}).values():
                for f in facts:
                    if f.get("form") not in ACCEPTED_FORMS or not f.get("start"):
                        continue
                    accn = f["accn"]
                    if accn not in fallback or f["end"] > fallback[accn]["end"]:
                        fallback[accn] = f
                    days = _days(_d(f["start"]), _d(f["end"]))
                    own_length = _is_annual(days) if f["form"].startswith("10-K") else _is_quarter(days)
                    if own_length and (accn not in best or f["end"] > best[accn]["end"]):
                        best[accn] = f
        out = {}
        for accn, f in fallback.items():
            g = best.get(accn, f)
            out[accn] = _FilingInfo(accn, f["form"], f["filed"], f.get("fy"), f.get("fp"), _d(g["end"]))
        return out

    def _canonical_periods(self) -> tuple[list[Period], list[Period]]:
        """Build the company's fiscal calendar from its own filings, original filings first.

        Fiscal years come only from 10-Ks (the filing's fiscal-year metadata names them, e.g. Target's year ending
        3 Feb 2024 is "FY2023"). A 10-Q's quarter number is its position after the previous fiscal year end, and its
        fiscal year is that of the 10-K that closes the year: the SEC's per-filing fy/fp metadata is sometimes wrong
        (Disney's 2022 10-Qs are tagged "FY").
        """
        by_end: dict[date, _FilingInfo] = {}
        for info in sorted(self._filings.values(), key=lambda i: (i.filed, "/A" in i.form)):
            if info.period_end not in by_end and "/A" not in info.form:
                by_end[info.period_end] = info
        for info in self._filings.values():  # amendments only fill gaps
            by_end.setdefault(info.period_end, info)

        year_ends = sorted((end, info) for end, info in by_end.items() if info.form.startswith("10-K"))
        quarters: list[Period] = []
        annuals: list[Period] = []
        for end, info in year_ends:
            fy = info.fy
            annuals.append(Period("annual", end, f"FY{fy}" if fy else f"FY ended {end}", fiscal_year=fy))
            quarters.append(Period("quarter", end, f"Q4 FY{fy}" if fy else f"quarter ended {end}", fiscal_year=fy, fiscal_quarter=4))
        for end, info in by_end.items():
            if info.form.startswith("10-K"):
                continue
            prev = max(((e, i) for e, i in year_ends if e < end), default=None, key=lambda x: x[0])
            nxt = min(((e, i) for e, i in year_ends if e > end), default=None, key=lambda x: x[0])
            q, fy = None, None
            if prev and (end - prev[0]).days < 300:
                q = min(3, max(1, round((end - prev[0]).days / 91.3)))
                fy = nxt[1].fy if nxt and (nxt[0] - prev[0]).days < 400 else (prev[1].fy + 1 if prev[1].fy else None)
            else:  # no fiscal year end on record before this quarter: fall back to the filing metadata
                q = int(info.fp[1]) if info.fp and re.fullmatch(r"Q[1-3]", info.fp) else None
                fy = info.fy
            label = f"Q{q} FY{fy}" if q and fy else f"quarter ended {end}"
            quarters.append(Period("quarter", end, label, fiscal_year=fy, fiscal_quarter=q))
        quarters.sort(key=lambda p: p.end)

        def with_starts(periods: list[Period], gap: tuple[int, int]) -> list[Period]:
            out = []
            for i, p in enumerate(periods):
                start = None
                if i > 0 and gap[0] <= (p.end - periods[i - 1].end).days <= gap[1]:
                    start = periods[i - 1].end + timedelta(days=1)
                out.append(replace(p, start=start))
            return out

        return with_starts(quarters, QUARTER_DAYS), with_starts(annuals, (350, 380))

    def filing(self, accession: str) -> _FilingInfo | None:
        return self._filings.get(accession)

    # ------------------------------------------------------------------ periods

    def periods(self, period_type: PeriodType) -> list[Period]:
        return list(self._quarters if period_type == "quarter" else self._annuals)

    def latest(self, period_type: PeriodType) -> Period:
        periods = self.periods(period_type)
        if not periods:
            raise DataError("no_periods", f"No {period_type} periods found in {self.name}'s 10-Q/10-K XBRL data.")
        return periods[-1]

    def period_at(self, end: date, period_type: PeriodType, tolerance: int = 5) -> Period | None:
        best = min(self.periods(period_type), key=lambda p: abs((p.end - end).days), default=None)
        if best and abs((best.end - end).days) <= tolerance:
            return best
        return None

    def shift(self, period: Period, offset: Literal["prior_quarter", "prior_year"]) -> Period:
        """The comparable earlier period. Quarters can shift by a quarter or a year; years only by a year."""
        if offset == "prior_quarter":
            if period.type != "quarter":
                raise DataError("bad_comparison", "prior_quarter comparison only applies to quarterly periods.")
            # The preceding fiscal quarter, whatever its length.
            quarters = self.periods("quarter")
            idx = next((i for i, q in enumerate(quarters) if q.end == period.end), None)
            prev = quarters[idx - 1] if idx else None
            found = prev if prev and QUARTER_DAYS[0] <= (period.end - prev.end).days <= QUARTER_DAYS[1] else None
        else:
            found = self.period_at(period.end - timedelta(days=365), period.type, tolerance=16)
        if not found:
            raise DataError(
                "period_not_found",
                f"No {offset.replace('_', ' ')} period found for {period.label} in {self.name}'s filings.",
                period=period.label,
            )
        return found

    def resolve_period(self, spec: str | None, period_type: PeriodType = "quarter") -> Period:
        """Parse 'latest', 'FY2025', 'Q2 FY2025', 'Q2 2025', '2025Q2' or an ISO end date."""
        s = (spec or "latest").strip().upper().replace("FISCAL", "FY")
        if s in ("", "LATEST", "MOST RECENT", "CURRENT"):
            return self.latest(period_type)
        if m := re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s):
            found = self.period_at(date.fromisoformat(s), period_type)
            if found:
                return found
        if m := re.fullmatch(r"Q([1-4])\s*(?:FY)?\s*'?(\d{2,4})|(?:FY)?\s*(\d{4})\s*-?\s*Q([1-4])", s):
            q = int(m.group(1) or m.group(4))
            fy = int(m.group(2) or m.group(3))
            fy = fy + 2000 if fy < 100 else fy
            for p in self._quarters:
                if p.fiscal_year == fy and p.fiscal_quarter == q:
                    return p
        elif m := re.fullmatch(r"(?:FY)\s*'?(\d{2,4})|(\d{4})", s):
            fy = int(m.group(1) or m.group(2))
            fy = fy + 2000 if fy < 100 else fy
            for p in self._annuals:
                if p.fiscal_year == fy:
                    return p
        available = [p.label for p in self.periods(period_type)[-8:]]
        raise DataError(
            "period_not_found",
            f"Could not find period '{spec}' for {self.name}.",
            available_periods=available,
        )

    # ------------------------------------------------------------------ series

    def _all(self, concept: str, unit: str) -> list[Fact]:
        """Every 10-Q/10-K value of a concept, including the same period repeated by later filings."""
        key = (concept, unit)
        if key not in self._all_cache:
            self._all_cache[key] = [
                Fact(concept, float(f["val"]), unit, _d(f["end"]), _d(f.get("start")), f["accn"], f["form"], f["filed"])
                for f in self._iter_raw(concept, unit)
            ]
        return self._all_cache[key]

    def _dedup(self, concept: str, unit: str) -> list[Fact]:
        """One value per (start, end): the most recently filed, so restatements win."""
        key = (concept, unit)
        if key not in self._dedup_cache:
            latest: dict[tuple, Fact] = {}
            for f in self._all(concept, unit):
                k = (f.start, f.end)
                if k not in latest or f.filed > latest[k].filed:
                    latest[k] = f
            self._dedup_cache[key] = list(latest.values())
        return self._dedup_cache[key]

    def _direct(self, concept: str, unit: str, period: Period, kind: Literal["duration", "instant"]) -> list[Fact]:
        """All directly reported values (any filing) for exactly this period."""
        if kind == "instant":
            return [f for f in self._all(concept, unit) if f.start is None and abs((f.end - period.end).days) <= 3]
        want = _is_quarter if period.type == "quarter" else _is_annual
        return [f for f in self._all(concept, unit)
                if f.start is not None and abs((f.end - period.end).days) <= 3 and want(f.days)]

    def common_filing(
        self, concepts: Iterable[str], periods: list[Period], kind: Literal["duration", "instant"], unit: str = "USD"
    ) -> tuple[str, str] | None:
        """(concept, accession) of the latest filing that reports ALL the periods for one concept.

        Comparing figures from one filing keeps them on the same basis: a filing restates its comparatives for
        stock splits, reclassifications and accounting changes, while mixing filings can compare a split-adjusted
        EPS with an unadjusted one.
        """
        for c in concepts:
            per_period = [{f.accession: f.filed for f in self._direct(c, unit, p, kind)} for p in periods]
            common = set(per_period[0]).intersection(*per_period[1:]) if per_period else set()
            if common:
                return c, max(common, key=lambda a: per_period[0][a])
        return None

    def _duration_lookup(self, concept: str, unit: str, period: Period, additive: bool) -> Fact | None:
        facts = [f for f in self._dedup(concept, unit) if f.start is not None]
        want = _is_quarter if period.type == "quarter" else _is_annual

        def near(f: Fact) -> bool:
            return abs((f.end - period.end).days) <= 3

        direct = [f for f in facts if near(f) and want(f.days)]
        if direct:
            return max(direct, key=lambda f: f.filed)
        if period.type != "quarter" or not additive:
            return None
        # Derive the quarter from two cumulative figures that share a start date.
        enders = [f for f in facts if near(f)]
        for later in sorted(enders, key=lambda f: f.filed, reverse=True):
            for earlier in facts:
                if earlier.start == later.start and earlier.end < later.end and _is_quarter((later.end - earlier.end).days):
                    note = (
                        f"derived: {later.days}-day value to {later.end} ({later.value:,.0f}) minus "
                        f"{earlier.days}-day value to {earlier.end} ({earlier.value:,.0f}); "
                        "may be distorted if the later filing reclassified earlier periods"
                    )
                    return replace(later, value=later.value - earlier.value, start=earlier.end + timedelta(days=1), derived=note)
        return None

    def _instant_lookup(self, concept: str, unit: str, end: date) -> Fact | None:
        facts = [f for f in self._dedup(concept, unit) if f.start is None and abs((f.end - end).days) <= 3]
        return max(facts, key=lambda f: f.filed) if facts else None

    def lookup(
        self,
        concepts: Iterable[str],
        period: Period,
        kind: Literal["duration", "instant"],
        unit: str = "USD",
        additive: bool = True,
        accession: str | None = None,
    ) -> Fact | None:
        """First concept (in priority order) that has a value for the period.

        With `accession`, only a value reported directly in that filing is accepted.
        """
        if accession:
            for c in concepts:
                hits = [f for f in self._direct(c, unit, period, kind) if f.accession == accession]
                if hits:
                    return hits[0]
            return None
        for c in concepts:
            fact = (
                self._instant_lookup(c, unit, period.end)
                if kind == "instant"
                else self._duration_lookup(c, unit, period, additive)
            )
            if fact is not None:
                return fact
        return None

    def has_concept(self, concept: str) -> bool:
        return concept in self._gaap

    # ------------------------------------------------------------------ provenance

    def provenance(self, fact: Fact, period: Period) -> Provenance:
        info = self._filings.get(fact.accession)
        fiscal = period.label
        return Provenance(
            form=fact.form,
            fiscal_period=fiscal,
            filing_date=fact.filed,
            accession=fact.accession,
            url=filing_index_url(self.cik, fact.accession),
            period_start=fact.start.isoformat() if fact.start else None,
            period_end=fact.end.isoformat(),
            concept=f"us-gaap:{fact.concept}",
            note=fact.derived
            or (f"comparative figure reported in the filing for period ending {info.period_end}" if info and info.period_end != fact.end else None),
        )
