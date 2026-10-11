"""Offline fixtures: a synthetic company whose XBRL facts mimic how real 10-Q/10-K filings report data."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import pytest

from earnings_analyst.config import Settings
from earnings_analyst.sec.filings import Company
from earnings_analyst.sec.xbrl import CompanyFacts
from earnings_analyst.session import AnalystSession

# Calendar fiscal year. Quarterly revenue by (year, quarter).
REVENUE = {(2023, 1): 90, (2023, 2): 95, (2023, 3): 100, (2023, 4): 105,
           (2024, 1): 100, (2024, 2): 110, (2024, 3): 120, (2024, 4): 130,
           (2025, 1): 115, (2025, 2): 121, (2025, 3): 132}
QEND = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}


def qend(y: int, q: int) -> str:
    return f"{y}-{QEND[q]}"


def qstart(y: int, q: int) -> str:
    return f"{y}-01-01" if q == 1 else (date.fromisoformat(qend(y, q - 1)) + timedelta(days=1)).isoformat()


def accn(y: int, q: int) -> str:
    return f"0000000001-{str(y)[2:]}-00000{q}"


def filed(y: int, q: int) -> str:
    return f"{y + 1}-02-15" if q == 4 else (date.fromisoformat(qend(y, q)) + timedelta(days=35)).isoformat()


def build_raw() -> dict:
    """Facts as each filing reports them: 10-Qs give the quarter + YTD (+ prior-year comparatives);
    10-Ks give full years only. Cash flow is YTD only. EPS is per-share (not derivable)."""
    us: dict[str, dict] = {}

    def add(concept: str, unit: str, start: str | None, end: str, val: float, y: int, q: int) -> None:
        form = "10-K" if q == 4 else "10-Q"
        fact = {"end": end, "val": val, "accn": accn(y, q), "fy": y, "fp": "FY" if q == 4 else f"Q{q}",
                "form": form, "filed": filed(y, q)}
        if start:
            fact["start"] = start
        us.setdefault(concept, {"units": {}})["units"].setdefault(unit, []).append(fact)

    for (y, q) in [k for k in REVENUE if k[0] >= 2024] + [(2023, 4)]:
        for yy in (y, y - 1):  # current period and prior-year comparative in the same filing
            if q == 4:
                if all((yy, i) in REVENUE for i in range(1, 5)):
                    rev = sum(REVENUE[(yy, i)] for i in range(1, 5))
                    add("Revenues", "USD", f"{yy}-01-01", f"{yy}-12-31", rev, y, q)
                    add("CostOfRevenue", "USD", f"{yy}-01-01", f"{yy}-12-31", rev * 0.6, y, q)
                    add("NetCashProvidedByUsedInOperatingActivities", "USD", f"{yy}-01-01", f"{yy}-12-31", rev * 0.2, y, q)
                    add("PaymentsToAcquirePropertyPlantAndEquipment", "USD", f"{yy}-01-01", f"{yy}-12-31", rev * 0.05, y, q)
                    add("EarningsPerShareDiluted", "USD/shares", f"{yy}-01-01", f"{yy}-12-31", rev / 100, y, q)
                continue
            if (yy, q) not in REVENUE:
                continue
            rev_q = REVENUE[(yy, q)]
            ytd = sum(REVENUE[(yy, i)] for i in range(1, q + 1))
            add("Revenues", "USD", qstart(yy, q), qend(yy, q), rev_q, y, q)
            add("CostOfRevenue", "USD", qstart(yy, q), qend(yy, q), rev_q * 0.6, y, q)
            add("EarningsPerShareDiluted", "USD/shares", qstart(yy, q), qend(yy, q), rev_q / 100, y, q)
            if q > 1:
                add("Revenues", "USD", f"{yy}-01-01", qend(yy, q), ytd, y, q)
            add("NetCashProvidedByUsedInOperatingActivities", "USD", f"{yy}-01-01", qend(yy, q), ytd * 0.2, y, q)
            add("PaymentsToAcquirePropertyPlantAndEquipment", "USD", f"{yy}-01-01", qend(yy, q), ytd * 0.05, y, q)

    # Balance sheet: point-in-time values at each period end (plus the prior year-end comparative).
    for (y, q) in [k for k in REVENUE if k[0] >= 2024] + [(2023, 4)]:
        ends = {qend(y, q), f"{y - 1}-12-31"}
        for end in ends:
            n = int(end[:4]) * 10 + int(end[5:7])
            add("AssetsCurrent", "USD", None, end, 200 + n % 50, y, q)
            add("LiabilitiesCurrent", "USD", None, end, 100.0, y, q)
            add("StockholdersEquity", "USD", None, end, 400.0, y, q)
            add("Assets", "USD", None, end, 1000.0, y, q)
            add("LongTermDebtNoncurrent", "USD", None, end, 150.0, y, q)
            add("AccountsReceivableNetCurrent", "USD", None, end, 60.0, y, q)

    # A restatement: the Q1 2025 10-Q restated Q1 2024 net income (later filing must win).
    add("NetIncomeLoss", "USD", "2024-01-01", "2024-03-31", 10.0, 2024, 1)
    add("NetIncomeLoss", "USD", "2024-01-01", "2024-03-31", 12.0, 2025, 1)
    add("NetIncomeLoss", "USD", "2025-01-01", "2025-03-31", 15.0, 2025, 1)
    return {"cik": 1, "entityName": "Acme Corp", "facts": {"us-gaap": us}}


@pytest.fixture
def raw_facts() -> dict:
    return build_raw()


@pytest.fixture
def facts(raw_facts) -> CompanyFacts:
    return CompanyFacts(raw_facts, ticker="ACME")


class FakeEDGAR:
    """Stands in for sec.filings.EDGAR in offline tests."""

    def __init__(self, facts: CompanyFacts):
        self._facts = facts
        self.company = Company(1, "ACME", "Acme Corp")

    def resolve_company(self, query: str) -> Company:
        return self.company

    def facts(self, company: Company) -> CompanyFacts:
        return self._facts


@pytest.fixture
def settings(tmp_path) -> Settings:
    return replace(Settings(), log_dir=tmp_path / "logs", brief_dir=tmp_path / "briefs", data_dir=tmp_path / "data")


@pytest.fixture
def session(facts, settings) -> AnalystSession:
    return AnalystSession("How did Acme's latest quarter compare to a year ago?", settings=settings, edgar=FakeEDGAR(facts))
