"""Provenance records and structured errors shared by every tool.

Design rules from the brief:
* Every tool result carries provenance: form type, fiscal period and filing date.
* Missing data is a structured error (e.g. "inventory not reported"), never an empty result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def filing_index_url(cik: int, accession: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{accession}-index.htm"


@dataclass
class Provenance:
    form: str
    fiscal_period: str
    filing_date: str
    accession: str
    url: str
    period_end: str | None = None
    period_start: str | None = None
    concept: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def citation(self) -> str:
        return f"{self.form}, {self.fiscal_period}, filed {self.filing_date}"


class DataError(Exception):
    """A recoverable, explainable data problem that tools turn into a structured error."""

    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return error_result(self.code, self.message, **self.details)


def error_result(code: str, message: str, **details: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": False, "error": code, "message": message}
    out.update({k: v for k, v in details.items() if v is not None})
    return out


@dataclass
class SourceRecord:
    """A filing consulted during a run (used to assemble the brief's source list)."""

    kind: str
    form: str
    fiscal_period: str
    filing_date: str
    accession: str
    url: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(d.pop("extra"))
        return d
