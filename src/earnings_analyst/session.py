"""Per-run state: the evidence ledger every tool writes to, and the JSONL call log."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .sec.filings import EDGAR


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Ledger:
    """Everything the agent learned from tools during one run. save_brief serialises this as the brief's JSON."""

    metrics: list[dict[str, Any]] = field(default_factory=list)
    histories: list[dict[str, Any]] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    python_calls: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    text_snippets: list[str] = field(default_factory=list)  # fetched filing text, for grounding checks
    brief_path: str | None = None

    def add_source(self, source: dict[str, Any]) -> None:
        key = (source.get("accession"), source.get("kind"), source.get("section"))
        if not any((s.get("accession"), s.get("kind"), s.get("section")) == key for s in self.sources):
            self.sources.append(source)


class AnalystSession:
    def __init__(
        self,
        question: str = "",
        settings: Settings | None = None,
        edgar: EDGAR | None = None,
        run_id: str | None = None,
    ):
        self.settings = settings or get_settings()
        self.edgar = edgar or EDGAR()
        self.question = question
        self.run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.ledger = Ledger()
        self.log_dir = Path(self.settings.log_dir)
        self.run_log_path = self.log_dir / "runs" / f"{self.run_id}.jsonl"
        self.python_log_path = self.log_dir / "run_python.jsonl"

    def log(self, event: str, **data: Any) -> None:
        self.run_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": utcnow(), "run_id": self.run_id, "event": event, **data}
        with self.run_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def log_python(self, record: dict[str, Any]) -> None:
        self.python_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.python_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": utcnow(), "run_id": self.run_id, "question": self.question, **record}, default=str) + "\n")
