"""Runtime settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_SEC_USER_AGENT = "EarningsAnalyst set-SEC_USER_AGENT@example.com"


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    ollama_api_key: str | None = field(default_factory=lambda: os.getenv("OLLAMA_API_KEY") or None)
    model: str = field(default_factory=lambda: _env("EARNINGS_MODEL", "qwen3.5:9b"))
    temperature: float = field(default_factory=lambda: float(_env("EARNINGS_TEMPERATURE", "0")))
    num_ctx: int = field(default_factory=lambda: int(_env("EARNINGS_NUM_CTX", "32768")))
    max_steps: int = field(default_factory=lambda: int(_env("EARNINGS_MAX_STEPS", "10")))
    # "true"/"false" toggles thinking on models that support it; "low"/"medium"/"high" for gpt-oss. Empty = model default.
    reasoning: str = field(default_factory=lambda: _env("EARNINGS_REASONING", ""))
    sec_user_agent: str = field(default_factory=lambda: _env("SEC_USER_AGENT", DEFAULT_SEC_USER_AGENT))
    data_dir: Path = field(default_factory=lambda: Path(_env("EARNINGS_DATA_DIR", "data")))
    log_dir: Path = field(default_factory=lambda: Path(_env("EARNINGS_LOG_DIR", "logs")))
    brief_dir: Path = field(default_factory=lambda: Path(_env("EARNINGS_BRIEF_DIR", "briefs")))

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


def get_settings() -> Settings:
    return Settings()
