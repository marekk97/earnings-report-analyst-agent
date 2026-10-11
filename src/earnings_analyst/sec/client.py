"""Thin SEC EDGAR HTTP client: polite rate limiting, a descriptive User-Agent and a disk cache."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from ..config import Settings, get_settings
from ..provenance import DataError

# SEC asks for no more than 10 requests per second.
_MIN_INTERVAL = 0.12
# Mutable endpoints (new filings appear) are refreshed after this many seconds;
# Archives documents are immutable once filed and are cached forever.
_MUTABLE_TTL = 6 * 3600


class SECClient:
    def __init__(self, settings: Settings | None = None, cache_dir: Path | None = None, offline: bool = False):
        self.settings = settings or get_settings()
        self.cache_dir = Path(cache_dir or self.settings.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self._lock = threading.Lock()
        self._last_request = 0.0
        self._http = httpx.Client(
            headers={"User-Agent": self.settings.sec_user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=30.0,
            follow_redirects=True,
        )

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode()).hexdigest()[:32]
        return self.cache_dir / digest[:2] / digest

    def get_text(self, url: str) -> str:
        immutable = "/Archives/edgar/data/" in url and not url.endswith("index.json")
        path = self._cache_path(url)
        if path.exists() and (immutable or self.offline or time.time() - path.stat().st_mtime < _MUTABLE_TTL):
            return path.read_text(encoding="utf-8")
        if self.offline:
            raise DataError("offline", f"No cached copy of {url} and offline mode is on.")
        text = self._fetch(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return text

    def get_json(self, url: str) -> Any:
        return json.loads(self.get_text(url))

    def _fetch(self, url: str) -> str:
        for attempt in range(4):
            with self._lock:
                wait = _MIN_INTERVAL - (time.monotonic() - self._last_request)
                if wait > 0:
                    time.sleep(wait)
                self._last_request = time.monotonic()
            try:
                resp = self._http.get(url)
            except httpx.HTTPError as exc:
                if attempt == 3:
                    raise DataError("sec_unreachable", f"Could not reach SEC EDGAR: {exc}", url=url) from exc
                time.sleep(1.5 * (attempt + 1))
                continue
            if resp.status_code == 404:
                raise DataError("not_found", f"SEC returned 404 for {url}", url=url)
            if resp.status_code in (429, 503) or resp.status_code >= 500:
                time.sleep(2.0 * (attempt + 1))
                continue
            if resp.status_code == 403:
                raise DataError(
                    "sec_forbidden",
                    "SEC rejected the request (403). Set SEC_USER_AGENT to 'AppName your-email@domain.com'.",
                    url=url,
                )
            resp.raise_for_status()
            return resp.text
        raise DataError("sec_unavailable", f"SEC EDGAR kept failing for {url}", url=url)
