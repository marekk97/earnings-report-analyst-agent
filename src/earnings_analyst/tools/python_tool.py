"""run_python: the logged fallback for calculations that are not (yet) in the metric registry.

Every call is appended to logs/run_python.jsonl with its stated purpose, so `earnings-analyst fallbacks`
can show which metrics are requested often enough to deserve a registry entry.

The code runs in a separate, isolated Python process (`-I`, empty temp working dir, scrubbed environment,
CPU/memory limits on POSIX, wall-clock timeout). This limits accidents; it is not a security sandbox.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time

from langchain_core.tools import BaseTool, tool

from ..provenance import error_result
from ..session import AnalystSession
from .common import tool_guard

TIMEOUT_S = 15
MAX_OUTPUT = 4000


def _limits() -> None:  # pragma: no cover - runs in the child process
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    except (ImportError, ValueError, OSError):
        pass


def execute(code: str, timeout: int = TIMEOUT_S) -> dict:
    with tempfile.TemporaryDirectory(prefix="earnings_py_") as tmp:
        started = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=tmp,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"},
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=_limits if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout", "message": f"Code ran longer than {timeout}s."}
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-MAX_OUTPUT:],
            "stderr": proc.stderr[-MAX_OUTPUT:],
            "seconds": round(time.monotonic() - started, 2),
        }


def build(session: AnalystSession) -> list[BaseTool]:
    @tool("run_python")
    @tool_guard(session, "run_python")
    def run_python(code: str, purpose: str, metric_name: str = "") -> dict:
        """FALLBACK ONLY: run a short Python snippet for a calculation that calculate_metric does not support.

        Use numbers you already obtained from other tools (hard-code them in the snippet) and print() the result.
        Never use this for registered metrics. The standard library is available; there is no network access.

        Args:
            code: Python source. Print the values you need.
            purpose: One sentence describing what is being calculated and why.
            metric_name: Short snake_case name for the calculation (e.g. "operating_leverage"), used to track
                which metrics should be added to the registry.
        """
        if not purpose.strip():
            return error_result("missing_purpose", "Describe the calculation in `purpose` so the fallback can be tracked.")
        result = execute(code)
        record = {"metric_name": metric_name or None, "purpose": purpose, "code": code, **result}
        session.log_python(record)
        session.ledger.python_calls.append(record)
        result["note"] = "Logged as a registry fallback. Cite the tool results whose numbers you used as inputs."
        return result

    return [run_python]
