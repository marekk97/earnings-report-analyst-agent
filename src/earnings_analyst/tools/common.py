"""Helpers shared by the tools: formatting, JSON output and uniform error handling."""

from __future__ import annotations

import functools
import json
import traceback
from typing import Any, Callable

from ..provenance import DataError, error_result
from ..session import AnalystSession


def fmt_usd(v: float) -> str:
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1e9:
        return f"{sign}${a / 1e9:,.2f}B"
    if a >= 1e6:
        return f"{sign}${a / 1e6:,.1f}M"
    if a >= 1e3:
        return f"{sign}${a / 1e3:,.1f}K"
    return f"{sign}${a:,.2f}"


def fmt_value(v: float, unit: str) -> str:
    if unit == "percent":
        return f"{v:.1f}%"
    if unit == "usd":
        return fmt_usd(v)
    if unit == "days":
        return f"{v:.1f} days"
    if unit == "times":
        return f"{v:.2f}x"
    return f"{v:.2f}"


def fmt_input(v: float, field: str) -> str:
    return f"${v:,.2f}" if field.startswith("eps") else fmt_usd(v)


def dump(obj: Any) -> str:
    # Compact JSON: tool results are re-sent to the model on every later step, so whitespace costs tokens.
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def tool_guard(session: AnalystSession, tool_name: str) -> Callable:
    """Turn exceptions into structured errors so the model always gets an explainable result."""

    def deco(fn: Callable[..., dict]) -> Callable[..., str]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> str:
            try:
                result = fn(*args, **kwargs)
            except DataError as exc:
                result = exc.to_dict()
            except Exception as exc:  # unexpected bug: still structured, and logged with a traceback
                session.log("tool_exception", tool=tool_name, args=kwargs, traceback=traceback.format_exc())
                result = error_result("internal_error", f"{type(exc).__name__}: {exc}")
            if not result.get("ok", True):
                session.ledger.errors.append({"tool": tool_name, "args": kwargs, **result})
            return dump(result)

        return wrapper

    return deco
