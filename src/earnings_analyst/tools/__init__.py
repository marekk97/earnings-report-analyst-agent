"""The agent's toolset. Tools are built per session so they share one evidence ledger and call log."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from ..session import AnalystSession
from . import brief_tool, metric_tools, python_tool, segment_tools, text_tools


def build_tools(session: AnalystSession) -> list[BaseTool]:
    return [
        *metric_tools.build(session),
        *text_tools.build(session),
        *segment_tools.build(session),
        *python_tool.build(session),
        *brief_tool.build(session),
    ]
