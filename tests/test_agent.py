"""Runs the real LangChain agent loop with a scripted chat model (no Ollama needed)."""

from __future__ import annotations

import json
from dataclasses import replace
from itertools import count
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from earnings_analyst.agent import extract_draft, run
from earnings_analyst.session import AnalystSession


class ScriptedChatModel(BaseChatModel):
    script: Callable[[int, list[BaseMessage]], AIMessage]
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any) -> ChatResult:
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.script(len(self.seen), messages))])


def tool_call(name: str, args: dict, i: int) -> dict:
    return {"name": name, "args": args, "id": f"call_{name}_{i}", "type": "tool_call"}


BRIEF = """# Acme Corp (ACME) - Q3 FY2025 earnings brief
**Question:** How did Acme's latest quarter compare to a year ago?
## Bottom line
Revenue grew 10.0% year over year [10-Q, Q3 FY2025, filed 2025-11-04] and gross margin held at 40.0%.
"""


def test_agent_computes_metrics_and_saves_brief(session):
    def script(step: int, messages: list[BaseMessage]) -> AIMessage:
        if step == 1:  # parallel tool calls in one step
            return AIMessage("", tool_calls=[
                tool_call("calculate_metric", {"ticker": "ACME", "metric": "revenue_growth_yoy"}, 1),
                tool_call("calculate_metric", {"ticker": "ACME", "metric": "gross_margin", "compare_to": "prior_year"}, 2),
            ])
        if step == 2:
            results = [json.loads(m.content) for m in messages if m.type == "tool"]
            assert {r["metric"] for r in results} == {"revenue_growth_yoy", "gross_margin"}
            return AIMessage("", tool_calls=[tool_call("save_brief", {
                "markdown": BRIEF, "ticker": "ACME", "period_label": "Q3 FY2025"}, 3)])
        return AIMessage("Brief saved.")

    result = run(session.question, model=ScriptedChatModel(script=script, seen=[]), session=session)
    assert result.brief_path and result.model_calls == 3
    assert [c["tool"] for c in result.tool_calls].count("calculate_metric") == 2
    log = [json.loads(line) for line in open(result.log_path)]
    events = [r["event"] for r in log]
    assert events[0] == "run_start" and events[-1] == "run_end"
    assert events.count("tool_end") == 3 and events.count("model_end") == 3
    data = json.loads(open(result.brief_path.replace(".md", ".json")).read())
    assert {m["metric"] for m in data["metrics"]} == {"revenue_growth_yoy", "gross_margin"}


def test_agent_stops_at_step_limit_and_is_told_to_save(facts, settings):
    from tests.conftest import FakeEDGAR

    session = AnalystSession("loop forever", settings=replace(settings, max_steps=4), edgar=FakeEDGAR(facts))
    ids = count()
    model = ScriptedChatModel(
        script=lambda step, msgs: AIMessage("", tool_calls=[
            tool_call("calculate_metric", {"ticker": "ACME", "metric": "net_margin"}, next(ids))]),
        seen=[],
    )
    result = run("loop forever", model=model, session=session)
    assert result.model_calls == 4
    last_prompt = model.seen[-1]
    assert isinstance(last_prompt[-1], HumanMessage) and "final step" in last_prompt[-1].content


def test_brief_written_as_text_is_sent_back_to_save_brief(session):
    def script(step: int, messages: list[BaseMessage]) -> AIMessage:
        if step == 1:
            return AIMessage("", tool_calls=[tool_call("calculate_metric", {"ticker": "ACME", "metric": "revenue_growth_yoy"}, 1)])
        if step == 2:
            return AIMessage(BRIEF + "\n" + "Filler sentence about results. " * 10)  # brief as plain text
        if step == 3:
            assert "Call the save_brief tool" in messages[-1].content
            return AIMessage("", tool_calls=[tool_call("save_brief", {
                "markdown": BRIEF, "ticker": "ACME", "period_label": "Q3 FY2025"}, 2)])
        return AIMessage("Saved.")

    result = run(session.question, model=ScriptedChatModel(script=script, seen=[]), session=session)
    assert result.brief_path and result.model_calls == 4


def test_brief_is_auto_saved_when_budget_runs_out(facts, settings):
    from tests.conftest import FakeEDGAR

    session = AnalystSession("q", settings=replace(settings, max_steps=2), edgar=FakeEDGAR(facts))

    def script(step: int, messages: list[BaseMessage]) -> AIMessage:
        if step == 1:
            return AIMessage("", tool_calls=[tool_call("calculate_metric", {"ticker": "ACME", "metric": "revenue_growth_yoy"}, 1)])
        return AIMessage(BRIEF + "\n" + "Filler sentence about results. " * 10)

    result = run("q", model=ScriptedChatModel(script=script, seen=[]), session=session)
    assert result.brief_path and "ACME_Q3-FY2025" in result.brief_path


def test_extract_draft_recovers_printed_tool_arguments():
    md = BRIEF + "Filler. " * 40
    assert extract_draft(json.dumps({"markdown": md, "ticker": "ACME"})) == md
    assert extract_draft("**Acme - Q3 brief**\n## Bottom line\n" + "x" * 400)
    assert extract_draft("The brief has been saved.") is None
