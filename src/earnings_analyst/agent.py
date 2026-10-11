"""The LangChain agent: an open model (Ollama local or Ollama Cloud) in a step-limited tool loop.

The model plans and explains; the tools fetch and calculate. Every model call and tool call is written to
logs/runs/<run_id>.jsonl.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelCallLimitMiddleware, hook_config
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import LLMResult

from .config import Settings, get_settings
from .prompts import system_prompt
from .session import AnalystSession
from .tools import build_tools


def build_model(settings: Settings | None = None) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    s = settings or get_settings()
    client_kwargs: dict[str, Any] = {}
    if s.ollama_api_key:
        client_kwargs["headers"] = {"Authorization": f"Bearer {s.ollama_api_key}"}
    reasoning: bool | str | None = {"": None, "true": True, "false": False}.get(s.reasoning.lower(), s.reasoning)
    return ChatOllama(
        model=s.model,
        base_url=s.ollama_host,
        temperature=s.temperature,
        num_ctx=s.num_ctx,
        reasoning=reasoning,
        client_kwargs=client_kwargs,
    )


class RunLogger(BaseCallbackHandler):
    """Writes every model call and tool call (inputs and outputs) to the session's JSONL log."""

    def __init__(self, session: AnalystSession):
        self.session = session
        self.model_calls = 0
        self.tool_calls: list[dict[str, Any]] = []
        self._tool_starts: dict[UUID, tuple[str, Any, float]] = {}

    def on_chat_model_start(self, serialized: dict, messages: list[list[BaseMessage]], *, run_id: UUID, **kw: Any) -> None:
        self.model_calls += 1
        self.session.log("model_start", step=self.model_calls, n_messages=len(messages[0]) if messages else 0)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kw: Any) -> None:
        gen = response.generations[0][0] if response.generations and response.generations[0] else None
        msg = getattr(gen, "message", None)
        self.session.log(
            "model_end",
            step=self.model_calls,
            content=getattr(msg, "content", None),
            tool_calls=[{"name": c["name"], "args": c["args"]} for c in getattr(msg, "tool_calls", []) or []],
            usage=getattr(msg, "usage_metadata", None),
        )

    def on_tool_start(self, serialized: dict, input_str: str, *, run_id: UUID, inputs: dict | None = None, **kw: Any) -> None:
        name = (serialized or {}).get("name") or kw.get("name") or "tool"
        self._tool_starts[run_id] = (name, inputs if inputs is not None else input_str, time.monotonic())
        self.session.log("tool_start", tool=name, args=inputs if inputs is not None else input_str)

    def on_tool_end(self, output: Any, *, run_id: UUID, **kw: Any) -> None:
        name, args, started = self._tool_starts.pop(run_id, ("tool", None, time.monotonic()))
        content = getattr(output, "content", output)
        self.tool_calls.append({"tool": name, "args": args})
        self.session.log("tool_end", tool=name, seconds=round(time.monotonic() - started, 2), output=content)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kw: Any) -> None:
        name, args, _ = self._tool_starts.pop(run_id, ("tool", None, 0.0))
        self.tool_calls.append({"tool": name, "args": args, "error": str(error)})
        self.session.log("tool_error", tool=name, args=args, error=repr(error))


def extract_draft(text: str) -> str | None:
    """A brief the model wrote as plain text, including save_brief arguments printed as JSON instead of called."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").removeprefix("json").removeprefix("markdown").strip()
    if t.startswith("{"):
        try:
            md = json.loads(t).get("markdown")
            if isinstance(md, str) and len(md) > 200:
                return md
        except (ValueError, AttributeError):
            pass
    if len(t) > 300 and (t.startswith(("#", "**")) or "bottom line" in t.lower()):
        return t
    return None


NUDGE = (
    "No brief has been saved yet. Call the save_brief tool now, as a real tool call rather than text, with the "
    "full Markdown brief. If the user asked for investment advice, do not refuse: write the factual brief from the "
    "tool results and state that it contains no recommendation."
)


class StepBudgetMiddleware(AgentMiddleware):
    """Keeps the run on track to a saved brief.

    * Before the final allowed model call, tell the model to save the brief with what it has.
    * If the model stops without having saved a brief (wrote it as text, printed the tool arguments, claimed to
      have saved it, or refused), send it back once with an explicit instruction to call save_brief.
    """

    def __init__(self, session: AnalystSession, max_steps: int):
        super().__init__()
        self.session = session
        self.max_steps = max_steps

    def before_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        msgs = state["messages"]
        last_human = max((i for i, m in enumerate(msgs) if isinstance(m, HumanMessage)), default=-1)
        steps_done = sum(isinstance(m, AIMessage) for m in msgs[last_human + 1:])
        if steps_done == self.max_steps - 1 and not self.session.ledger.brief_path:
            return {"messages": [HumanMessage(
                "This is your final step. Call save_brief now with the best brief you can write from the "
                "tool results you already have; list anything you could not retrieve under Caveats."
            )]}
        return None

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        msgs = state["messages"]
        last = msgs[-1] if msgs else None
        if not isinstance(last, AIMessage) or last.tool_calls or self.session.ledger.brief_path:
            return None
        if any(isinstance(m, HumanMessage) and m.content == NUDGE for m in msgs):
            return None
        self.session.log("nudge_save_brief", last_content=last.text[:300])
        return {"jump_to": "model", "messages": [HumanMessage(NUDGE)]}


def build_agent(session: AnalystSession, model: BaseChatModel | None = None):
    settings = session.settings
    return create_agent(
        model or build_model(settings),
        tools=build_tools(session),
        system_prompt=system_prompt(settings.max_steps),
        middleware=[
            StepBudgetMiddleware(session, settings.max_steps),
            ModelCallLimitMiddleware(run_limit=settings.max_steps, exit_behavior="end"),
        ],
    )


@dataclass
class RunResult:
    question: str
    run_id: str
    answer: str
    brief_path: str | None
    model_calls: int
    tool_calls: list[dict[str, Any]]
    seconds: float
    log_path: str
    session: AnalystSession = field(repr=False)


def run(question: str, settings: Settings | None = None, model: BaseChatModel | None = None,
        session: AnalystSession | None = None) -> RunResult:
    session = session or AnalystSession(question, settings=settings)
    session.question = question
    logger = RunLogger(session)
    s = session.settings
    session.log("run_start", question=question, model=getattr(model, "model", None) or s.model, max_steps=s.max_steps)
    agent = build_agent(session, model)
    started = time.monotonic()
    try:
        state = agent.invoke(
            {"messages": [HumanMessage(question)]},
            config={"callbacks": [logger], "recursion_limit": 8 * s.max_steps + 10},
        )
    except Exception as exc:
        session.log("run_error", error=repr(exc))
        raise
    final = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
    answer = final.text if final is not None else ""
    draft = next((d for m in reversed(state["messages"]) if isinstance(m, AIMessage) and (d := extract_draft(m.text))), None)
    if not session.ledger.brief_path and draft:
        # Last resort (e.g. step budget exhausted): save the model's brief through the real tool.
        save = next(t for t in build_tools(session) if t.name == "save_brief")
        ticker = next((m["ticker"] for m in session.ledger.metrics), "UNKNOWN")
        period = next((m["period"]["label"] for m in session.ledger.metrics), "latest")
        out = save.invoke({"markdown": draft, "ticker": ticker, "period_label": period})
        session.log("auto_save_brief", output=out)
    result = RunResult(
        question=question,
        run_id=session.run_id,
        answer=answer if isinstance(answer, str) else str(answer),
        brief_path=session.ledger.brief_path,
        model_calls=logger.model_calls,
        tool_calls=logger.tool_calls,
        seconds=round(time.monotonic() - started, 1),
        log_path=str(session.run_log_path),
        session=session,
    )
    session.log("run_end", brief_path=result.brief_path, model_calls=result.model_calls,
                tool_calls=len(result.tool_calls), seconds=result.seconds)
    return result
