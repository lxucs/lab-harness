"""One query, start to finish: run the agent, keep what it did.

This is the whole of using the harness from Python — route to a provider,
choose tools, build an `Agent`, send a message, read what comes back. Around
that it does the three things an experiment wants: it records the run as a
session, so the HTML report comes for free; it writes the numbers the report
shows in its heading as data; and it writes the transcript as JSON.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

from lab_harness.agent import Agent
from lab_harness.agent.agent import Listener
from lab_harness.agent.types import declarations_for
from lab_harness.cli.export import export_html
from lab_harness.cli.usage import UsageSummary, cache_hit_rate, prompt_tokens, summarize
from lab_harness.config import Config
from lab_harness.llm.providers import route
from lab_harness.llm.serde import to_dict
from lab_harness.llm.types import (
    AssistantMessage,
    Message,
    TextBlock,
    Usage,
    UserMessage,
    text,
)
from lab_harness.session import start_recorder
from lab_harness.tools import AgentTool, ToolRegistry, default_registry
from search_agent.settings import Settings
from search_agent.tools import SEARCH_NAME, corpus_search

#: What the agent is told before the query. Short, because the query carries
#: the difficulty: it says where the corpus is and what to answer with.
SYSTEM_PROMPT = """\
Answer the query using the corpus in {corpus}/.

Search it with search_docs, then read the documents that look promising — the
corpus holds distractors, so the first hit is not always the answer. Reply with
just what was asked for, and name the docids you relied on.
"""


@dataclass(frozen=True)
class Outcome:
    """What one query produced, for the run's summary."""

    query_id: str
    answer: str
    requests: int
    tool_calls: int
    usage: Usage
    cost: float | None
    seconds: float
    directory: Path


def tools_for(names: tuple[str, ...], corpus: str) -> tuple[AgentTool, ...]:
    """The configured built-ins plus the corpus search, which is not built in."""
    builtins = [name for name in names if name != SEARCH_NAME]
    registry = ToolRegistry(default_registry().select(builtins))
    if SEARCH_NAME in names:
        registry.register(corpus_search(corpus))
    return tuple(registry)


async def run_query(
    query: dict[str, object],
    settings: Settings,
    config: Config,
    *,
    notify: Listener | None = None,
) -> Outcome:
    """Run one query, write its three files, and report what happened.

    Files land in `settings.out / <query_id>`: the session transcript, the
    HTML report built from it, and `trajectory.json`.
    """
    started = time.monotonic()
    model, provider = route(None, config)
    tools = tools_for(settings.tools, settings.corpus)
    question = settings.first_turn(str(query["query"]))
    identifier = str(query.get("query_id", "query"))
    system_prompt = SYSTEM_PROMPT.format(corpus=settings.corpus)

    directory = settings.out / identifier
    directory.mkdir(parents=True, exist_ok=True)
    # Named for the query, because that is the session id the recorder takes
    # from the filename — and the one the report shows in its heading.
    session = directory / f"{identifier}.jsonl"
    # A path makes the recorder append, which is how resume continues one; a
    # re-run of a query is a new run, not a second half of the last one.
    session.unlink(missing_ok=True)

    agent = Agent(
        model=model,
        provider=provider,
        system_prompt=system_prompt,
        tools=tools,
        config=config,
        cwd=str(settings.cwd),
    )
    # A path, not just a session id, so the transcript lands beside the rest of
    # this query's output instead of in the harness's session directory.
    recorder = start_recorder(
        cwd=str(settings.cwd),
        model=model,
        system_prompt=system_prompt,
        tools=declarations_for(tools),
        config=config,
        restored=0,
        path=session,
    )
    agent.subscribe(recorder)
    if notify is not None:
        agent.subscribe(notify)

    produced = await agent.prompt(question)
    recorder.close()
    report = export_html(session, config).rename(directory / "report.html")

    # `prompt` hands back what the run produced, not the message that asked for
    # it; a trajectory is the whole exchange, so the request goes in front.
    messages: list[Message] = [UserMessage(content=text(question)), *produced]
    answer = _answer(messages)
    summary = summarize(messages, config.models)
    seconds = time.monotonic() - started
    trajectory = _trajectory(
        query,
        question=question,
        answer=answer,
        system_prompt=system_prompt,
        model=model.id,
        tools=tools,
        config=config,
        messages=messages,
        summary=summary,
        session=session,
        report=report,
        seconds=seconds,
    )
    (directory / "trajectory.json").write_text(
        json.dumps(trajectory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return Outcome(
        query_id=identifier,
        answer=answer,
        requests=summary.requests,
        tool_calls=_tool_calls(messages),
        usage=summary.usage,
        cost=summary.cost,
        seconds=seconds,
        directory=directory,
    )


def _trajectory(
    query: dict[str, object],
    *,
    question: str,
    answer: str,
    system_prompt: str,
    model: str,
    tools: tuple[AgentTool, ...],
    config: Config,
    messages: list[Message],
    summary: UsageSummary,
    session: Path,
    report: Path,
    seconds: float,
) -> dict[str, object]:
    """The run as data: what was asked, what it spent, and every message."""
    usage = summary.usage
    return {
        "query_id": query.get("query_id"),
        "query": query.get("query"),
        "gold": query.get("answer"),
        "docs": {
            "gold": query.get("gold_docids", []),
            "evidence": query.get("evidence_docids", []),
            "negatives": query.get("negative_docids", []),
        },
        "model": model,
        "tools": [tool.name for tool in tools],
        "system_prompt": system_prompt,
        "request": question,
        "session": {"id": session.stem, "file": session.name, "report": report.name},
        "summary": {
            "turns": sum(1 for m in messages if isinstance(m, UserMessage)),
            "requests": summary.requests,
            "tool_calls": _tool_calls(messages),
            "input": usage.input,
            "output": usage.output,
            "cache_read": usage.cache_read,
            "cache_write": usage.cache_write,
            "prompt_tokens": prompt_tokens(usage),
            "cache_hit_rate": cache_hit_rate(usage),
            "cost": summary.cost,
            "currency": config.terminal.currency_symbol,
            "seconds": round(seconds, 1),
        },
        "answer": answer,
        "messages": [to_dict(m) for m in messages],
    }


def _tool_calls(messages: list[Message]) -> int:
    """Every tool call the run made."""
    return sum(len(m.tool_calls) for m in messages if isinstance(m, AssistantMessage))


def _answer(messages: list[Message]) -> str:
    """The agent's last word: the text of the final assistant message."""
    for message in reversed(messages):
        if isinstance(message, AssistantMessage):
            return "".join(block.text for block in message.content if isinstance(block, TextBlock))
    return ""
