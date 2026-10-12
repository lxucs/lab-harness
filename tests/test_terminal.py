"""The interactive terminal, driven by a scripted input session.

prompt_toolkit's real PromptSession needs a tty, so `_session` is replaced with
a scripted stand-in. Everything else — the run loop, cancellation handling, the
approval prompt, exit conditions — is the production code path.
"""

import asyncio
import io

import pytest
from rich.console import Console

from harness.agent import Agent
from harness.cli.terminal import Terminal, build_approval_gate
from harness.config import Config
from harness.llm.fake import FAKE_MODEL, FakeProvider, text_response, tool_call_response
from harness.llm.types import TextBlock, ToolCallBlock


class ScriptedSession:
    """Replays lines, then raises EOFError like Ctrl-D."""

    def __init__(self, lines: list[str | type[BaseException]]) -> None:
        self._lines = list(lines)
        self.prompts: list[str] = []

    async def prompt_async(self, message: str = "") -> str:
        self.prompts.append(message)
        if not self._lines:
            raise EOFError
        item = self._lines.pop(0)
        if isinstance(item, type) and issubclass(item, BaseException):
            raise item
        return str(item)


def _terminal(lines: list, config: Config | None = None) -> tuple[Terminal, io.StringIO]:
    buffer = io.StringIO()
    console = Console(file=buffer, width=90, force_terminal=False, no_color=True)
    terminal = Terminal(console=console, config=config or Config())
    terminal._session = ScriptedSession(lines)  # type: ignore[assignment]
    return terminal, buffer


def _attach(terminal: Terminal, provider: FakeProvider, **kwargs: object) -> Agent:
    agent = Agent(model=FAKE_MODEL, provider=provider, config=terminal.config, **kwargs)  # type: ignore[arg-type]
    terminal.attach(agent)
    return agent


# --- the loop -------------------------------------------------------------


async def test_a_prompt_runs_a_turn_and_renders_the_answer() -> None:
    terminal, buffer = _terminal(["hello"])
    _attach(terminal, FakeProvider([text_response("Hi there")]))

    assert await terminal.run() == 0
    assert "Hi there" in buffer.getvalue()


async def test_eof_exits_cleanly() -> None:
    terminal, _ = _terminal([])
    _attach(terminal, FakeProvider([]))

    assert await terminal.run() == 0


async def test_blank_lines_are_ignored() -> None:
    """An empty line must not burn a provider request."""
    terminal, _ = _terminal(["", "   ", "real"])
    provider = FakeProvider([text_response("answer")])
    _attach(terminal, provider)

    await terminal.run()

    assert len(provider.requests) == 1


@pytest.mark.parametrize("word", [":q", ":quit", "exit"])
async def test_exit_words_leave_the_session(word: str) -> None:
    terminal, _ = _terminal([word, "never reached"])
    provider = FakeProvider([text_response("unused")])
    _attach(terminal, provider)

    assert await terminal.run() == 0
    assert provider.requests == []


async def test_successive_turns_share_one_transcript() -> None:
    terminal, _ = _terminal(["first", "second"])
    provider = FakeProvider([text_response("one"), text_response("two")])
    agent = _attach(terminal, provider)

    await terminal.run()

    assert [m.role for m in agent.state.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert len(provider.requests[1].messages) == 3


async def test_ctrl_c_at_an_idle_prompt_does_not_exit() -> None:
    """It clears the line; only Ctrl-D ends the session."""
    terminal, buffer = _terminal([KeyboardInterrupt, "after interrupt"])
    provider = FakeProvider([text_response("still here")])
    _attach(terminal, provider)

    assert await terminal.run() == 0
    assert "still here" in buffer.getvalue()


async def test_a_runtime_error_ends_the_turn_not_the_session() -> None:
    class Exploding:
        name = "boom"

        def stream(self, *args: object, **kwargs: object):
            raise RuntimeError("provider exploded")

    terminal, buffer = _terminal(["go", "again"])
    agent = Agent(model=FAKE_MODEL, provider=Exploding())  # type: ignore[arg-type]
    terminal.attach(agent)

    assert await terminal.run() == 0
    assert "provider exploded" in buffer.getvalue()


# --- cancellation ---------------------------------------------------------


async def test_ctrl_c_during_a_run_cancels_the_turn_and_keeps_the_session() -> None:
    """The core promise: cancel the turn, survive to the next prompt."""
    from pydantic import BaseModel

    from harness.tools.base import ToolContext, ToolResult, ok, tool

    class _P(BaseModel):
        pass

    @tool(name="slow", description="slow", execution_mode="sequential")
    async def slow(p: _P, ctx: ToolContext) -> ToolResult:
        for _ in range(200):
            if ctx.signal is not None and ctx.signal.is_set():
                return ok("stopped early")
            await asyncio.sleep(0.01)
        return ok("ran to completion")

    call = ToolCallBlock(id="c1", name="slow", arguments={})
    terminal, _ = _terminal(["slow thing", "next turn"])
    provider = FakeProvider(
        [tool_call_response([call]), text_response("after abort"), text_response("next answer")]
    )
    agent = _attach(terminal, provider, tools=(slow,))

    original = terminal._run_turn

    async def interrupting_run_turn(message: str) -> None:
        task = asyncio.create_task(original(message))
        if message == "slow thing":
            await asyncio.sleep(0.05)
            agent.abort()  # what a KeyboardInterrupt triggers
        await task

    terminal._run_turn = interrupting_run_turn  # type: ignore[method-assign]

    assert await terminal.run() == 0

    assert any(getattr(m, "stop_reason", None) == "aborted" for m in agent.state.messages), (
        "the cancelled turn must be recorded as aborted"
    )

    tool_results = [m for m in agent.state.messages if m.role == "tool_result"]
    body = "".join(b.text for b in tool_results[0].content if isinstance(b, TextBlock))
    assert body == "stopped early"

    last = agent.state.messages[-1]
    assert last.role == "assistant"
    assert "".join(b.text for b in last.content if isinstance(b, TextBlock))


# --- approval prompt ------------------------------------------------------


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("y", "yes"),
        ("yes", "yes"),
        ("Y", "yes"),
        ("n", "no"),
        ("no", "no"),
        ("N", "no"),
        ("a", "always"),
        ("always", "always"),
        ("A", "always"),
        ("", "no"),
        ("maybe", "no"),
        ("q", "no"),
    ],
)
async def test_approval_answers_map_as_documented(typed: str, expected: str) -> None:
    terminal, _ = _terminal([typed])

    assert await terminal.ask_approval("bash", {"command": "ls"}) == expected


async def test_an_empty_answer_denies() -> None:
    """Fail closed: hitting Enter is not consent."""
    terminal, _ = _terminal([""])

    assert await terminal.ask_approval("bash", {}) == "no"


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, EOFError])
async def test_interrupting_the_approval_question_denies(interrupt: type) -> None:
    terminal, _ = _terminal([interrupt])

    assert await terminal.ask_approval("bash", {}) == "no"


async def test_the_question_names_the_tool() -> None:
    terminal, _ = _terminal(["y"])

    await terminal.ask_approval("write", {"path": "a.txt"})

    session = terminal._session
    assert "write" in session.prompts[-1]  # type: ignore[attr-defined]


async def test_a_denied_tool_does_not_run_through_the_terminal(tmp_path) -> None:
    """End to end: terminal answer -> gate -> loop -> tool never executes."""
    from harness.tools import default_registry

    config = Config()
    terminal, buffer = _terminal(["write a file", "n"], config)
    gate = build_approval_gate(terminal, config)
    call = ToolCallBlock(id="c1", name="write", arguments={"path": "new.txt", "content": "x"})
    provider = FakeProvider([tool_call_response([call]), text_response("blocked then")])
    _attach(
        terminal,
        provider,
        tools=default_registry().select(["write"]),
        cwd=str(tmp_path),
        before_tool_call=gate,
    )

    await terminal.run()

    assert not (tmp_path / "new.txt").exists(), "a denied write must not touch the disk"
    assert "denied" in buffer.getvalue().lower()


async def test_an_approved_tool_runs_through_the_terminal(tmp_path) -> None:
    from harness.tools import default_registry

    config = Config()
    terminal, _ = _terminal(["write a file", "y"], config)
    gate = build_approval_gate(terminal, config)
    call = ToolCallBlock(id="c1", name="write", arguments={"path": "new.txt", "content": "hi"})
    provider = FakeProvider([tool_call_response([call]), text_response("done")])
    _attach(
        terminal,
        provider,
        tools=default_registry().select(["write"]),
        cwd=str(tmp_path),
        before_tool_call=gate,
    )

    await terminal.run()

    assert (tmp_path / "new.txt").read_text() == "hi"


# --- wiring ---------------------------------------------------------------


def test_running_without_an_attached_agent_is_a_clear_error() -> None:
    terminal, _ = _terminal([])

    with pytest.raises(RuntimeError, match="attach"):
        _ = terminal.agent
