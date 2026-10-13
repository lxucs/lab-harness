"""The interactive terminal, driven by a scripted input session.

prompt_toolkit's real PromptSession needs a tty, so `_session` is replaced with
a scripted stand-in. Everything else — the run loop, cancellation handling, the
approval prompt, exit conditions — is the production code path.
"""

import asyncio
import io
import re
import signal
import threading

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from harness.agent import Agent
from harness.cli.terminal import Terminal, _ctrl_c_calls, build_approval_gate
from harness.config import Config, PricingConfig, TerminalConfig
from harness.llm.fake import FAKE_MODEL, FakeProvider, text_response, tool_call_response
from harness.llm.types import AssistantMessage, TextBlock, ToolCallBlock, Usage, UserMessage, text


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
    """The core promise: cancel the turn, survive to the next prompt.

    A real SIGINT, because that is what Ctrl-C is during a run: prompt_toolkit
    is not reading, so the terminal is not in raw mode. Under asyncio.run, an
    unhandled one cancels the main task and ends the session.
    """
    from pydantic import BaseModel

    from harness.tools.base import ToolContext, ToolResult, ok, tool

    class _P(BaseModel):
        pass

    @tool(name="slow", description="slow", execution_mode="sequential")
    async def slow(p: _P, ctx: ToolContext) -> ToolResult:
        signal.raise_signal(signal.SIGINT)
        for _ in range(200):
            if ctx.signal is not None and ctx.signal.is_set():
                return ok("stopped early")
            await asyncio.sleep(0.01)
        return ok("ran to completion")

    call = ToolCallBlock(id="c1", name="slow", arguments={})
    terminal, buffer = _terminal(["slow thing", "next turn"])
    provider = FakeProvider(
        [tool_call_response([call]), text_response("after abort"), text_response("next answer")]
    )
    agent = _attach(terminal, provider, tools=(slow,))

    # Stands in for asyncio.run's handler. Catching the signal here, rather
    # than letting it raise KeyboardInterrupt, keeps a regression a failed
    # assertion instead of an interrupted test session.
    reached_previous: list[int] = []

    def previous(signum: int, frame: object) -> None:
        reached_previous.append(signum)

    original = signal.signal(signal.SIGINT, previous)
    try:
        assert await terminal.run() == 0
        assert signal.getsignal(signal.SIGINT) is previous, "the turn must restore the handler"
    finally:
        signal.signal(signal.SIGINT, original)

    assert reached_previous == [], "Ctrl-C during a turn must not reach the outer handler"
    assert "turn cancelled" in buffer.getvalue()
    assert any(getattr(m, "stop_reason", None) == "aborted" for m in agent.state.messages), (
        "the cancelled turn must be recorded as aborted"
    )

    tool_results = [m for m in agent.state.messages if m.role == "tool_result"]
    body = "".join(b.text for b in tool_results[0].content if isinstance(b, TextBlock))
    assert body == "stopped early"

    last = agent.state.messages[-1]
    assert last.role == "assistant"
    assert "".join(b.text for b in last.content if isinstance(b, TextBlock))


def test_a_terminal_off_the_main_thread_leaves_ctrl_c_alone() -> None:
    """Only the main thread may handle signals; an embedder elsewhere must still run."""
    terminal, buffer = _terminal(["hello"])
    _attach(terminal, FakeProvider([text_response("Hi there")]))
    before = signal.getsignal(signal.SIGINT)
    exit_codes: list[int] = []

    thread = threading.Thread(target=lambda: exit_codes.append(asyncio.run(terminal.run())))
    thread.start()
    thread.join(timeout=10)

    assert exit_codes == [0]
    assert "Hi there" in buffer.getvalue()
    assert signal.getsignal(signal.SIGINT) is before


async def test_ctrl_c_still_cancels_after_an_approval_question() -> None:
    """A real prompt_toolkit prompt mid-turn must not take the handler with it.

    prompt_toolkit swaps SIGINT handling around every prompt; a handler it does
    not restore would leave Ctrl-C doing nothing for the rest of the turn.
    """
    calls: list[str] = []
    with create_pipe_input() as pipe, _ctrl_c_calls(lambda: calls.append("cancel")):
        pipe.send_text("y\n")
        session: PromptSession[str] = PromptSession(input=pipe, output=DummyOutput())
        assert await session.prompt_async("Allow bash? ") == "y"

        signal.raise_signal(signal.SIGINT)
        await asyncio.sleep(0.05)

    assert calls == ["cancel"]


# --- usage ----------------------------------------------------------------
#
# FakeProvider reports 10 input and 5 output tokens for every request.


async def test_a_run_ends_with_a_usage_footer() -> None:
    terminal, buffer = _terminal(["hello"])
    _attach(terminal, FakeProvider([text_response("Hi")]))

    await terminal.run()

    assert "10 in · 5 out" in buffer.getvalue()


async def test_the_footer_covers_every_request_in_the_run() -> None:
    """A tool call makes two requests for one prompt; the footer is per run."""
    call = ToolCallBlock(id="c1", name="nothing", arguments={})
    terminal, buffer = _terminal(["go"])
    _attach(terminal, FakeProvider([tool_call_response([call]), text_response("done")]))

    await terminal.run()

    assert "20 in · 10 out" in buffer.getvalue()


async def test_the_footer_can_be_turned_off() -> None:
    config = Config(terminal=TerminalConfig(show_usage=False))
    terminal, buffer = _terminal(["hello"], config)
    _attach(terminal, FakeProvider([text_response("Hi")]))

    await terminal.run()

    assert "10 in" not in buffer.getvalue()


async def test_the_footer_shows_a_cost_once_the_model_is_priced() -> None:
    config = Config(pricing={FAKE_MODEL.id: PricingConfig(input=1_000.0, output=1_000.0)})
    terminal, buffer = _terminal(["hello"], config)
    _attach(terminal, FakeProvider([text_response("Hi")]))

    await terminal.run()

    # (10 + 5) tokens x $1,000 per million.
    assert "$0.0150" in buffer.getvalue()


async def test_an_unpriced_model_shows_no_cost() -> None:
    terminal, buffer = _terminal(["hello"])
    _attach(terminal, FakeProvider([text_response("Hi")]))

    await terminal.run()

    assert "$" not in buffer.getvalue()


async def test_status_shows_session_totals_without_a_request() -> None:
    terminal, buffer = _terminal(["first", "second", ":status"])
    provider = FakeProvider([text_response("one"), text_response("two")])
    _attach(terminal, provider)

    assert await terminal.run() == 0

    assert len(provider.requests) == 2, ":status must not reach the provider"
    status = buffer.getvalue().split("two")[-1]
    assert re.search(r"requests\s+2\b", status)
    assert re.search(r"output\s+10\b", status)


async def test_status_includes_resumed_history() -> None:
    """Restored messages never pass a subscriber, so totals come from the transcript."""
    terminal, buffer = _terminal([":status"])
    agent = _attach(terminal, FakeProvider([]))
    agent.restore(
        [
            UserMessage(content=text("earlier")),
            AssistantMessage(
                content=text("old answer"),
                model=FAKE_MODEL.id,
                provider="fake",
                usage=Usage(input=4_000, output=100),
            ),
        ]
    )

    await terminal.run()

    output = buffer.getvalue()
    assert re.search(r"requests\s+1\b", output)
    assert re.search(r"output\s+100\b", output)


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
