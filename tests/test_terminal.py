"""The interactive terminal, driven by a scripted input session.

prompt_toolkit's real PromptSession needs a tty, so `_session` is replaced with
a scripted stand-in. Everything else — the run loop, cancellation handling, the
approval prompt, exit conditions — is the production code path.
"""

import asyncio
import io
import logging
import re
import signal
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.application import create_app_session
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from lab_harness.agent import Agent
from lab_harness.agent.types import AgentStart, MessageStart
from lab_harness.cli.commands import Command
from lab_harness.cli.renderers import GLYPH, EventRenderer
from lab_harness.cli.terminal import (
    LogStream,
    Terminal,
    _bindings,
    _ctrl_c_calls,
    _keys_wait_quietly,
    build_approval_gate,
)
from lab_harness.config import (
    Config,
    ConfigError,
    PricingConfig,
    SessionConfig,
    TerminalConfig,
    ThemeConfig,
    lab_harness_home,
)
from lab_harness.llm.fake import FAKE_MODEL, FakeProvider, text_response, tool_call_response
from lab_harness.llm.types import (
    AssistantMessage,
    TextBlock,
    ToolCallBlock,
    Usage,
    UserMessage,
    text,
)
from lab_harness.logs import configure_logging
from tests.entries import model_entry

#: Stands in for a command list that is never up, so Alt-Enter keeps its newline.
_never_listing = Condition(lambda: False)


class ScriptedSession:
    """Replays lines, then raises EOFError like Ctrl-D."""

    def __init__(self, lines: list[str | type[BaseException]]) -> None:
        self._lines = list(lines)
        self.prompts: list[object] = []
        #: What the terminal asked to draw at the right of each input.
        self.rprompts: list[object] = []

    async def prompt_async(self, message: object = "", *, rprompt: object = None) -> str:
        self.prompts.append(message)
        self.rprompts.append(rprompt)
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


async def test_blank_lines_are_ignored() -> None:
    """An empty line must not burn a provider request."""
    terminal, _ = _terminal(["", "   ", "real"])
    provider = FakeProvider([text_response("answer")])
    _attach(terminal, provider)

    await terminal.run()

    assert len(provider.requests) == 1


@pytest.mark.parametrize("word", ["/exit", "/quit", "exit"])
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
    assert not terminal.renderer.active, "a crashed turn must not leave its region up"


# --- cancellation ---------------------------------------------------------


async def test_ctrl_c_during_a_run_cancels_the_turn_and_keeps_the_session() -> None:
    """The core promise: cancel the turn, survive to the next prompt.

    A real SIGINT, because that is what Ctrl-C is during a run: prompt_toolkit
    is not reading, so the terminal is not in raw mode. Under asyncio.run, an
    unhandled one cancels the main task and ends the session.
    """
    from pydantic import BaseModel

    from lab_harness.tools.base import ToolContext, ToolResult, ok, tool

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
    price = PricingConfig(input=1_000.0, output=1_000.0)
    config = Config(models={FAKE_MODEL.id: model_entry(price=price)})
    terminal, buffer = _terminal(["hello"], config)
    _attach(terminal, FakeProvider([text_response("Hi")]))

    await terminal.run()

    # (10 + 5) tokens x $1,000 per million.
    assert "$0.0150" in buffer.getvalue()


# --- the status beside the input ------------------------------------------


async def test_a_run_leaves_the_model_and_the_context_beside_the_input() -> None:
    """15 tokens of a 20-token window: `fake-model · context 75%`, dim."""
    terminal, _ = _terminal(["hello"])
    model = replace(FAKE_MODEL, context_window=20)
    agent = Agent(model=model, provider=FakeProvider([text_response("Hi")]), config=terminal.config)
    terminal.attach(agent)

    await terminal.run()

    status = terminal._rprompt()  # type: ignore[attr-defined]
    assert "".join(fragment[1] for fragment in status) == "fake-model · context 75%"
    assert [fragment[0] for fragment in status] == ["class:status"]


async def test_the_status_warns_once_the_window_is_nearly_full() -> None:
    config = Config(terminal=TerminalConfig(context_warning=0.5))
    terminal, buffer = _terminal(["hello"], config)
    agent = Agent(
        model=replace(FAKE_MODEL, context_window=20),
        provider=FakeProvider([text_response("Hi")]),
        config=config,
    )
    terminal.attach(agent)

    await terminal.run()

    assert [f[0] for f in terminal._rprompt()] == ["class:status.warning"]  # type: ignore[attr-defined]
    assert "/clear to start over" in buffer.getvalue()


async def test_the_status_is_only_offered_at_the_input_prompt() -> None:
    """The approval question shares the session; it draws no status."""
    terminal, _ = _terminal(["hello", "y"])
    _attach(terminal, FakeProvider([text_response("Hi")]))

    await terminal.run()
    await terminal.ask_approval("bash", {"command": "ls"})

    rprompts = terminal._session.rprompts  # type: ignore[attr-defined]
    assert callable(rprompts[0]), "the input prompt carries one"
    assert rprompts[-1] is None, "the question does not"


async def test_before_any_run_the_status_is_the_model_alone() -> None:
    """`context 0%` would be a measurement that was never taken."""
    terminal, _ = _terminal([])
    _attach(terminal, FakeProvider([]))

    assert "".join(f[1] for f in terminal._rprompt()) == FAKE_MODEL.id  # type: ignore[attr-defined]


# --- input history --------------------------------------------------------


async def test_typed_lines_are_kept_in_the_session_home() -> None:
    config = Config()
    path = lab_harness_home(config.session) / config.session.history_file
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        console = Console(file=io.StringIO(), width=90, force_terminal=False, no_color=True)
        terminal = Terminal(console=console, config=config)
        _attach(terminal, FakeProvider([text_response("Hi")]))
        pipe.send_text("hello\r\x04")

        assert await terminal.run() == 0

    assert "+hello" in path.read_text(encoding="utf-8")


def test_history_stays_in_memory_when_nothing_is_persisted(tmp_path: Path) -> None:
    console = Console(file=io.StringIO(), width=90, force_terminal=False, no_color=True)
    config = Config(session=replace(SessionConfig(), enabled=False, home=str(tmp_path)))

    terminal = Terminal(console=console, config=config)

    assert isinstance(terminal._session.history, InMemoryHistory)  # type: ignore[attr-defined]
    assert list(tmp_path.iterdir()) == [], "nothing written"


async def test_status_shows_session_totals_without_a_request() -> None:
    terminal, buffer = _terminal(["first", "second", "/status"])
    provider = FakeProvider([text_response("one"), text_response("two")])
    _attach(terminal, provider)

    assert await terminal.run() == 0

    assert len(provider.requests) == 2, "/status must not reach the provider"
    status = buffer.getvalue().split("two")[-1]
    assert re.search(r"requests\s+2\b", status)
    assert re.search(r"output\s+10\b", status)


async def test_status_includes_resumed_history() -> None:
    """Restored messages never pass a subscriber, so totals come from the transcript."""
    terminal, buffer = _terminal(["/status"])
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


async def test_effort_changes_the_level_the_next_runs_ask_for() -> None:
    terminal, buffer = _terminal(["/effort low", "first", "/status", "/effort default"])
    provider = FakeProvider([text_response("one")])
    model = replace(FAKE_MODEL, supports_reasoning=True)
    agent = Agent(model=model, provider=provider, config=terminal.config)
    terminal.attach(agent)
    efforts: list[str] = []

    async def record(event: object) -> None:
        if isinstance(event, AgentStart):
            efforts.append(event.model.reasoning_effort)

    agent.subscribe(record)

    await terminal.run()

    assert efforts == ["low"], "the run after it asks for the level"
    assert len(provider.requests) == 1, "/effort must not reach the provider"
    assert re.search(r"effort\s+low\b", buffer.getvalue()), "/status shows it"
    assert agent.state.model == model, "default sends none again"


async def test_effort_says_when_the_model_does_not_reason() -> None:
    terminal, buffer = _terminal(["/effort high"])
    _attach(terminal, FakeProvider([]))

    await terminal.run()

    assert "does not reason" in buffer.getvalue()


# --- commands -------------------------------------------------------------


async def test_an_unknown_command_says_so_instead_of_sending() -> None:
    terminal, buffer = _terminal(["/stauts"])
    provider = FakeProvider([])
    _attach(terminal, provider)

    assert await terminal.run() == 0

    assert "Unknown command /stauts" in buffer.getvalue()
    assert provider.requests == []


async def test_a_path_is_a_message_not_a_command() -> None:
    terminal, _ = _terminal(["/tmp/notes.md what is this?"])
    provider = FakeProvider([text_response("a file")])
    _attach(terminal, provider)

    await terminal.run()

    assert len(provider.requests) == 1


async def test_a_registered_command_is_run_with_its_arguments() -> None:
    terminal, _ = _terminal(["/echo one two"])
    _attach(terminal, FakeProvider([]))
    given: list[str] = []

    async def echo(arguments: str) -> None:
        given.append(arguments)

    terminal.commands.add(Command("echo", "Repeat", echo))
    await terminal.run()

    assert given == ["one two"]


async def test_the_session_opens_with_a_box_naming_the_model_and_the_directory() -> None:
    terminal, buffer = _terminal([])
    _attach(terminal, FakeProvider([]), cwd="proj")

    terminal.welcome()

    output = buffer.getvalue()
    assert f"{FAKE_MODEL.id} via {FAKE_MODEL.provider}" in output
    assert "proj" in output, "the directory the tools run in"
    assert "/ for commands" in output
    assert "╭" in output, "a box, not another line of text"


async def test_help_lists_every_command_with_what_it_does() -> None:
    terminal, buffer = _terminal(["/help"])
    _attach(terminal, FakeProvider([]))

    await terminal.run()

    output = buffer.getvalue()
    assert "/status" in output and "Token, cache and cost totals so far" in output
    assert "/exit (quit)" in output and "Exit the session" in output


async def test_clear_forgets_the_conversation_but_keeps_the_session() -> None:
    terminal, buffer = _terminal(["hello", "/clear", "again"])
    provider = FakeProvider([text_response("one"), text_response("two")])
    agent = _attach(terminal, provider)

    await terminal.run()

    assert len(agent.state.messages) == 2, "only the second run's pair is left"
    assert "Conversation cleared" in buffer.getvalue()


async def test_model_shows_the_model_and_the_ones_on_its_provider() -> None:
    config = Config(
        models={
            FAKE_MODEL.id: model_entry(provider="fake"),
            "other-fake": model_entry(provider="fake"),
            "elsewhere": model_entry(provider="anthropic"),
        }
    )
    terminal, buffer = _terminal(["/model"], config)
    agent = _attach(terminal, FakeProvider([]))

    await terminal.run()

    output = buffer.getvalue()
    assert "model: fake-model via fake" in output
    assert "other-fake" in output, "the ones this session could switch to"
    assert "elsewhere" not in output, "one this session cannot switch to"
    assert agent.state.model == FAKE_MODEL


async def test_model_switches_to_one_this_provider_serves() -> None:
    config = Config(
        models={FAKE_MODEL.id: model_entry(provider="fake"), "other": model_entry(provider="fake")}
    )
    terminal, buffer = _terminal(["/model other"], config)
    agent = _attach(terminal, FakeProvider([]))

    await terminal.run()

    assert agent.state.model.id == "other"
    assert agent.state.model.provider == "fake", "the session's provider, not the catalog's"
    assert "model: other via fake" in buffer.getvalue()


async def test_model_refuses_one_another_provider_serves() -> None:
    config = Config(
        models={FAKE_MODEL.id: model_entry(provider="fake"), "gpt": model_entry(provider="openai")}
    )
    terminal, buffer = _terminal(["/model gpt", "/model nope"], config)
    agent = _attach(terminal, FakeProvider([]))

    await terminal.run()

    assert agent.state.model == FAKE_MODEL, "a refusal changes nothing"
    assert "served by openai" in buffer.getvalue()
    assert "not a configured model" in buffer.getvalue()


async def test_a_failing_command_ends_itself_not_the_session() -> None:
    terminal, buffer = _terminal(["/boom", "after"])
    provider = FakeProvider([text_response("still here")])
    _attach(terminal, provider)

    async def boom(arguments: str) -> None:
        raise RuntimeError("it broke")

    terminal.commands.add(Command("boom", "Fail", boom))
    await terminal.run()

    assert "error: it broke" in buffer.getvalue()
    assert "still here" in buffer.getvalue()


async def test_the_prompt_completes_a_command_as_typed() -> None:
    """The real session, so the menu is wired to the prompt it serves."""
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        buffer = io.StringIO()
        console = Console(file=buffer, width=90, force_terminal=False, no_color=True)
        terminal = Terminal(console=console, config=Config())
        provider = FakeProvider([])
        _attach(terminal, provider)
        pipe.send_text("/st\r\x04")

        assert await terminal.run() == 0

    assert re.search(r"requests\s+0\b", buffer.getvalue()), "/st ran /status"
    assert provider.requests == []


async def test_escape_is_a_newline_except_where_it_closes_the_list() -> None:
    """Through the terminal's own session, so both Escape bindings are in play
    at once: prompt_toolkit holds an Escape that could begin a longer
    sequence, which is why the newline binding stands down while a list is up.
    """
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        console = Console(file=io.StringIO(), width=90, force_terminal=False, no_color=True)
        terminal = Terminal(console=console, config=Config())
        _attach(terminal, FakeProvider([]))
        session = terminal._session  # type: ignore[assignment]
        pipe.send_text("ab\x1b\rcd\r")

        assert await session.prompt_async("› ") == "ab\ncd"  # noqa: RUF001

        with terminal._menu.offered():  # type: ignore[attr-defined]
            # A second Enter, so that a held Escape fails the assertion rather
            # than leaving the prompt waiting for one.
            pipe.send_text("/e\x1b\r\r")

            assert await session.prompt_async("› ") == "/e"  # noqa: RUF001


# --- approval prompt ------------------------------------------------------


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("y", "yes"),
        ("yes", "yes"),
        ("Y", "yes"),
        ("n", "no"),
        ("a", "always"),
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
    assert "Allow Write?" in str(session.prompts[-1])  # type: ignore[attr-defined]


async def test_the_question_shows_the_whole_command() -> None:
    """The call line is cut short; what is being approved must not be."""
    terminal, buffer = _terminal(["n"])
    _attach(terminal, FakeProvider([]))
    command = "make clean && " + "x" * 80 + " && rm -rf build/"

    await terminal.ask_approval("bash", {"command": command})

    assert command in buffer.getvalue().replace("\n", "")


async def test_the_question_is_asked_with_the_live_region_down() -> None:
    """A prompt under a live region is drawn over by its next redraw."""
    terminal, _ = _terminal([])
    _attach(terminal, FakeProvider([]))
    seen: list[bool] = []

    class Watching(ScriptedSession):
        async def prompt_async(self, message: object = "", *, rprompt: object = None) -> str:
            seen.append(terminal.renderer.active)
            return "y"

    terminal._session = Watching([])  # type: ignore[assignment]
    await terminal.renderer(AgentStart(system_prompt="", tools=(), model=FAKE_MODEL, history=0))
    assert terminal.renderer.active

    await terminal.ask_approval("bash", {"command": "ls"})

    assert seen == [False]
    assert terminal.renderer.active, "the region comes back once answered"
    terminal.renderer.close()


async def test_a_denied_tool_does_not_run_through_the_terminal(tmp_path) -> None:
    """End to end: terminal answer -> gate -> loop -> tool never executes."""
    from lab_harness.tools import default_registry

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
    from lab_harness.tools import default_registry

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


# --- thinking toggle ------------------------------------------------------


async def test_ctrl_o_at_the_prompt_toggles_thinking() -> None:
    """Through a real PromptSession, so the binding beats prompt_toolkit's own
    Ctrl-O (emacs operate-and-get-next) and the typed line is left intact."""
    calls: list[str] = []
    with create_pipe_input() as pipe:
        pipe.send_text("ab\x0fcd\r")
        session: PromptSession[str] = PromptSession(
            input=pipe,
            output=DummyOutput(),
            key_bindings=_bindings(lambda: calls.append("t"), _never_listing),
        )
        assert await session.prompt_async("› ") == "abcd"  # noqa: RUF001

    assert calls == ["t"]


async def test_toggling_expands_the_last_runs_thinking() -> None:
    terminal, buffer = _terminal(["think"])
    _attach(terminal, FakeProvider([text_response("answer", thinking="the reasoning")]))
    await terminal.run()
    assert "the reasoning" not in buffer.getvalue()

    terminal._toggle_thinking()

    assert "the reasoning" in buffer.getvalue()
    assert terminal.renderer.thinking_mode == "full"


def test_an_invalid_prompt_style_is_a_config_error() -> None:
    config = Config(terminal=TerminalConfig(theme=ThemeConfig(prompt="rgb(1,2,3)")))

    with pytest.raises(ConfigError, match=r"terminal\.theme\.prompt"):
        Terminal(console=Console(file=io.StringIO()), config=config)


def test_ctrl_o_cannot_discard_output_during_a_run() -> None:
    """Cooked-mode Ctrl-O is DISCARD on macOS and the BSDs: it would silently
    drop the run's output. Disabled for the run, restored after."""
    termios = pytest.importorskip("termios")
    import os
    import pty

    master, slave = pty.openpty()
    try:
        disabled = bytes([os.fpathconf(slave, "PC_VDISABLE")])
        before = termios.tcgetattr(slave)
        with _keys_wait_quietly(slave):
            during = termios.tcgetattr(slave)
        after = termios.tcgetattr(slave)
    finally:
        os.close(master)
        os.close(slave)

    assert before[6][termios.VDISCARD] != disabled
    assert during[6][termios.VDISCARD] == disabled
    assert after == before


def test_keys_typed_during_a_run_wait_unechoed() -> None:
    """Echoed, a key lands inside the live region, and once it wraps the next
    redraw leaves a stale line of the region behind. Unechoed it still waits
    in the input queue, where the next prompt reads it."""
    termios = pytest.importorskip("termios")
    import os
    import pty
    import select

    master, slave = pty.openpty()
    try:
        before = termios.tcgetattr(slave)
        with _keys_wait_quietly(slave):
            os.write(master, b"typed ahead\n")
            echoed = select.select([master], [], [], 0.2)[0]
        after = termios.tcgetattr(slave)
        waiting = os.read(slave, 100)
    finally:
        os.close(master)
        os.close(slave)

    assert not echoed, "nothing may be written to the screen"
    assert waiting == b"typed ahead\n"
    assert after == before


async def test_a_run_keeps_keys_quiet_for_its_whole_length(monkeypatch) -> None:
    termios = pytest.importorskip("termios")
    import os
    import pty

    from pydantic import BaseModel

    from lab_harness.tools.base import ToolContext, ToolResult, ok, tool

    class _P(BaseModel):
        pass

    master, slave = pty.openpty()
    disabled = bytes([os.fpathconf(slave, "PC_VDISABLE")])
    during: list[bytes] = []
    echoing: list[bool] = []

    @tool(name="probe", description="probe")
    async def probe(p: _P, ctx: ToolContext) -> ToolResult:
        attributes = termios.tcgetattr(slave)
        during.append(attributes[6][termios.VDISCARD])
        echoing.append(bool(attributes[3] & termios.ECHO))
        return ok("probed")

    monkeypatch.setattr("lab_harness.cli.terminal._stdin_fd", lambda: slave)
    call = ToolCallBlock(id="c1", name="probe", arguments={})
    terminal, _ = _terminal(["go"])
    _attach(
        terminal,
        FakeProvider([tool_call_response([call]), text_response("done")]),
        tools=(probe,),
    )
    try:
        await terminal.run()
        after = termios.tcgetattr(slave)
    finally:
        os.close(master)
        os.close(slave)

    assert during == [disabled]
    assert echoing == [False]
    assert after[6][termios.VDISCARD] != disabled
    assert after[3] & termios.ECHO


# --- the prompt -----------------------------------------------------------


async def test_the_prompt_glyph_is_styled() -> None:
    terminal, _ = _terminal([])
    _attach(terminal, FakeProvider([]))

    await terminal.run()

    first = terminal._session.prompts[0]  # type: ignore[attr-defined]
    assert first == [("class:glyph", GLYPH)]


# --- wiring ---------------------------------------------------------------


# --- log records ----------------------------------------------------------


async def test_a_log_record_prints_above_the_live_region_and_leaves_it_whole() -> None:
    """Written straight to the terminal, a record lands inside the region: the
    next redraw overwrites it and leaves a stale line of the region behind."""
    from tests.screen import screen

    out = io.StringIO()
    console = Console(file=out, width=60, height=20, force_terminal=True, no_color=True)
    renderer = EventRenderer(console, TerminalConfig(), clock=lambda: 0.0)
    configure_logging("debug", stream=LogStream(console, "dim"))
    message = AssistantMessage(content=(), model=FAKE_MODEL.id, provider="fake")

    await renderer(AgentStart(system_prompt="", tools=(), model=FAKE_MODEL, history=0))
    try:
        logging.getLogger("lab_harness.sample").debug("a record")
        await renderer(MessageStart(message=message))  # anything that redraws
        shown = [line for line in screen(out.getvalue()) if line]
    finally:
        renderer.close()

    assert len(shown) == 2, shown
    assert shown[0] == "DEBUG lab_harness.sample: a record"
    assert shown[1].endswith("Working… (0s · Ctrl-C to cancel)")


def test_a_log_record_is_printed_in_the_style_given() -> None:
    out = io.StringIO()
    console = Console(file=out, force_terminal=True, color_system="standard")

    LogStream(console, "dim").write("DEBUG lab_harness.sample: quiet\n")

    assert out.getvalue() == "\x1b[2mDEBUG lab_harness.sample: quiet\x1b[0m\n"
