"""The terminal renderer, against a rich console capturing to a buffer."""

import io

from rich.console import Console

from harness.agent import Agent
from harness.cli.renderers import EventRenderer, _headline
from harness.config import Config, TerminalConfig
from harness.llm.fake import FAKE_MODEL, FakeProvider, text_response, tool_call_response
from harness.llm.types import AssistantMessage, TextBlock, ToolCallBlock
from harness.tools import ToolRegistry
from harness.tools.fs import FILESYSTEM_TOOLS


def _console(buffer: io.StringIO) -> Console:
    # width fixed so wrapping cannot make assertions flaky; no color codes.
    return Console(file=buffer, width=100, force_terminal=False, no_color=True, highlight=False)


def _renderer(buffer: io.StringIO, **kwargs: object) -> EventRenderer:
    return EventRenderer(_console(buffer), TerminalConfig(**kwargs))  # type: ignore[arg-type]


async def _run(buffer: io.StringIO, provider: FakeProvider, tools=(), **kwargs: object) -> Agent:
    agent = Agent(model=FAKE_MODEL, provider=provider, tools=tools, config=Config(), **kwargs)  # type: ignore[arg-type]
    agent.subscribe(_renderer(buffer))
    await agent.prompt("go")
    return agent


# --- assistant text -------------------------------------------------------


async def test_assistant_text_is_rendered() -> None:
    buffer = io.StringIO()
    await _run(buffer, FakeProvider([text_response("Hello there")]))

    assert "Hello there" in buffer.getvalue()


async def test_streaming_deltas_appear_as_they_arrive() -> None:
    """Text must be visible during the stream, not only at the end."""
    buffer = io.StringIO()
    renderer = _renderer(buffer, markdown=False)
    agent = Agent(model=FAKE_MODEL, provider=FakeProvider([text_response("abcdef", chunk_size=2)]))
    agent.subscribe(renderer)

    await agent.prompt("go")

    # Rendered once from deltas; not duplicated by the final message.
    assert buffer.getvalue().count("abcdef") == 1


async def test_markdown_mode_renders_the_final_message() -> None:
    buffer = io.StringIO()
    renderer = _renderer(buffer, markdown=True)
    agent = Agent(model=FAKE_MODEL, provider=FakeProvider([text_response("# Title")]))
    agent.subscribe(renderer)

    await agent.prompt("go")

    assert "Title" in buffer.getvalue()


async def test_an_error_message_is_shown() -> None:
    from harness.llm.fake import error_response

    buffer = io.StringIO()
    await _run(buffer, FakeProvider([error_response("rate limited")]))

    assert "rate limited" in buffer.getvalue()


async def test_an_empty_assistant_message_renders_nothing_extra() -> None:
    buffer = io.StringIO()
    renderer = _renderer(buffer)
    message = AssistantMessage(content=(), model="m", provider="p")

    from harness.agent.types import MessageEnd

    await renderer(MessageEnd(message=message))

    assert buffer.getvalue().strip() == ""


# --- tool calls -----------------------------------------------------------


async def test_a_tool_call_shows_the_name_and_its_headline_argument(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("hello\n")
    call = ToolCallBlock(id="c1", name="read", arguments={"path": "a.txt"})
    buffer = io.StringIO()

    await _run(
        buffer,
        FakeProvider([tool_call_response([call]), text_response("done")]),
        tools=tuple(ToolRegistry(FILESYSTEM_TOOLS)),
        cwd=str(tmp_path),
    )

    output = buffer.getvalue()
    assert "read" in output
    assert "a.txt" in output


async def test_tool_output_is_shown_under_the_call(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("the contents\n")
    call = ToolCallBlock(id="c1", name="read", arguments={"path": "a.txt"})
    buffer = io.StringIO()

    await _run(
        buffer,
        FakeProvider([tool_call_response([call]), text_response("done")]),
        tools=tuple(ToolRegistry(FILESYSTEM_TOOLS)),
        cwd=str(tmp_path),
    )

    assert "the contents" in buffer.getvalue()
    assert "└" in buffer.getvalue()


async def test_long_tool_output_is_collapsed(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("\n".join(f"line{i}" for i in range(50)))
    call = ToolCallBlock(id="c1", name="read", arguments={"path": "a.txt"})
    buffer = io.StringIO()
    agent = Agent(
        model=FAKE_MODEL,
        provider=FakeProvider([tool_call_response([call]), text_response("done")]),
        tools=tuple(ToolRegistry(FILESYSTEM_TOOLS)),
        cwd=str(tmp_path),
    )
    agent.subscribe(_renderer(buffer, max_result_lines=3))

    await agent.prompt("go")

    output = buffer.getvalue()
    assert "more line(s)" in output
    assert "line49" not in output


async def test_a_failed_tool_is_rendered_distinctly(tmp_path) -> None:
    call = ToolCallBlock(id="c1", name="read", arguments={"path": "absent.txt"})
    buffer = io.StringIO()

    await _run(
        buffer,
        FakeProvider([tool_call_response([call]), text_response("done")]),
        tools=tuple(ToolRegistry(FILESYSTEM_TOOLS)),
        cwd=str(tmp_path),
    )

    assert "not found" in buffer.getvalue().lower()


# --- headline selection ---------------------------------------------------


def test_headline_picks_the_meaningful_argument_per_tool() -> None:
    assert _headline("bash", {"command": "ls -la", "timeout_seconds": 5}) == "ls -la"
    assert _headline("read", {"path": "a.py", "offset": 3}) == "a.py"
    assert _headline("grep", {"pattern": "def foo", "path": "."}) == "def foo"


def test_headline_falls_back_to_the_first_argument() -> None:
    assert _headline("unknown_tool", {"first": "value", "second": "other"}) == "value"


def test_headline_handles_no_arguments() -> None:
    assert _headline("ls", {}) == ""


def test_headline_is_single_line_and_bounded() -> None:
    """A multi-line heredoc must not blow up the one-line call notice."""
    result = _headline("bash", {"command": "line one\nline two\n" + "x" * 500})

    assert "\n" not in result
    assert len(result) <= 61


# --- isolation ------------------------------------------------------------


async def test_the_renderer_holds_no_reference_to_the_agent() -> None:
    """cli/ must stay deletable; the renderer is a pure event consumer."""
    buffer = io.StringIO()
    renderer = _renderer(buffer)

    assert not any("agent" in name.lower() for name in vars(renderer))


# --- no duplication -------------------------------------------------------


async def test_streamed_text_is_not_printed_twice() -> None:
    """Regression: deltas printed directly plus a markdown re-render showed
    the whole answer twice. Scrollback output cannot be taken back, so the
    streaming region must be the same region that ends up final."""
    buffer = io.StringIO()
    renderer = EventRenderer(
        Console(file=buffer, width=100, force_terminal=True, no_color=True),
        TerminalConfig(markdown=True),
    )
    agent = Agent(
        model=FAKE_MODEL,
        provider=FakeProvider([text_response("The unique answer", chunk_size=4)]),
    )
    agent.subscribe(renderer)

    await agent.prompt("go")

    assert buffer.getvalue().count("The unique answer") == 1


async def test_plain_text_mode_also_renders_once() -> None:
    buffer = io.StringIO()
    renderer = EventRenderer(
        Console(file=buffer, width=100, force_terminal=True, no_color=True),
        TerminalConfig(markdown=False),
    )
    agent = Agent(
        model=FAKE_MODEL,
        provider=FakeProvider([text_response("Single rendering", chunk_size=3)]),
    )
    agent.subscribe(renderer)

    await agent.prompt("go")

    assert buffer.getvalue().count("Single rendering") == 1


async def test_a_non_streaming_message_still_renders_once() -> None:
    """A provider that sends no deltas must still show its text."""
    buffer = io.StringIO()
    renderer = EventRenderer(
        Console(file=buffer, width=100, force_terminal=False, no_color=True),
        TerminalConfig(markdown=False),
    )
    from harness.agent.types import MessageEnd

    message = AssistantMessage(content=(TextBlock(text="No deltas here"),), model="m", provider="p")
    await renderer(MessageEnd(message=message))

    assert buffer.getvalue().count("No deltas here") == 1


async def test_tool_notices_are_not_swallowed_by_a_live_region(tmp_path) -> None:
    """A tool call mid-stream must still appear; Live would overwrite it."""
    (tmp_path / "a.txt").write_text("file body\n")
    call = ToolCallBlock(id="c1", name="read", arguments={"path": "a.txt"})
    buffer = io.StringIO()
    renderer = EventRenderer(
        Console(file=buffer, width=100, force_terminal=True, no_color=True),
        TerminalConfig(),
    )
    agent = Agent(
        model=FAKE_MODEL,
        provider=FakeProvider(
            [
                tool_call_response([call], preamble="Looking now"),
                text_response("Finished"),
            ]
        ),
        tools=tuple(ToolRegistry(FILESYSTEM_TOOLS)),
        cwd=str(tmp_path),
    )
    agent.subscribe(renderer)

    await agent.prompt("go")

    output = buffer.getvalue()
    assert "read" in output
    assert "file body" in output
    assert output.count("Finished") == 1
