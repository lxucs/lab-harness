"""The terminal renderer, against a rich console capturing to a buffer.

Two kinds of console. A non-terminal one receives only what is *printed* —
exactly what ends up in scrollback — because rich draws no live region on a
file. A terminal one also receives every frame of the live region; those
tests read it back through `screen`, which replays the cursor movement the
way a terminal would, so an erased frame does not count as output.
"""

import io
import re

import pytest
from rich.console import Console

from lab_harness.agent import Agent
from lab_harness.agent.types import (
    AgentEnd,
    AgentStart,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolExecutionUpdate,
)
from lab_harness.cli.renderers import EventRenderer, headline
from lab_harness.config import Config, ConfigError, TerminalConfig, ThemeConfig
from lab_harness.llm.fake import FAKE_MODEL, FakeProvider, text_response, tool_call_response
from lab_harness.llm.types import AssistantMessage, TextBlock, ThinkingBlock, ToolCallBlock, text
from lab_harness.tools import ToolRegistry
from lab_harness.tools.fs import FILESYSTEM_TOOLS
from tests.screen import visible


class Clock:
    """Time the test moves by hand, so durations and redraws are exact."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def _scrollback(buffer: io.StringIO) -> Console:
    # width fixed so wrapping cannot make assertions flaky; no color codes.
    return Console(file=buffer, width=100, force_terminal=False, no_color=True, highlight=False)


def _terminal(buffer: io.StringIO, *, color: bool = False) -> Console:
    return Console(
        file=buffer,
        width=100,
        height=40,
        force_terminal=True,
        no_color=not color,
        color_system="standard" if color else None,
        highlight=False,
    )


def _renderer(
    buffer: io.StringIO, *, terminal: bool = False, clock: Clock | None = None, **kwargs: object
) -> EventRenderer:
    console = _terminal(buffer) if terminal else _scrollback(buffer)
    return EventRenderer(console, TerminalConfig(**kwargs), clock=clock or Clock())  # type: ignore[arg-type]


async def _run(
    buffer: io.StringIO, provider: FakeProvider, tools=(), *, terminal: bool = False, **kwargs
) -> EventRenderer:
    renderer = _renderer(buffer, terminal=terminal, **kwargs.pop("renderer", {}))
    agent = Agent(model=FAKE_MODEL, provider=provider, tools=tools, config=Config(), **kwargs)
    agent.subscribe(renderer)
    await agent.prompt("go")
    return renderer


def _assistant(*blocks: object) -> AssistantMessage:
    return AssistantMessage(content=blocks, model="m", provider="p")  # type: ignore[arg-type]


def _started() -> MessageStart:
    return MessageStart(message=_assistant())


def _thought(delta: str, index: int = 0) -> MessageUpdate:
    return MessageUpdate(block_index=index, delta=delta, kind="thinking")


def _said(delta: str, index: int = 1) -> MessageUpdate:
    return MessageUpdate(block_index=index, delta=delta, kind="text")


def _read_tools(tmp_path):
    return tuple(ToolRegistry(FILESYSTEM_TOOLS)), str(tmp_path)


# --- assistant text -------------------------------------------------------


async def test_assistant_text_is_rendered_behind_a_bullet() -> None:
    buffer = io.StringIO()
    await _run(buffer, FakeProvider([text_response("Hello there")]))

    assert "⏺ Hello there" in buffer.getvalue()


async def test_streaming_text_is_visible_before_the_message_ends() -> None:
    buffer = io.StringIO()
    clock = Clock()
    renderer = _renderer(buffer, terminal=True, clock=clock, markdown=False)

    await renderer(_started())
    await renderer(_said("half an "))
    clock.now += 1
    await renderer(_said("answer"))

    assert "half an answer" in visible(buffer.getvalue())
    renderer.close()


async def test_markdown_mode_renders_the_final_message() -> None:
    buffer = io.StringIO()
    await _run(buffer, FakeProvider([text_response("# Title")]), renderer={"markdown": True})

    assert "Title" in buffer.getvalue()
    assert "#" not in buffer.getvalue()


async def test_an_error_message_is_shown() -> None:
    from lab_harness.llm.fake import error_response

    buffer = io.StringIO()
    await _run(buffer, FakeProvider([error_response("rate limited")]))

    assert "rate limited" in buffer.getvalue()


async def test_an_empty_assistant_message_renders_nothing() -> None:
    buffer = io.StringIO()
    renderer = _renderer(buffer)

    await renderer(MessageEnd(message=_assistant()))

    assert buffer.getvalue().strip() == ""


# --- nothing shown twice --------------------------------------------------
#
# Regression: deltas printed directly plus a markdown re-render showed the
# whole answer twice. Text written to scrollback cannot be taken back, so the
# streaming copy must be one that gets erased.


@pytest.mark.parametrize("markdown", [True, False])
async def test_streamed_text_is_left_on_screen_once(markdown: bool) -> None:
    buffer = io.StringIO()
    await _run(
        buffer,
        FakeProvider([text_response("The unique answer", chunk_size=4)]),
        terminal=True,
        renderer={"markdown": markdown},
    )

    assert visible(buffer.getvalue()).count("The unique answer") == 1


async def test_a_message_that_never_streamed_still_renders_once() -> None:
    """A provider that sends no deltas must still show its text."""
    buffer = io.StringIO()
    renderer = _renderer(buffer, markdown=False)

    await renderer(MessageEnd(message=_assistant(TextBlock(text="No deltas here"))))

    assert buffer.getvalue().count("No deltas here") == 1


async def test_nothing_is_left_behind_by_the_live_region(tmp_path) -> None:
    """The spinner and running-tool lines are in-flight state; once the run
    ends, the screen holds the conversation and nothing else."""
    (tmp_path / "a.txt").write_text("file body\n")
    call = ToolCallBlock(id="c1", name="read", arguments={"path": "a.txt"})
    tools, cwd = _read_tools(tmp_path)
    buffer = io.StringIO()

    await _run(
        buffer,
        FakeProvider([tool_call_response([call], preamble="Looking now"), text_response("Done")]),
        tools=tools,
        cwd=cwd,
        terminal=True,
    )

    shown = visible(buffer.getvalue())
    assert shown.count("Looking now") == 1
    assert shown.count("read(a.txt)") == 1
    assert "file body" in shown
    assert shown.count("Done") == 1
    assert "Ctrl-C to cancel" not in shown


# --- thinking -------------------------------------------------------------


async def test_thinking_is_visible_while_it_streams() -> None:
    buffer = io.StringIO()
    renderer = _renderer(buffer, terminal=True)

    await renderer(_started())
    await renderer(_thought("weighing the options"))

    shown = visible(buffer.getvalue())
    assert "Thinking" in shown
    assert "weighing the options" in shown
    renderer.close()


async def test_collapsed_thinking_leaves_one_line_in_scrollback() -> None:
    buffer = io.StringIO()
    await _run(buffer, FakeProvider([text_response("the answer", thinking="deep thoughts")]))

    output = buffer.getvalue()
    assert "deep thoughts" not in output
    assert "✻ Thought" in output
    assert "Ctrl-O to expand" in output
    assert output.index("Thought") < output.index("the answer")


async def test_the_collapsed_line_says_how_long_it_thought() -> None:
    buffer = io.StringIO()
    clock = Clock()
    renderer = _renderer(buffer, clock=clock)

    await renderer(_started())
    await renderer(_thought("hmm"))
    clock.now += 12.4
    await renderer(_said("answer"))
    await renderer(
        MessageEnd(message=_assistant(ThinkingBlock(thinking="hmm"), TextBlock(text="answer")))
    )

    assert "Thought for 12s" in buffer.getvalue()


async def test_full_thinking_stays_in_scrollback_ahead_of_the_answer() -> None:
    buffer = io.StringIO()
    await _run(
        buffer,
        FakeProvider([text_response("the answer", thinking="deep thoughts")]),
        terminal=True,
        renderer={"thinking": "full"},
    )

    shown = visible(buffer.getvalue())
    assert shown.count("deep thoughts") == 1
    assert shown.index("deep thoughts") < shown.index("the answer")


async def test_hidden_thinking_leaves_nothing() -> None:
    buffer = io.StringIO()
    await _run(
        buffer,
        FakeProvider([text_response("the answer", thinking="deep thoughts")]),
        renderer={"thinking": "hidden"},
    )

    output = buffer.getvalue()
    assert "deep thoughts" not in output
    assert "Thought" not in output
    assert "the answer" in output


async def test_thinking_that_never_streamed_still_settles() -> None:
    """Responses can deliver a summary only on the finished item."""
    buffer = io.StringIO()
    renderer = _renderer(buffer, thinking="full")

    await renderer(MessageEnd(message=_assistant(ThinkingBlock(thinking="arrived whole"))))

    assert "arrived whole" in buffer.getvalue()


async def test_empty_thinking_renders_nothing() -> None:
    """Claude with display omitted: the block exists, with no text to show."""
    buffer = io.StringIO()
    renderer = _renderer(buffer)

    await renderer(
        MessageEnd(
            message=_assistant(ThinkingBlock(thinking="", signature="s"), TextBlock(text="hi"))
        )
    )

    assert "Thought" not in buffer.getvalue()


async def test_the_preview_shows_only_the_latest_lines() -> None:
    buffer = io.StringIO()
    renderer = _renderer(buffer, terminal=True, thinking_preview_lines=2)

    await renderer(_started())
    await renderer(_thought("\n".join(f"step{i}" for i in range(10))))

    shown = visible(buffer.getvalue())
    assert "step8" in shown
    assert "step9" in shown
    assert "step0" not in shown
    renderer.close()


async def test_expand_prints_what_the_latest_run_collapsed() -> None:
    buffer = io.StringIO()
    renderer = await _run(
        buffer, FakeProvider([text_response("the answer", thinking="deep thoughts")])
    )

    assert renderer.expand_thinking() is True
    assert "deep thoughts" in buffer.getvalue()
    assert renderer.expand_thinking() is False, "a thought prints once"
    assert buffer.getvalue().count("deep thoughts") == 1

    await renderer(AgentStart())
    renderer.close()


async def test_a_new_run_forgets_what_the_last_one_collapsed() -> None:
    buffer = io.StringIO()
    renderer = await _run(
        buffer, FakeProvider([text_response("the answer", thinking="deep thoughts")])
    )

    await renderer(AgentStart())

    assert renderer.expand_thinking() is False
    renderer.close()


@pytest.mark.parametrize(
    ("configured", "toggled"),
    [("collapse", "full"), ("hidden", "full"), ("full", "collapse")],
)
async def test_toggling_switches_between_full_and_the_other_mode(
    configured: str, toggled: str
) -> None:
    renderer = _renderer(io.StringIO(), thinking=configured)

    renderer.toggle_thinking()
    assert renderer.thinking_mode == toggled
    renderer.toggle_thinking()
    assert renderer.thinking_mode == configured


async def test_toggling_to_full_expands_the_latest_run() -> None:
    buffer = io.StringIO()
    renderer = await _run(
        buffer, FakeProvider([text_response("the answer", thinking="deep thoughts")])
    )

    renderer.toggle_thinking()

    assert "deep thoughts" in buffer.getvalue()


# --- tool calls -----------------------------------------------------------


async def test_a_tool_call_shows_its_name_headline_and_result(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("the contents\n")
    call = ToolCallBlock(id="c1", name="read", arguments={"path": "a.txt"})
    tools, cwd = _read_tools(tmp_path)
    buffer = io.StringIO()

    await _run(
        buffer,
        FakeProvider([tool_call_response([call]), text_response("done")]),
        tools=tools,
        cwd=cwd,
    )

    output = buffer.getvalue()
    assert "⏺ read(a.txt)" in output
    assert "⎿" in output
    assert "the contents" in output
    assert output.index("read(a.txt)") < output.index("the contents")


async def test_long_tool_output_is_collapsed(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("\n".join(f"line{i}" for i in range(50)))
    call = ToolCallBlock(id="c1", name="read", arguments={"path": "a.txt"})
    tools, cwd = _read_tools(tmp_path)
    buffer = io.StringIO()

    await _run(
        buffer,
        FakeProvider([tool_call_response([call]), text_response("done")]),
        tools=tools,
        cwd=cwd,
        renderer={"max_result_lines": 3},
    )

    output = buffer.getvalue()
    assert "more line(s)" in output
    assert "line49" not in output


async def test_a_failed_tool_shows_its_error(tmp_path) -> None:
    call = ToolCallBlock(id="c1", name="read", arguments={"path": "absent.txt"})
    tools, cwd = _read_tools(tmp_path)
    buffer = io.StringIO()

    await _run(
        buffer,
        FakeProvider([tool_call_response([call]), text_response("done")]),
        tools=tools,
        cwd=cwd,
    )

    assert "not found" in buffer.getvalue().lower()


def _end(is_error: bool) -> ToolExecutionEnd:
    return ToolExecutionEnd(
        tool_call_id="c1", tool_name="bash", content=text("out"), is_error=is_error, duration_ms=1
    )


@pytest.mark.parametrize(("is_error", "code"), [(False, "\x1b[32m"), (True, "\x1b[31m")])
async def test_the_bullet_is_colored_by_outcome(is_error: bool, code: str) -> None:
    buffer = io.StringIO()
    renderer = EventRenderer(_terminal(buffer, color=True), TerminalConfig(), clock=Clock())

    await renderer(_end(is_error))

    assert f"{code}⏺" in buffer.getvalue()


async def test_a_running_tool_is_shown_until_it_ends() -> None:
    start = ToolExecutionStart(tool_call_id="c1", tool_name="bash", arguments={"command": "make"})

    live = io.StringIO()
    renderer = _renderer(live, terminal=True)
    await renderer(start)
    assert "bash(make)" in visible(live.getvalue())
    renderer.close()

    printed = io.StringIO()
    renderer = _renderer(printed)
    await renderer(start)
    assert "make" not in printed.getvalue(), "nothing settles before the result does"
    await renderer(_end(False))
    assert "bash(make)" in printed.getvalue()


async def test_a_running_tools_latest_output_is_shown() -> None:
    buffer = io.StringIO()
    renderer = _renderer(buffer, terminal=True)

    await renderer(ToolExecutionStart(tool_call_id="c1", tool_name="bash", arguments={}))
    await renderer(
        ToolExecutionUpdate(tool_call_id="c1", tool_name="bash", content=text("first\nsecond\n"))
    )

    shown = visible(buffer.getvalue())
    assert "second" in shown
    assert "first" not in shown
    renderer.close()


# --- diffs ----------------------------------------------------------------


def _diff_end(diff: str, *, content: str = "Replaced 1 occurrence(s) in a.py") -> ToolExecutionEnd:
    return ToolExecutionEnd(
        tool_call_id="c1",
        tool_name="edit",
        content=text(content),
        is_error=False,
        duration_ms=1,
        details={"diff": diff},
    )


def _rows(output: str) -> list[str]:
    return [line.rstrip() for line in output.splitlines()]


async def test_an_edit_shows_its_diff_with_line_numbers(tmp_path) -> None:
    (tmp_path / "a.py").write_text("".join(f"line{i}\n" for i in range(1, 11)))
    call = ToolCallBlock(
        id="c1", name="edit", arguments={"path": "a.py", "old": "line5\n", "new": "five\nsix\n"}
    )
    tools, cwd = _read_tools(tmp_path)
    buffer = io.StringIO()

    await _run(
        buffer,
        FakeProvider([tool_call_response([call]), text_response("done")]),
        tools=tools,
        cwd=cwd,
    )

    rows = _rows(buffer.getvalue())
    start = rows.index("⏺ edit(a.py)")
    assert rows[start + 1] == "  ⎿  Replaced 1 occurrence(s) in a.py  +2 -1"
    assert rows[start + 2 : start + 11] == [
        "    2   line2",
        "    3   line3",
        "    4   line4",
        "    5 - line5",
        "    5 + five",
        "    6 + six",
        "    7   line6",
        "    8   line7",
        "    9   line8",
    ]


@pytest.mark.parametrize(("kind", "background"), [("+", "48;2;34;92;43"), ("-", "48;2;122;41;54")])
async def test_changed_lines_are_whole_colored_bars(kind: str, background: str) -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, width=60, force_terminal=True, color_system="truecolor")
    renderer = EventRenderer(console, TerminalConfig(), clock=Clock())

    await renderer(_diff_end(f"@@ -1 +1 @@\n{kind}changed"))

    row = next(line for line in buffer.getvalue().splitlines() if "changed" in line)
    assert background in row
    # The bar runs to the edge of the terminal, not the end of the text — so
    # measured with only the color codes removed, trailing fill included.
    assert len(re.sub(r"\x1b\[[0-9;]*m", "", row)) == 60


async def test_the_diff_colors_come_from_the_theme() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, width=60, force_terminal=True, color_system="truecolor")
    config = TerminalConfig(theme=ThemeConfig(diff_added="on #010203"))
    renderer = EventRenderer(console, config, clock=Clock())

    await renderer(_diff_end("@@ -1 +1 @@\n+changed"))

    assert "48;2;1;2;3" in buffer.getvalue()


async def test_separate_hunks_are_marked_apart() -> None:
    buffer = io.StringIO()

    await _renderer(buffer)(_diff_end("@@ -2 +2 @@\n-a\n+b\n@@ -40 +40 @@\n-c\n+d"))

    rows = _rows(buffer.getvalue())
    # Numbers right-aligned in a column as wide as the largest.
    assert rows[-5:] == ["     2 - a", "     2 + b", "     ⋮", "    40 - c", "    40 + d"]


async def test_a_long_diff_is_collapsed() -> None:
    buffer = io.StringIO()
    diff = "@@ -0,0 +1,50 @@\n" + "\n".join(f"+line{i}" for i in range(50))

    await _renderer(buffer, max_diff_lines=5)(_diff_end(diff))

    output = buffer.getvalue()
    assert "line4" in output
    assert "line5" not in output
    assert "… 45 more line(s)" in output


async def test_the_file_header_is_not_a_change_but_a_changed_dash_line_is() -> None:
    """`--- a/x` opens the diff; `---x` inside a hunk removes a line `--x`."""
    buffer = io.StringIO()

    await _renderer(buffer)(_diff_end("--- a/x\n+++ b/x\n@@ -1 +1 @@\n---x\n+++y"))

    rows = _rows(buffer.getvalue())
    assert rows[-2:] == ["    1 - --x", "    1 + ++y"]
    assert "a/x" not in buffer.getvalue()


async def test_an_empty_diff_shows_only_the_result() -> None:
    buffer = io.StringIO()

    await _renderer(buffer)(_diff_end("", content="Updated a.py (1 lines)"))

    assert _rows(buffer.getvalue())[-1] == "  ⎿  Updated a.py (1 lines)"


async def test_a_failed_call_shows_its_error_not_a_diff() -> None:
    buffer = io.StringIO()
    end = ToolExecutionEnd(
        tool_call_id="c1",
        tool_name="edit",
        content=text("hook failed"),
        is_error=True,
        duration_ms=1,
        details={"diff": "@@ -1 +1 @@\n-a\n+b"},
    )

    await _renderer(buffer)(end)

    assert "hook failed" in buffer.getvalue()
    assert "- a" not in buffer.getvalue()


async def test_approving_an_edit_shows_the_change() -> None:
    """Approving an edit by its path alone is approving something unseen."""
    buffer = io.StringIO()
    renderer = _renderer(buffer)

    renderer.show_call("edit", {"path": "a.py", "old": "keep\nx = 1", "new": "keep\nx = 2"})

    assert _rows(buffer.getvalue())[-3:] == ["      keep", "    - x = 1", "    + x = 2"]


async def test_approving_a_write_shows_the_content() -> None:
    buffer = io.StringIO()
    renderer = _renderer(buffer, max_diff_lines=2)

    renderer.show_call("write", {"path": "n.py", "content": "one\ntwo\nthree\n"})

    rows = _rows(buffer.getvalue())
    assert rows[-3:] == ["    1 one", "    2 two", "     … 1 more line(s)"]


# --- the run --------------------------------------------------------------


async def test_the_spinner_runs_for_the_length_of_a_run() -> None:
    buffer = io.StringIO()
    clock = Clock()
    renderer = _renderer(buffer, terminal=True, clock=clock)

    await renderer(AgentStart())
    clock.now += 3
    await renderer(_started())
    shown = visible(buffer.getvalue())
    assert "Working" in shown
    assert "3s · Ctrl-C to cancel" in shown
    assert renderer.active

    await renderer(AgentEnd(messages=()))
    assert "Ctrl-C to cancel" not in visible(buffer.getvalue())
    assert not renderer.active


async def test_suspending_takes_the_region_down_and_brings_it_back() -> None:
    """A prompt drawn under a live region would be redrawn over."""
    buffer = io.StringIO()
    renderer = _renderer(buffer, terminal=True)
    await renderer(AgentStart())

    with renderer.suspended():
        assert not renderer.active
        assert "Ctrl-C to cancel" not in visible(buffer.getvalue())
        await renderer(_said("arrives while suspended"))
        assert not renderer.active, "an event must not bring the region back early"

    assert renderer.active
    renderer.close()


async def test_closing_mid_stream_keeps_what_was_shown() -> None:
    """A crashed run must not wipe the answer the user was watching."""
    buffer = io.StringIO()
    renderer = _renderer(buffer, markdown=False)

    await renderer(_started())
    await renderer(_said("partial answer"))
    renderer.close()

    assert "partial answer" in buffer.getvalue()
    assert not renderer.active


# --- configuration --------------------------------------------------------


async def test_styles_come_from_the_theme() -> None:
    buffer = io.StringIO()
    config = TerminalConfig(theme=ThemeConfig(success="magenta"))
    renderer = EventRenderer(_terminal(buffer, color=True), config, clock=Clock())

    await renderer(_end(False))

    assert "\x1b[35m⏺" in buffer.getvalue()


def test_an_unknown_thinking_mode_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match=r"terminal\.thinking"):
        EventRenderer(_scrollback(io.StringIO()), TerminalConfig(thinking="sometimes"))


def test_an_invalid_style_is_a_config_error() -> None:
    config = TerminalConfig(theme=ThemeConfig(error="nonsense-colour"))

    with pytest.raises(ConfigError, match=r"terminal\.theme\.error"):
        EventRenderer(_scrollback(io.StringIO()), config)


# --- headline selection ---------------------------------------------------


def test_headline_picks_the_meaningful_argument_per_tool() -> None:
    assert headline("bash", {"command": "ls -la", "timeout_seconds": 5}) == "ls -la"
    assert headline("read", {"path": "a.py", "offset": 3}) == "a.py"
    assert headline("grep", {"pattern": "def foo", "path": "."}) == "def foo"


def test_headline_falls_back_to_the_first_argument() -> None:
    assert headline("unknown_tool", {"first": "value", "second": "other"}) == "value"


def test_headline_handles_no_arguments() -> None:
    assert headline("ls", {}) == ""


def test_headline_is_single_line_and_bounded() -> None:
    """A multi-line heredoc must not blow up the one-line call notice."""
    result = headline("bash", {"command": "line one\nline two\n" + "x" * 500})

    assert "\n" not in result
    assert len(result) <= 61


# --- isolation ------------------------------------------------------------


def test_the_renderer_holds_no_reference_to_the_agent() -> None:
    """cli/ must stay deletable; the renderer is a pure event consumer."""
    renderer = _renderer(io.StringIO())

    assert not any("agent" in name.lower() for name in vars(renderer))
