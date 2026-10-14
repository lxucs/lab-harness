"""Rendering agent events to a rich console.

A pure event subscriber: it reads events and writes to a console, and holds no
reference to the agent. That is what keeps `cli/` deletable without touching the
runtime.

Output is **inline** — written to normal stdout rather than into a full-screen
application — so a session stays in real terminal scrollback and scrolls the way
every other command does.

**Two kinds of output.** What is finished is printed once and stays in
scrollback: a completed message, a tool call with its result. What is still in
flight — the streaming message, running tools, the spinner — is drawn in one
`rich.Live` region below it, redrawn in place and erased when the run ends.
One region, because rich allows only one at a time. Erased, because text
already written to a scrollback terminal cannot be taken back: a region that
stayed behind would leave a second copy of every answer.

**Thinking** streams into the region like text. When its message ends it is
kept in full, collapsed to one line, or dropped, by `terminal.thinking`; and
whatever the latest run collapsed can still be printed by `expand_thinking`.

**Diffs** come from the tools, not from here: a result whose `details` carry
one — `edit` and `write` — is drawn as Claude Code draws an edit. This module
reads no files, which is also why an approval preview has no line numbers.
"""

import asyncio
import difflib
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, fields
from typing import Literal

from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.errors import StyleSyntaxError
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.segment import Segment
from rich.style import Style
from rich.table import Table
from rich.text import Text

from lab_harness.agent.types import (
    AgentEnd,
    AgentEvent,
    AgentStart,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolExecutionUpdate,
)
from lab_harness.config import ConfigError, TerminalConfig
from lab_harness.llm.types import AssistantMessage, ContentBlock, TextBlock, ThinkingBlock

#: Per-tool argument shown in the one-line call notice. A tool not listed here
#: falls back to the first argument it has.
_HEADLINE_ARGUMENT = {
    "read": "path",
    "write": "path",
    "edit": "path",
    "ls": "path",
    "bash": "command",
    "grep": "pattern",
    "glob": "pattern",
}

_MAX_HEADLINE_LENGTH = 60
_REFRESH_PER_SECOND = 12

THINKING_MODES = ("full", "collapse", "hidden")

# Claude Code's glyphs: a bullet per message and call, a hook under a call for
# its result, and the asterisk it thinks under.
_BULLET = "⏺"
_RESULT = "⎿"
_THINKING = "✻"
_SPINNER = ("·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✳", "✢")
#: Keys the terminal binds; named here because this is where they are shown.
_CANCEL_HINT = "Ctrl-C to cancel"
_EXPAND_HINT = "Ctrl-O to expand"
_TOGGLE_HINT = "Ctrl-O to toggle"

#: Under a call's `⎿`, where its output starts.
_INDENT = "     "
#: Between two hunks of a diff, standing in for the unchanged lines skipped.
_GAP = "⋮"
#: A unified diff's hunk header: where the hunk starts, on each side.
_HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")

_MODE_NOTICES = {
    "full": "shown in full",
    "collapse": "collapsed",
    "hidden": "hidden",
}


@dataclass(slots=True)
class _Block:
    """One block of the message being streamed."""

    kind: Literal["text", "thinking"]
    index: int
    text: str


@dataclass(frozen=True, slots=True)
class _DiffLine:
    #: "+", "-", " " for an unchanged line, or `_GAP` between hunks.
    kind: str
    #: Its line in the file: the old file's for a removal, the new one's
    #: otherwise. None where it is not known.
    number: int | None
    text: str


@dataclass(slots=True)
class _RunningTool:
    name: str
    headline: str
    #: The last line of output it has reported, if it streams any.
    progress: str = ""


class EventRenderer:
    """Renders a run to the console as it happens."""

    def __init__(
        self,
        console: Console,
        config: TerminalConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        _check(config)
        self._console = console
        self._config = config
        self._theme = config.theme
        self._clock = clock
        #: What thinking leaves behind from now on. Starts as configured;
        #: `toggle_thinking` moves it.
        self.thinking_mode = config.thinking
        # The message in flight.
        self._blocks: list[_Block] = []
        self._thinking_since: float | None = None
        self._thought_for: float | None = None
        # The run in flight.
        self._run_since: float | None = None
        self._tools: dict[str, _RunningTool] = {}
        self._collapsed: list[str] = []
        # The region showing them.
        self._live: Live | None = None
        self._ticker: asyncio.Task[None] | None = None
        self._refreshed_at = float("-inf")
        self._suspended = False

    async def __call__(self, event: AgentEvent) -> None:
        if isinstance(event, AgentStart):
            self._run_since = self._clock()
            self._collapsed = []
            self._ensure_live()
        elif isinstance(event, AgentEnd):
            self.close()
        elif isinstance(event, MessageStart):
            if isinstance(event.message, AssistantMessage):
                self._start_message()
        elif isinstance(event, MessageUpdate):
            self._stream(event)
            # Throttled: a long answer redrawn on every delta is quadratic.
            self._refresh(throttle=True)
            return
        elif isinstance(event, MessageEnd):
            if isinstance(event.message, AssistantMessage):
                self._finish_message(event.message)
        elif isinstance(event, ToolExecutionStart):
            self._tools[event.tool_call_id] = _RunningTool(
                event.tool_name, headline(event.tool_name, event.arguments)
            )
            self._ensure_live()
        elif isinstance(event, ToolExecutionUpdate):
            running = self._tools.get(event.tool_call_id)
            if running is not None:
                running.progress = _last_line(event.content)
        elif isinstance(event, ToolExecutionEnd):
            self._finish_tool(event)
        self._refresh()

    @property
    def active(self) -> bool:
        """Whether a live region is on screen."""
        return self._live is not None

    def close(self) -> None:
        """Settle anything still streaming and take the region down.

        The terminal calls this after every turn, a crashed one included, so
        that no region outlives its run. Idempotent.
        """
        if self._blocks:
            # Settle what the user was watching rather than erasing it: a run
            # that crashed mid-answer should still show the answer so far.
            self._settle(
                [b.text for b in self._blocks if b.kind == "thinking"],
                "".join(b.text for b in self._blocks if b.kind == "text"),
            )
        self._start_message()
        self._run_since = None
        self._tools.clear()
        self._stop_live()

    @contextmanager
    def suspended(self) -> Iterator[None]:
        """Take the region down while something else draws — a prompt.

        A prompt drawn under a live region is drawn over by its next redraw.
        """
        was_live = self._live is not None
        self._stop_live()
        self._suspended = True
        try:
            yield
        finally:
            self._suspended = False
            if was_live:
                self._ensure_live()

    def expand_thinking(self) -> bool:
        """Print in full what the latest run collapsed or hid. False if nothing.

        Each thought prints once: expanding again finds nothing new, rather
        than a second copy in scrollback.
        """
        expanded, self._collapsed = self._collapsed, []
        for thought in expanded:
            self._print(self._thinking(thought, header=_THINKING + " Thinking"))
        return bool(expanded)

    def toggle_thinking(self) -> str:
        """Switch between full thinking and the configured mode; return the new one.

        Switching to full also prints what the latest run collapsed, so the
        key does something visible straight away rather than only next time.
        """
        if self.thinking_mode == "full":
            configured = self._config.thinking
            self.thinking_mode = configured if configured != "full" else "collapse"
        else:
            self.thinking_mode = "full"
            self.expand_thinking()
        notice = f"{_THINKING} Thinking {_MODE_NOTICES[self.thinking_mode]} · {_TOGGLE_HINT}"
        self._print(Text(notice, style=self._theme.muted))
        return self.thinking_mode

    def show_call(self, tool_name: str, arguments: dict[str, object]) -> None:
        """Print a call that is waiting on approval: its argument in full and,
        for an edit or a write, the change it will make.

        Its running line is in the region, which the approval prompt takes
        down, so without this the question would appear with nothing to say
        what it is about. In full, because approving a command cut off at
        sixty characters — or an edit known only by its path — is approving
        something unseen.
        """
        short = headline(tool_name, arguments)
        full = _headline_value(tool_name, arguments)
        self._print(self._call_line(tool_name, short, self._theme.warning))
        if full != short:
            # Soft-wrapped: the terminal folds it at the edge, and rich adds
            # no breaks or indentation that would change how it reads.
            self._console.print(Text(full, style=self._theme.muted), soft_wrap=True)
        for row in self._preview(tool_name, arguments):
            self._console.print(row)

    def _preview(self, tool_name: str, arguments: dict[str, object]) -> list[RenderableType]:
        """What an edit or write is about to do, from its arguments alone.

        No line numbers for an edit: the renderer reads no files, so where the
        text sits is unknown until the tool runs. Shown whole, since an edit
        is what the model wrote and is short; a write's content is a whole
        file, so it collapses like any diff.
        """
        if tool_name == "edit":
            old, new = arguments.get("old"), arguments.get("new")
            if not isinstance(old, str) or not isinstance(new, str):
                return []
            before, after = old.splitlines(), new.splitlines()
            # Enough context to show both fragments in full, as one hunk.
            context = max(len(before), len(after))
            diff = difflib.unified_diff(before, after, n=context, lineterm="")
            return self._diff(_parse_diff("\n".join(diff)), limit=None, numbered=False)
        if tool_name == "write":
            content = arguments.get("content")
            if not isinstance(content, str):
                return []
            lines = [_DiffLine(" ", n, line) for n, line in enumerate(content.splitlines(), 1)]
            # Unmarked: whether each line is new depends on a file only the
            # tool reads.
            return self._diff(lines, limit=self._config.max_diff_lines, marked=False)
        return []

    # --- messages ---------------------------------------------------------

    def _start_message(self) -> None:
        self._blocks = []
        self._thinking_since = None
        self._thought_for = None

    def _stream(self, event: MessageUpdate) -> None:
        now = self._clock()
        if event.kind == "thinking":
            if self._thinking_since is None:
                self._thinking_since = now
        elif self._thinking_since is not None and self._thought_for is None:
            # The answer starting is where thinking ended, for "Thought for".
            self._thought_for = now - self._thinking_since

        last = self._blocks[-1] if self._blocks else None
        if last is not None and last.index == event.block_index and last.kind == event.kind:
            last.text += event.delta
        else:
            self._blocks.append(_Block(event.kind, event.block_index, event.delta))
        self._ensure_live()

    def _finish_message(self, message: AssistantMessage) -> None:
        if self._thinking_since is not None and self._thought_for is None:
            self._thought_for = self._clock() - self._thinking_since
        # From the message rather than the deltas: it is the authoritative
        # copy, and the only one when nothing streamed — a non-streaming
        # provider, or a summary that arrived whole.
        self._settle(
            [b.thinking for b in message.content if isinstance(b, ThinkingBlock)],
            "".join(b.text for b in message.content if isinstance(b, TextBlock)),
        )
        self._start_message()
        if message.error_message:
            self._print(Text(message.error_message, style=self._theme.error))

    def _settle(self, thoughts: list[str], body: str) -> None:
        """Print a finished message: its thinking as configured, then its text."""
        thoughts = [t for t in thoughts if t.strip()]
        parts: list[RenderableType] = []
        if thoughts:
            if self.thinking_mode == "full":
                parts += [self._thinking(t, header=self._thought_label()) for t in thoughts]
            else:
                self._collapsed += thoughts
                if self.thinking_mode == "collapse":
                    parts.append(self._collapsed_line())
        if body:
            parts.append(self._assistant(body))
        # The streamed copy leaves the region before the settled one prints,
        # or for one frame the screen would hold both.
        self._blocks = []
        self._refresh()
        for part in parts:
            self._print(part)

    def _assistant(self, body: str) -> RenderableType:
        grid = Table.grid(padding=(0, 1), expand=True)
        grid.add_column(no_wrap=True)
        grid.add_column(ratio=1)
        grid.add_row(_BULLET, Markdown(body) if self._config.markdown else Text(body))
        return grid

    def _thinking(self, thought: str, *, header: str, last: int | None = None) -> RenderableType:
        body: RenderableType = Padding(
            Text(thought.strip(), style=self._theme.thinking), (0, 0, 0, 2)
        )
        if last is not None:
            body = _Tail(body, last)
        return Group(Text(header, style=self._theme.muted), body)

    def _thought_label(self) -> str:
        if self._thought_for is None:
            return f"{_THINKING} Thought"
        return f"{_THINKING} Thought for {_duration(max(self._thought_for, 1))}"

    def _collapsed_line(self) -> Text:
        return Text(f"{self._thought_label()} · {_EXPAND_HINT}", style=self._theme.muted)

    # --- tools ------------------------------------------------------------

    def _finish_tool(self, event: ToolExecutionEnd) -> None:
        running = self._tools.pop(event.tool_call_id, None)
        call = running.headline if running is not None else ""
        bullet = self._theme.error if event.is_error else self._theme.success
        diff = event.details.get("diff")
        if isinstance(diff, str) and diff and not event.is_error:
            rows = self._changed(_text(event.content), _parse_diff(diff))
        else:
            rows = self._output(event)
        self._refresh()
        self._print(Group(self._call_line(event.tool_name, call, bullet), *rows))

    def _output(self, event: ToolExecutionEnd) -> list[RenderableType]:
        lines = _text(event.content).splitlines()
        style = self._theme.error if event.is_error else self._theme.muted
        shown = lines[: self._config.max_result_lines] or ["(no output)"]
        rows: list[RenderableType] = [Text(f"  {_RESULT}  {shown[0]}", style)]
        rows += [Text(f"{_INDENT}{line}", style) for line in shown[1:]]
        if len(lines) > len(shown):
            rows.append(self._more(len(lines) - len(shown)))
        return rows

    def _changed(self, summary: str, lines: list[_DiffLine]) -> list[RenderableType]:
        """A call that changed a file: its result, the counts, then the diff."""
        added = sum(line.kind == "+" for line in lines)
        removed = sum(line.kind == "-" for line in lines)
        head = Text.assemble(
            (f"  {_RESULT}  {summary.splitlines()[0] if summary else ''}", self._theme.muted),
            "  ",
            (f"+{added}", self._theme.success),
            " ",
            (f"-{removed}", self._theme.error),
        )
        return [head, *self._diff(lines, limit=self._config.max_diff_lines)]

    def _diff(
        self,
        lines: list[_DiffLine],
        *,
        limit: int | None,
        numbered: bool = True,
        marked: bool = True,
    ) -> list[RenderableType]:
        """Claude Code's diff: a number, a marker, and the line, changed lines
        as colored bars that run to the edge of the terminal."""
        shown = lines if limit is None else lines[:limit]
        grid = Table.grid(padding=(0, 1), expand=True)
        if numbered:
            grid.add_column(justify="right", no_wrap=True, style=self._theme.muted)
        if marked:
            grid.add_column(no_wrap=True)
        grid.add_column(ratio=1, overflow="fold")
        styles = {"+": self._theme.diff_added, "-": self._theme.diff_removed}
        for line in shown:
            gap = line.kind == _GAP
            cells: list[RenderableType] = []
            if numbered:
                number = _GAP if gap else "" if line.number is None else str(line.number)
                cells.append(number)
            if marked:
                cells.append("" if gap else line.kind)
            cells.append(line.text)
            grid.add_row(*cells, style=styles.get(line.kind))
        rows: list[RenderableType] = [Padding(grid, (0, 0, 0, len(_INDENT) - 1))]
        if len(lines) > len(shown):
            rows.append(self._more(len(lines) - len(shown)))
        return rows

    def _more(self, count: int) -> Text:
        return Text(f"{_INDENT}… {count} more line(s)", style=self._theme.muted)

    def _call_line(self, name: str, call: str, bullet: str, *, lit: bool = True) -> Text:
        muted = self._theme.muted
        return Text.assemble(
            (_BULLET if lit else " ", bullet),
            " ",
            (name, self._theme.tool),
            ("(", muted),
            (call, muted),
            (")", muted),
        )

    # --- the live region --------------------------------------------------

    def _frame(self) -> RenderableType:
        """Everything in flight: the message, running tools, the spinner."""
        parts: list[RenderableType] = []
        collapsed_shown = False
        for position, block in enumerate(self._blocks):
            if block.kind == "text":
                parts.append(self._assistant(block.text))
            elif self.thinking_mode == "hidden":
                continue
            elif position == len(self._blocks) - 1:
                last = self._config.thinking_preview_lines
                parts.append(
                    self._thinking(
                        block.text,
                        header=f"{_THINKING} Thinking…",
                        last=last if self.thinking_mode == "collapse" else None,
                    )
                )
            elif self.thinking_mode == "full":
                parts.append(self._thinking(block.text, header=self._thought_label()))
            elif not collapsed_shown:
                parts.append(self._collapsed_line())
                collapsed_shown = True

        # Blinks, as Claude Code's does, so a hung tool looks different from
        # a finished one at a glance.
        lit = int(self._clock() * 2) % 2 == 0
        for running in self._tools.values():
            line = self._call_line(running.name, running.headline, self._theme.muted, lit=lit)
            parts.append(line)
            if running.progress:
                parts.append(Text(f"  {_RESULT}  {running.progress}", style=self._theme.muted))

        if self._run_since is not None:
            elapsed = self._clock() - self._run_since
            frame = _SPINNER[int(elapsed * _REFRESH_PER_SECOND / 2) % len(_SPINNER)]
            parts.append(
                Text.assemble(
                    (f"{frame} Working…", self._theme.accent),
                    (f" ({_duration(elapsed)} · {_CANCEL_HINT})", self._theme.muted),
                )
            )
        return _Tail(Group(*parts))

    def _ensure_live(self) -> None:
        if self._live is not None or self._suspended:
            return
        self._live = Live(
            console=self._console,
            get_renderable=self._frame,
            # Redrawn by `_refresh` and the ticker below, both on the event
            # loop. rich's own refresh thread would read this object's state
            # while the loop is changing it.
            auto_refresh=False,
            transient=True,
        )
        self._live.start()
        if self._console.is_terminal:
            with suppress(RuntimeError):
                self._ticker = asyncio.get_running_loop().create_task(self._tick())

    def _stop_live(self) -> None:
        if self._ticker is not None:
            self._ticker.cancel()
            self._ticker = None
        if self._live is not None:
            self._live.stop()
            self._live = None

    async def _tick(self) -> None:
        """Keep the spinner and elapsed time moving between events."""
        while True:
            await asyncio.sleep(1 / _REFRESH_PER_SECOND)
            self._refresh()

    def _refresh(self, *, throttle: bool = False) -> None:
        if self._live is None:
            return
        now = self._clock()
        if throttle and now - self._refreshed_at < 1 / _REFRESH_PER_SECOND:
            return
        self._refreshed_at = now
        self._live.refresh()

    def _print(self, renderable: RenderableType) -> None:
        """Into scrollback, above the region if one is up, after a blank line.

        One print, not two: each one redraws the region beneath it.
        """
        self._console.print(Group(Text(), renderable))


class _Tail:
    """The last lines of a renderable — of the whole region, by default.

    rich crops an overflowing live region from the bottom, which would hide
    the newest text and the spinner. It is also what keeps the region shorter
    than the terminal: rows scrolled off the top can no longer be erased, so a
    taller region would leave its frames in scrollback.
    """

    def __init__(self, renderable: RenderableType, lines: int | None = None) -> None:
        self._renderable = renderable
        self._lines = lines

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        lines = console.render_lines(self._renderable, options, pad=False)
        limit = self._lines if self._lines is not None else options.size.height - 1
        for number, line in enumerate(lines[-max(limit, 1) :]):
            if number:
                yield Segment.line()
            yield from line


def headline(tool_name: str, arguments: dict[str, object]) -> str:
    """The one argument worth showing beside a tool call, on one short line."""
    rendered = _headline_value(tool_name, arguments).replace("\n", " ")
    if len(rendered) > _MAX_HEADLINE_LENGTH:
        rendered = rendered[:_MAX_HEADLINE_LENGTH] + "…"
    return rendered


def _headline_value(tool_name: str, arguments: dict[str, object]) -> str:
    key = _HEADLINE_ARGUMENT.get(tool_name)
    value = arguments.get(key) if key else None
    if value is None:
        value = next(iter(arguments.values()), "")
    return str(value)


def _check(config: TerminalConfig) -> None:
    """Fail at startup on a setting that would otherwise fail mid-run."""
    if config.thinking not in THINKING_MODES:
        available = ", ".join(THINKING_MODES)
        raise ConfigError(f"Unknown terminal.thinking {config.thinking!r}. Available: {available}.")
    for spec in fields(config.theme):
        if spec.name == "prompt":
            # prompt_toolkit's syntax, checked by the terminal that uses it.
            continue
        value = getattr(config.theme, spec.name)
        try:
            Style.parse(value)
        except StyleSyntaxError as exc:
            raise ConfigError(f"terminal.theme.{spec.name}: {exc}") from exc


def _parse_diff(diff: str) -> list[_DiffLine]:
    """The lines of a unified diff, numbered from its hunk headers.

    Lines before the first hunk are the `---`/`+++` file header. Inside a
    hunk, a line's first character is its kind whatever follows, so `---x`
    there is the removal of a line reading `--x`, not another header.
    """
    parsed: list[_DiffLine] = []
    old = new = 0
    in_hunk = False
    for raw in diff.splitlines():
        hunk = _HUNK.match(raw)
        if hunk is not None:
            if in_hunk:
                parsed.append(_DiffLine(_GAP, None, ""))
            old, new = int(hunk[1]), int(hunk[2])
            in_hunk = True
        elif not in_hunk:
            continue
        elif raw.startswith("-"):
            parsed.append(_DiffLine("-", old, raw[1:]))
            old += 1
        elif raw.startswith("+"):
            parsed.append(_DiffLine("+", new, raw[1:]))
            new += 1
        elif raw.startswith(" "):
            parsed.append(_DiffLine(" ", new, raw[1:]))
            old += 1
            new += 1
        # Anything else — `\ No newline at end of file` — is not a line of it.
    return parsed


def _text(content: tuple[ContentBlock, ...]) -> str:
    return "".join(b.text for b in content if isinstance(b, TextBlock))


def _last_line(content: tuple[ContentBlock, ...]) -> str:
    lines = [line for line in _text(content).splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _duration(seconds: float) -> str:
    whole = int(seconds)
    if whole < 60:
        return f"{whole}s"
    return f"{whole // 60}m {whole % 60:02d}s"
