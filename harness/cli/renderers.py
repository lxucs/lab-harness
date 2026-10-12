"""Rendering agent events to a rich console.

A pure event subscriber: it reads events and writes to a console, and holds no
reference to the agent. That is what keeps `cli/` deletable without touching the
runtime.

Output is **inline** — written to normal stdout rather than into a full-screen
application — so a session stays in real terminal scrollback and scrolls the way
every other command does.

Streaming assistant text uses `rich.Live`, which redraws one region in place and
leaves the final render in scrollback when it closes. Printing deltas directly
and then re-printing the finished message would show the answer twice, since
text already written to a scrollback terminal cannot be taken back.
"""

from rich.console import Console, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from harness.agent.types import (
    AgentEvent,
    MessageEnd,
    MessageStart,
    ToolExecutionEnd,
    ToolExecutionStart,
)
from harness.config import TerminalConfig
from harness.llm.types import AssistantMessage, TextBlock

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


class EventRenderer:
    """Renders a run to the console as it happens."""

    def __init__(self, console: Console, config: TerminalConfig) -> None:
        self._console = console
        self._config = config
        self._streaming = ""
        self._live: Live | None = None

    async def __call__(self, event: AgentEvent) -> None:
        if isinstance(event, MessageStart):
            if isinstance(event.message, AssistantMessage):
                self._streaming = ""
        elif event.type == "message_update":
            if event.kind == "text":
                self._streaming += event.delta
                self._update_live()
        elif isinstance(event, MessageEnd):
            self._finish_message(event.message)
        elif isinstance(event, ToolExecutionStart):
            # Close any open stream first: a tool notice printed through an
            # active Live would be swallowed by the next redraw.
            self._close_live()
            self._console.print(
                Text.assemble(
                    ("● ", "bold cyan"),
                    (event.tool_name, "bold"),
                    ("(", "dim"),
                    (_headline(event.tool_name, event.arguments), "dim"),
                    (")", "dim"),
                )
            )
        elif isinstance(event, ToolExecutionEnd):
            self._close_live()
            self._tool_result(event)

    # --- streaming --------------------------------------------------------

    def _render_streaming(self) -> RenderableType:
        if self._config.markdown:
            return Markdown(self._streaming)
        return Text(self._streaming)

    def _update_live(self) -> None:
        if not self._streaming:
            return
        if self._live is None:
            self._live = Live(
                console=self._console,
                refresh_per_second=_REFRESH_PER_SECOND,
                # Not transient: the finished answer must stay in scrollback.
                transient=False,
            )
            self._live.start()
        self._live.update(self._render_streaming())

    def _close_live(self) -> None:
        """End the streaming region and leave the cursor on a fresh line.

        `Live.stop()` leaves the cursor at the end of its last rendered line,
        so without this newline the next thing printed — a tool notice —
        starts mid-line on top of the assistant's text.
        """
        if self._live is None:
            return
        self._live.stop()
        self._live = None
        self._console.print()

    # --- messages ---------------------------------------------------------

    def _finish_message(self, message: object) -> None:
        if not isinstance(message, AssistantMessage):
            return

        body = "".join(b.text for b in message.content if isinstance(b, TextBlock))
        if body:
            # Render the authoritative text once: either as the Live region's
            # last frame, or directly when nothing streamed (a non-streaming
            # provider, or a replayed message).
            self._streaming = body
            if self._live is not None:
                self._live.update(self._render_streaming())
            else:
                self._console.print(self._render_streaming())
        self._close_live()
        self._streaming = ""

        if message.error_message:
            self._console.print(Text(message.error_message, style="bold red"))

    def _tool_result(self, event: ToolExecutionEnd) -> None:
        body = "".join(b.text for b in event.content if isinstance(b, TextBlock))
        lines = body.splitlines() or [""]
        style = "red" if event.is_error else "dim"

        shown = lines[: self._config.max_result_lines]
        for index, line in enumerate(shown):
            prefix = "  └ " if index == 0 else "    "
            self._console.print(Text(f"{prefix}{line}", style))
        if len(lines) > len(shown):
            self._console.print(Text(f"    … {len(lines) - len(shown)} more line(s)", style="dim"))


def _headline(tool_name: str, arguments: dict[str, object]) -> str:
    """The one argument worth showing beside a tool call."""
    key = _HEADLINE_ARGUMENT.get(tool_name)
    value = arguments.get(key) if key else None
    if value is None:
        value = next(iter(arguments.values()), "")
    rendered = str(value).replace("\n", " ")
    if len(rendered) > _MAX_HEADLINE_LENGTH:
        rendered = rendered[:_MAX_HEADLINE_LENGTH] + "…"
    return rendered
