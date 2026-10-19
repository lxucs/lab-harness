"""A saved session as one HTML page: `lab-harness --export`.

The page is a *view* of the transcript, never a second record: it reads the
file, writes a page beside it, and changes nothing. It shows what the terminal
cannot — every invocation's system prompt, tools and config, usage and cost per
request — and folds what would bury the conversation: thinking, tool input and
output, those declarations.

**Self-contained.** One file with its styles and script inline, opened from
disk with no server and nothing fetched. A Content-Security-Policy says so to
the browser: nothing may load, and the only script that runs is the template's
own, pinned by its hash — so session text that somehow reached the page as
markup still could not execute.

**Everything from the session is escaped.** A transcript holds what models and
tools wrote, which is untrusted. Text is escaped wherever it lands; the model's
Markdown is rendered with raw HTML off, so `<script>` in an answer reads as
text; images are inlined only for a known image type.

The shell, styles and script live in `export.html` beside this module. Slots
there are `{{name}}`, filled in one pass, so text that happens to look like a
slot is never filled in turn.
"""

import base64
import difflib
import hashlib
import html
import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt

from lab_harness.cli.renderers import headline
from lab_harness.cli.usage import (
    cache_hit_rate,
    cost_of,
    format_cost,
    format_tokens,
    price_for,
    prompt_tokens,
    summarize,
)
from lab_harness.config import Config
from lab_harness.llm.types import (
    AssistantMessage,
    ContentBlock,
    ImageBlock,
    ProviderBlock,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from lab_harness.session import RecordedMessage, SessionContext, Transcript, read_transcript

TEMPLATE_FILE = Path(__file__).with_name("export.html")
#: The page's extension, written beside the transcript.
SUFFIX = ".html"

#: Image types inlined as data URLs. Anything else is named, not shown: the
#: type lands inside an attribute, so it is allow-listed rather than escaped.
_IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
_BASE64 = re.compile(r"[A-Za-z0-9+/=\s]*")
_SLOT = re.compile(r"\{\{(\w+)\}\}")
_SCRIPT = re.compile(r"<script>(.*?)</script>", re.DOTALL)
#: Characters of a question the turn index shows, and of the page's heading.
_INDEX_CHARS = 64
_HEADING_CHARS = 96
#: Characters of a call's first line of output shown while it is folded.
_PREVIEW_CHARS = 140
#: A one-line argument up to this long is a `key value` row, not a block.
_INLINE_CHARS = 100

#: The model's answers. Raw HTML off, so markup in an answer is text; images
#: off, because the page loads nothing and a remote image would only break.
#: markdown-it already refuses `javascript:` and similar link targets.
_MARKDOWN = (
    MarkdownIt("commonmark", {"html": False}).enable(["table", "strikethrough"]).disable("image")
)


def export_html(path: Path, config: Config) -> Path:
    """Write the page for the transcript at `path` beside it; return where."""
    target = path.with_suffix(SUFFIX)
    target.write_text(render_html(read_transcript(path), config), encoding="utf-8")
    return target


def render_html(transcript: Transcript, config: Config) -> str:
    """The whole page for a parsed transcript."""
    page = _Page(transcript, config)
    return _fill(
        {
            "title": _escape(transcript.header.id),
            "overview": page.overview(),
            "index": page.index(),
            "body": page.body(),
            "exported": _time(datetime.now(UTC).isoformat(), "datetime"),
        }
    )


# --- the template ---------------------------------------------------------


def _fill(slots: dict[str, str]) -> str:
    template = TEMPLATE_FILE.read_text(encoding="utf-8")
    script = _SCRIPT.search(template)
    if script is None:
        raise RuntimeError(f"{TEMPLATE_FILE.name} has no inline script")
    digest = hashlib.sha256(script.group(1).encode("utf-8")).digest()
    slots = {**slots, "script_hash": "sha256-" + base64.b64encode(digest).decode("ascii")}
    # One pass: a slot's value is never scanned for slots of its own.
    return _SLOT.sub(lambda match: slots[match.group(1)], template)


# --- the page -------------------------------------------------------------


@dataclass
class _Turn:
    number: int
    question: str
    messages: list[RecordedMessage] = field(default_factory=list)


@dataclass
class _Segment:
    context: SessionContext | None
    #: "Started", "Resumed" or "Context changed".
    label: str
    changed: tuple[str, ...]
    dropped: int
    config_changed: bool
    #: Messages before the segment's first question, if any.
    loose: list[RecordedMessage] = field(default_factory=list)
    turns: list[_Turn] = field(default_factory=list)


class _Page:
    def __init__(self, transcript: Transcript, config: Config) -> None:
        self._transcript = transcript
        self._config = config
        self._results = {
            entry.message.tool_call_id: entry.message
            for entry in _every_entry(transcript)
            if isinstance(entry.message, ToolResultMessage)
        }
        self._calls = {
            call.id
            for entry in _every_entry(transcript)
            if isinstance(entry.message, AssistantMessage)
            for call in entry.message.tool_calls
        }
        self._segments = _segments(transcript)

    # --- the parts the template places ------------------------------------

    def overview(self) -> str:
        header = self._transcript.header
        messages = [entry.message for entry in _every_entry(self._transcript)]
        questions = [m for m in messages if isinstance(m, UserMessage)]
        summary = summarize(messages, self._config.models)
        usage = summary.usage
        heading = _first_line(_text(questions[0].content), _HEADING_CHARS) if questions else ""
        models = list(dict.fromkeys(m.model for m in messages if isinstance(m, AssistantMessage)))

        stats = [
            ("Turns", str(len(questions))),
            ("Requests", str(summary.requests)),
            ("Tool calls", str(len(self._calls))),
            ("Input", format_tokens(prompt_tokens(usage))),
            ("Output", format_tokens(usage.output)),
        ]
        rate = cache_hit_rate(usage)
        if rate is not None:
            stats.append(("Cached", f"{rate:.0%}"))
        if summary.cost is not None:
            stats.append(("Cost", format_cost(summary.cost, self._config.terminal.currency_symbol)))

        where = [
            f"<span>Session <code>{_escape(header.id)}</code></span>",
            f"<span>in <code>{_escape(header.cwd)}</code></span>",
            f"<span>{_escape(', '.join(models or [header.model]))}</span>",
            f"<span>started {_time(header.timestamp, 'datetime')}</span>",
        ]
        parts = [
            '<section class="overview">',
            f"<h1>{_escape(heading) or 'Session ' + _escape(header.id)}</h1>",
            f'<div class="where">{"".join(where)}</div>',
            '<div class="stats">',
            *(
                f'<div class="stat"><span class="label">{label}</span>'
                f'<span class="value">{_escape(value)}</span></div>'
                for label, value in stats
            ),
            "</div>",
            "</section>",
        ]
        if self._transcript.skipped:
            parts.append(
                f'<p class="note">{self._transcript.skipped} unreadable or unknown '
                "entrie(s) in the file were skipped.</p>"
            )
        return "\n".join(parts)

    def index(self) -> str:
        items: list[str] = []
        for segment in self._segments:
            if len(self._segments) > 1 or segment.label != "Started":
                when = _time(segment.context.timestamp, "datetime") if segment.context else ""
                items.append(f'<li class="segment-mark">{segment.label} · {when}</li>')
            for turn in segment.turns:
                items.append(
                    f'<li><a href="#turn-{turn.number}"><span class="n">{turn.number}</span>'
                    f'<span class="t">{_escape(_first_line(turn.question, _INDEX_CHARS))}</span>'
                    "</a></li>"
                )
        return f"<ol>{''.join(items)}</ol>"

    def body(self) -> str:
        if not any(True for _ in _every_entry(self._transcript)):
            segments = "\n".join(self._segment(s) for s in self._segments)
            return segments + '\n<p class="empty">No messages in this session.</p>'
        return "\n".join(self._segment(s) for s in self._segments)

    # --- segments -----------------------------------------------------------

    def _segment(self, segment: _Segment) -> str:
        parts = ['<section class="segment">', self._segment_head(segment)]
        parts += [self._message(entry) for entry in segment.loose]
        for turn in segment.turns:
            parts.append(f'<article class="turn" id="turn-{turn.number}">')
            parts += [self._message(entry, turn.number) for entry in turn.messages]
            parts.append("</article>")
        parts.append("</section>")
        return "\n".join(parts)

    def _segment_head(self, segment: _Segment) -> str:
        context = segment.context
        chips = [f'<span class="chip">{segment.label}</span>']
        if context is None:
            chips.append('<span class="chip quiet">no recorded context</span>')
            return f'<div class="segment-head">{"".join(chips)}</div>'
        chips.append(_time(context.timestamp, "datetime"))
        chips.append(f'<span class="chip quiet">{_escape(context.model.id)}</span>')
        if context.model.reasoning_effort:
            effort = _escape(context.model.reasoning_effort)
            chips.append(f'<span class="chip quiet">effort {effort}</span>')
        if segment.changed:
            chips.append(
                f'<span class="chip warn">changed: {_escape(", ".join(segment.changed))}</span>'
            )
        if segment.dropped:
            chips.append(
                f'<span class="chip warn">dropped {segment.dropped} message(s) '
                "from an interrupted turn</span>"
            )

        details = [
            _fold(
                "context",
                f"System prompt · {len(context.system_prompt):,} characters",
                f'<div class="decl-text">{_escape(context.system_prompt)}</div>'
                if context.system_prompt
                else '<p class="empty">None.</p>',
            ),
            _fold(
                "context",
                f"Tools · {len(context.tools)}",
                "".join(
                    '<div class="decl">'
                    f'<div class="decl-name">{_escape(tool.name)}</div>'
                    f'<div class="decl-text">{_escape(tool.description)}</div>'
                    f"<pre>{_escape(_json(tool.parameters))}</pre>"
                    "</div>"
                    for tool in context.tools
                )
                or '<p class="empty">None.</p>',
            ),
        ]
        if segment.config_changed:
            details.append(
                _fold(
                    "context",
                    "Config · recorded, not sent to the model · credentials redacted",
                    f'<pre class="long">{_escape(_json(context.config))}</pre>',
                )
            )
        return (
            f'<div class="segment-head">{"".join(chips)}'
            f'<div class="segment-context">{"".join(details)}</div></div>'
        )

    # --- messages -----------------------------------------------------------

    def _message(self, entry: RecordedMessage, turn: int | None = None) -> str:
        message = entry.message
        if isinstance(message, UserMessage):
            label = f"User · #{turn}" if turn is not None else "User"
            return (
                '<div class="msg user">'
                f'<div class="msg-head"><span class="who">{label}</span>'
                f"{_time(entry.timestamp, 'time')}</div>"
                f'<div class="bubble">{self._user_content(message.content)}</div>'
                "</div>"
            )
        if isinstance(message, AssistantMessage):
            return self._assistant(entry, message)
        if isinstance(message, ToolResultMessage):
            if message.tool_call_id in self._calls:
                return ""  # Drawn under its call.
            return f'<div class="msg">{self._call(None, message)}</div>'
        return ""

    def _user_content(self, content: tuple[ContentBlock, ...]) -> str:
        parts: list[str] = []
        for block in content:
            if isinstance(block, TextBlock):
                parts.append(_escape(block.text))
            elif isinstance(block, ImageBlock):
                parts.append(_image(block))
        return "".join(parts)

    def _assistant(self, entry: RecordedMessage, message: AssistantMessage) -> str:
        head = [
            '<span class="who">Assistant</span>',
            f"<span>{_escape(message.model)}</span>",
            _time(entry.timestamp, "time"),
        ]
        if entry.rewrite is not None:
            head.append('<span class="chip quiet">request rewritten by a hook</span>')
        if message.stop_reason == "aborted":
            head.append('<span class="chip warn">cancelled</span>')
        elif message.stop_reason == "length":
            head.append('<span class="chip warn">cut off at max_tokens</span>')
        if line := self._usage(message):
            head.append(f'<span class="usage">{_escape(line)}</span>')

        body: list[str] = []
        for block in message.content:
            if isinstance(block, ThinkingBlock):
                if block.thinking.strip():
                    words = len(block.thinking.split())
                    body.append(
                        _fold(
                            "thinking",
                            f'<span class="glyph">✻</span> Thought · {words:,} '
                            f"word{'' if words == 1 else 's'}",
                            _escape(block.thinking.strip()),
                            escape_summary=False,
                        )
                    )
            elif isinstance(block, TextBlock):
                if block.text.strip():
                    body.append(f'<div class="markdown">{_MARKDOWN.render(block.text)}</div>')
            elif isinstance(block, ToolCallBlock):
                body.append(self._call(block, self._results.get(block.id)))
            elif isinstance(block, ImageBlock):
                body.append(_image(block))
            elif isinstance(block, ProviderBlock):
                body.append(
                    _fold(
                        "provider",
                        f"{block.provider_type} (provider block)",
                        f"<pre>{_escape(_json(block.data))}</pre>",
                    )
                )
        if message.error_message:
            body.append(f'<div class="error-box">{_escape(message.error_message)}</div>')

        return (
            '<div class="msg assistant">'
            f'<div class="msg-head">{"".join(head)}</div>'
            f'<div class="content">{"".join(body)}</div>'
            "</div>"
        )

    def _usage(self, message: AssistantMessage) -> str:
        usage = message.usage
        if usage.total == 0:
            return ""
        parts = [f"{format_tokens(prompt_tokens(usage))} in", f"{format_tokens(usage.output)} out"]
        rate = cache_hit_rate(usage)
        if rate is not None:
            parts.append(f"{rate:.0%} cached")
        if (cost := self._cost(message.model, usage)) is not None:
            parts.append(cost)
        return " · ".join(parts)

    def _cost(self, model: str, usage: Usage) -> str | None:
        price = price_for(model, self._config.models)
        if price is None:
            return None
        return format_cost(cost_of(usage, price), self._config.terminal.currency_symbol)

    # --- tool calls ---------------------------------------------------------

    def _call(self, call: ToolCallBlock | None, result: ToolResultMessage | None) -> str:
        name = call.name if call is not None else result.tool_name if result else "tool"
        state = "pending" if result is None else "error" if result.is_error else "ok"
        output = _text(result.content) if result is not None else ""
        shown = headline(name, call.arguments) if call is not None else ""

        summary = [
            '<span class="status"></span>',
            f'<span class="tool-name">{_escape(name)}</span>',
            f'<span class="headline">{_escape(shown)}</span>',
            f'<span class="duration">{_duration(result.duration_ms) if result else ""}</span>',
        ]
        if result is None:
            summary.append('<span class="preview">no result recorded</span>')
        elif preview := _first_line(output, _PREVIEW_CHARS):
            summary.append(f'<span class="preview">{_escape(preview)}</span>')

        inside: list[str] = []
        if call is not None:
            inside.append(f'<div><div class="section-label">Input</div>{_arguments(call)}</div>')
        if result is None:
            inside.append(
                '<p class="empty">No result recorded — the turn was interrupted before '
                "the call returned.</p>"
            )
        else:
            images = "".join(_image(b) for b in result.content if isinstance(b, ImageBlock))
            body = f'<pre class="output">{_escape(output)}</pre>' if output else ""
            if not body and not images:
                body = '<p class="empty">(no output)</p>'
            inside.append(f'<div><div class="section-label">Output</div>{body}{images}</div>')

        return (
            f'<details class="tool-call {state}">'
            f"<summary>{''.join(summary)}</summary>"
            f'<div class="inside">{"".join(inside)}</div>'
            "</details>"
        )


# --- segments, from the transcript ----------------------------------------


def _every_entry(transcript: Transcript) -> Iterator[RecordedMessage]:
    """Every message the file holds, once, in the order written.

    Not `Transcript.history`: that drops a turn resume trimmed, and the page
    shows it where it happened.
    """
    yield from transcript.unattributed
    for segment in transcript.segments:
        yield from segment.entries


def _segments(transcript: Transcript) -> list[_Segment]:
    shown: list[_Segment] = []
    turns = 0
    history = len(transcript.unattributed)
    previous: SessionContext | None = None

    def fill(segment: _Segment, entries: Sequence[RecordedMessage]) -> None:
        nonlocal turns
        for entry in entries:
            if isinstance(entry.message, UserMessage):
                turns += 1
                segment.turns.append(_Turn(turns, _text(entry.message.content)))
            if segment.turns:
                segment.turns[-1].messages.append(entry)
            else:
                segment.loose.append(entry)

    if transcript.unattributed:
        segment = _Segment(None, "Recorded before contexts", (), 0, False)
        fill(segment, transcript.unattributed)
        shown.append(segment)

    pending_drop = 0
    for position, recorded in enumerate(transcript.segments):
        context = recorded.context
        pending_drop += max(history - context.restored, 0)
        history = context.restored + len(recorded.entries)
        # An invocation that sent nothing — opened and left — says nothing
        # worth a heading, unless it is all there is.
        if not recorded.entries and (shown or position < len(transcript.segments) - 1):
            continue
        changed = _changes(previous, context)
        label = "Started" if previous is None else "Context changed" if changed else "Resumed"
        config_changed = previous is None or previous.config != context.config
        segment = _Segment(context, label, changed, pending_drop, config_changed)
        pending_drop = 0
        fill(segment, recorded.entries)
        shown.append(segment)
        previous = context
    return shown


def _changes(previous: SessionContext | None, context: SessionContext) -> tuple[str, ...]:
    if previous is None:
        return ()
    changed = []
    if previous.system_prompt != context.system_prompt:
        changed.append("system prompt")
    if previous.tools != context.tools:
        changed.append("tools")
    if previous.model.id != context.model.id:
        changed.append("model")
    if previous.model.reasoning_effort != context.model.reasoning_effort:
        changed.append("reasoning effort")
    return tuple(changed)


# --- small pieces ---------------------------------------------------------


def _arguments(call: ToolCallBlock) -> str:
    """A call's input: an edit as the change it makes, anything else by argument."""
    arguments = call.arguments
    old, new = arguments.get("old"), arguments.get("new")
    if call.name == "edit" and isinstance(old, str) and isinstance(new, str):
        before, after = old.splitlines(), new.splitlines()
        lines = difflib.unified_diff(before, after, n=max(len(before), len(after)), lineterm="")
        rows = []
        for line in lines:
            if line.startswith(("---", "+++", "@@")):
                continue
            kind = {"+": "add", "-": "del"}.get(line[:1], "")
            rows.append(
                f'<span class="{kind}">{_escape(line)}</span>'
                if kind
                else f"<span>{_escape(line)}</span>"
            )
        rest = {k: v for k, v in arguments.items() if k not in ("old", "new")}
        return _named(rest) + f'<pre class="diff">{"".join(rows)}</pre>'
    return _named(arguments) or '<p class="empty">(none)</p>'


def _named(arguments: dict[str, object]) -> str:
    """Short values as `key value` rows, long or multi-line ones as blocks."""
    rows: list[str] = []
    blocks: list[str] = []
    for key, value in arguments.items():
        shown = value if isinstance(value, str) else _json(value)
        if "\n" in shown or len(shown) > _INLINE_CHARS:
            blocks.append(f'<div class="arg-key">{_escape(key)}</div><pre>{_escape(shown)}</pre>')
        else:
            rows.append(f"<dt>{_escape(key)}</dt><dd>{_escape(shown)}</dd>")
    table = f'<dl class="args">{"".join(rows)}</dl>' if rows else ""
    return table + "".join(blocks)


def _fold(kind: str, summary: str, inside: str, *, escape_summary: bool = True) -> str:
    title = _escape(summary) if escape_summary else summary
    return (
        f'<details class="{kind}"><summary>{title}</summary>'
        f'<div class="inside">{inside}</div></details>'
    )


def _image(block: ImageBlock) -> str:
    if block.mime_type not in _IMAGE_TYPES or not _BASE64.fullmatch(block.data):
        return '<p class="empty">An image, of a type this page does not show.</p>'
    return f'<img class="image" alt="" src="data:{block.mime_type};base64,{_escape(block.data)}">'


def _time(stamp: str, style: str) -> str:
    """A timestamp the page's script re-renders in the reader's own time zone.

    What it shows without the script is UTC, labelled as such.
    """
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        return ""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    fallback = moment.astimezone(UTC).strftime(
        "%Y-%m-%d %H:%M UTC" if style == "datetime" else "%H:%M:%S UTC"
    )
    return f'<time datetime="{_escape(moment.isoformat())}" data-format="{style}">{fallback}</time>'


def _duration(milliseconds: int | None) -> str:
    if milliseconds is None:
        return ""
    if milliseconds < 1_000:
        return f"{milliseconds}ms"
    seconds = milliseconds / 1_000
    if seconds < 60:
        return f"{seconds:.1f}s".replace(".0s", "s")
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes}m {rest:02d}s"


def _text(content: tuple[ContentBlock, ...]) -> str:
    return "".join(block.text for block in content if isinstance(block, TextBlock))


def _first_line(text: str, limit: int) -> str:
    line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return line if len(line) <= limit else line[:limit].rstrip() + "…"


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


__all__ = ["SUFFIX", "TEMPLATE_FILE", "export_html", "render_html"]
