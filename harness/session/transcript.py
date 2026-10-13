"""The append-only JSONL transcript format.

Line 1 is a header; every line after it is one entry:

    {"type": "session", "version": 2, "id": "...", "cwd": "...", ...}
    {"type": "context", "restored": 0, "system_prompt": "...", "tools": [...], ...}
    {"type": "message", "timestamp": "...", "message": {...}}
    {"type": "usage", "usage": {...}, "model": "..."}

**Each invocation opens with a `context` entry** — the system prompt, tool
declarations, model and config its requests were sent with, which are
everything a request carries besides the messages. A context entry and the
messages after it form a *segment*. Together with `restored`, the number of
earlier messages the invocation carried forward, that is enough to rebuild
every request exactly: see `recorded_requests`.

`restored` is also what keeps resume honest. Resume trims an interrupted turn
from the history it hands the agent, but the file is append-only, so the
trimmed messages stay on disk; without the count, the next resume would read
them straight back into the middle of the history.

**Append-only, flushed per line.** A crashed or killed run leaves a readable
transcript up to the last completed message, which is the whole point: the
cheap, boring format survives what a clever one would not.

**Unknown entry types are skipped, not fatal.** `compaction` is reserved and not
emitted yet; a reader that chokes on an entry type it has not seen would make
every future addition a breaking change.

Serialization is the same `to_dict` the event protocol uses. The session is not
the event stream, though — events describe a run as it happens, including
streaming deltas and tool progress; the transcript records only the durable
result. Deltas are never persisted.
"""

import json
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from harness.llm.serde import message_from_dict, to_dict
from harness.llm.types import (
    AssistantMessage,
    Context,
    Message,
    Model,
    ToolDeclaration,
    Usage,
)

ENTRY_SESSION = "session"
ENTRY_CONTEXT = "context"
ENTRY_MESSAGE = "message"
ENTRY_USAGE = "usage"


@dataclass(frozen=True, slots=True)
class SessionHeader:
    """The first line of a transcript. Identifies and validates the file."""

    id: str
    cwd: str
    #: The model the session started on. Each segment records its own.
    model: str
    timestamp: str
    version: int


@dataclass(frozen=True, slots=True)
class SessionContext:
    """What one invocation's requests were sent with, besides the messages."""

    timestamp: str
    #: How many messages of the history before this entry the invocation
    #: carried forward. Fewer than the history holds means resume trimmed an
    #: interrupted turn off its end.
    restored: int
    system_prompt: str
    tools: tuple[ToolDeclaration, ...]
    #: Its limits are the ones requests were sized by.
    model: Model
    #: `config.snapshot` of the config the invocation ran with: plain data,
    #: credentials redacted. Kept as data rather than rebuilt into a `Config`,
    #: so a schema that has since changed cannot make an old record unreadable.
    config: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RecordedMessage:
    """A message as the transcript holds it: the message, and when it was written."""

    message: Message
    #: UTC, ISO 8601, taken as the entry was written: for an assistant message,
    #: when its response finished; for the user message, when its run began.
    #: The results of one tool batch are written together once the batch is
    #: done — each one's own `duration_ms` says how long it ran. Empty in a
    #: version 1 file, which recorded none.
    timestamp: str = ""


@dataclass(frozen=True, slots=True)
class Segment:
    """One invocation's part of a session: its context, and what it added."""

    context: SessionContext
    entries: tuple[RecordedMessage, ...] = ()


class TranscriptWriter:
    """Appends entries to one session file.

    Holds the file open for the session's lifetime and flushes every line, so
    the cost of durability is one flush per message rather than a reopen.
    """

    def __init__(self, path: Path, header: SessionHeader) -> None:
        self.path = path
        self.header = header
        self._handle: TextIO | None = None

    def open(self) -> None:
        """Create the file and write the header if it is new."""
        if self._handle is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        self._handle = self.path.open("a", encoding="utf-8")
        if is_new:
            self._write({"type": ENTRY_SESSION, **to_dict(self.header)})

    def append_context(self, context: SessionContext) -> None:
        self._write({"type": ENTRY_CONTEXT, **to_dict(context)})

    def append_message(self, message: Message) -> None:
        """Write one message, stamped with the time it is written."""
        self._write(
            {"type": ENTRY_MESSAGE, "timestamp": now_timestamp(), "message": to_dict(message)}
        )

    def append_usage(self, usage: Usage, model: str) -> None:
        self._write({"type": ENTRY_USAGE, "usage": to_dict(usage), "model": model})

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "TranscriptWriter":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _write(self, entry: dict[str, Any]) -> None:
        if self._handle is None:
            self.open()
        assert self._handle is not None
        self._handle.write(json.dumps(entry) + "\n")
        # Flushed per line: an interrupted run must still leave the transcript
        # readable up to its last completed message.
        self._handle.flush()


@dataclass(frozen=True, slots=True)
class Transcript:
    """A parsed session file."""

    header: SessionHeader
    segments: tuple[Segment, ...] = ()
    #: Messages recorded before any `context` entry — all of a version 1 file.
    #: Part of the history, but with no record of what they were sent with.
    unattributed: tuple[RecordedMessage, ...] = ()
    #: Entries skipped because their type was unrecognized, and malformed lines.
    #: Surfaced rather than silently dropped so a resume can say so.
    skipped: int = 0

    @property
    def history(self) -> tuple[RecordedMessage, ...]:
        """The conversation as the last invocation left it, with timestamps."""
        history = list(self.unattributed)
        for segment in self.segments:
            del history[segment.context.restored :]
            history.extend(segment.entries)
        return tuple(history)

    @property
    def messages(self) -> tuple[Message, ...]:
        """`history` without the timestamps — what resume continues from."""
        return tuple(entry.message for entry in self.history)


@dataclass(frozen=True, slots=True)
class RecordedRequest:
    """One request a session made, rebuilt from its transcript."""

    #: Exactly what the provider was handed.
    request: Context
    response: AssistantMessage
    #: When the response finished — see `RecordedMessage.timestamp`.
    timestamp: str
    model: Model
    config: dict[str, Any]


def recorded_requests(transcript: Transcript) -> tuple[RecordedRequest, ...]:
    """Every request in the session, in order: what was sent, what came back.

    The trajectory, for replay or training. One entry per assistant message,
    so errored and aborted turns are included — and an abort between turns
    yields one that never reached the model at all. Filter on
    `response.stop_reason` for the turns worth keeping.

    Messages from before any `context` entry (a version 1 file) count as
    history for what follows, but yield no requests of their own: nothing
    recorded what they were sent with.
    """
    history = [entry.message for entry in transcript.unattributed]
    found: list[RecordedRequest] = []
    for segment in transcript.segments:
        context = segment.context
        del history[context.restored :]
        for entry in segment.entries:
            message = entry.message
            if isinstance(message, AssistantMessage):
                request = Context(
                    system_prompt=context.system_prompt,
                    messages=tuple(history),
                    tools=context.tools,
                )
                found.append(
                    RecordedRequest(
                        request=request,
                        response=message,
                        timestamp=entry.timestamp,
                        model=context.model,
                        config=context.config,
                    )
                )
            history.append(message)
    return tuple(found)


class TranscriptError(Exception):
    """The file is not a transcript this version can read."""


@dataclass
class _OpenSegment:
    context: SessionContext
    entries: list[RecordedMessage] = field(default_factory=list)


def read_transcript(path: Path) -> Transcript:
    """Parse a session file into a header, its segments, and its messages.

    Malformed lines are counted and skipped rather than fatal. A transcript
    truncated mid-write by a crash is exactly the case resume exists for, and
    losing the last partial line is better than refusing the whole history.
    """
    header: SessionHeader | None = None
    segments: list[_OpenSegment] = []
    unattributed: list[RecordedMessage] = []
    skipped = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(entry, dict):
                skipped += 1
                continue

            kind = entry.get("type")
            if kind == ENTRY_SESSION:
                if header is None:
                    header = _header_from(entry)
                continue
            if kind == ENTRY_CONTEXT:
                try:
                    segments.append(_OpenSegment(_context_from(entry)))
                except (KeyError, TypeError, ValueError):
                    skipped += 1
                continue
            if kind == ENTRY_MESSAGE:
                try:
                    recorded = RecordedMessage(
                        message_from_dict(entry["message"]), str(entry.get("timestamp", ""))
                    )
                except (KeyError, TypeError, ValueError):
                    skipped += 1
                    continue
                (segments[-1].entries if segments else unattributed).append(recorded)
                continue
            if kind == ENTRY_USAGE:
                # Read, not skipped: it repeats the `usage` of the assistant
                # message beside it, so there is nothing here to rebuild.
                continue
            # Unknown types — the reserved `compaction`, anything a later
            # version adds — are skipped by design.
            skipped += 1

    if header is None:
        raise TranscriptError(f"{path.name} has no session header")
    return Transcript(
        header=header,
        segments=tuple(Segment(s.context, tuple(s.entries)) for s in segments),
        unattributed=tuple(unattributed),
        skipped=skipped,
    )


def _header_from(entry: dict[str, Any]) -> SessionHeader:
    # Keys this version does not know are ignored, for the reason unknown entry
    # types are: a version 1 header also carries `system_prompt_hash`, which
    # the context entry replaced.
    known = {f.name for f in fields(SessionHeader)}
    payload = {k: v for k, v in entry.items() if k in known}
    try:
        return SessionHeader(**payload)
    except TypeError as exc:
        raise TranscriptError(f"Malformed session header: {exc}") from exc


def _context_from(entry: dict[str, Any]) -> SessionContext:
    restored = entry["restored"]
    if not isinstance(restored, int) or isinstance(restored, bool) or restored < 0:
        raise ValueError(f"Unusable restored count: {restored!r}")
    return SessionContext(
        timestamp=entry["timestamp"],
        restored=restored,
        system_prompt=entry["system_prompt"],
        tools=tuple(ToolDeclaration(**tool) for tool in entry["tools"]),
        model=Model(**entry["model"]),
        config=entry["config"],
    )


def now_timestamp() -> str:
    return datetime.now(UTC).isoformat()
