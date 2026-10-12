"""The append-only JSONL transcript format.

Line 1 is a header; every line after it is one entry:

    {"type": "session", "version": 1, "id": "...", "cwd": "...", ...}
    {"type": "message", "message": {...}}
    {"type": "usage", "usage": {...}, "model": "..."}

**Append-only, flushed per line.** A crashed or killed run leaves a readable
transcript up to the last completed message, which is the whole point: the
cheap, boring format survives what a clever one would not.

**Unknown entry types are skipped, not fatal.** `compaction` is reserved and not
emitted in v1; a reader that chokes on an entry type it has not seen would make
every future addition a breaking change.

Serialization is the same `to_dict` the event protocol uses. The session is not
the event stream, though — events describe a run as it happens, including
streaming deltas and tool progress; the transcript records only the durable
result. Deltas are never persisted.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from harness.llm.serde import message_from_dict, to_dict
from harness.llm.types import Message, Usage

ENTRY_SESSION = "session"
ENTRY_MESSAGE = "message"
ENTRY_USAGE = "usage"


@dataclass(frozen=True, slots=True)
class SessionHeader:
    """The first line of a transcript. Identifies and validates the file."""

    id: str
    cwd: str
    model: str
    timestamp: str
    version: int
    #: Present so a resumed session can warn when the prompt has changed since.
    #: Stored as a hash rather than the text: prompts can carry project context
    #: that does not belong in a file users may share.
    system_prompt_hash: str = ""


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

    def append_message(self, message: Message) -> None:
        self._write({"type": ENTRY_MESSAGE, "message": to_dict(message)})

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
    messages: tuple[Message, ...]
    #: Entries skipped because their type was unrecognized, and malformed lines.
    #: Surfaced rather than silently dropped so a resume can say so.
    skipped: int = 0


class TranscriptError(Exception):
    """The file is not a transcript this version can read."""


def read_transcript(path: Path) -> Transcript:
    """Parse a session file into a header and its messages.

    Malformed lines are counted and skipped rather than fatal. A transcript
    truncated mid-write by a crash is exactly the case resume exists for, and
    losing the last partial line is better than refusing the whole history.
    """
    header: SessionHeader | None = None
    messages: list[Message] = []
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
            if kind == ENTRY_MESSAGE:
                try:
                    messages.append(message_from_dict(entry["message"]))
                except (KeyError, TypeError, ValueError):
                    skipped += 1
                continue
            # Unknown types — `usage`, the reserved `compaction`, anything a
            # later version adds — are skipped by design.
            skipped += 1

    if header is None:
        raise TranscriptError(f"{path.name} has no session header")
    return Transcript(header=header, messages=tuple(messages), skipped=skipped)


def _header_from(entry: dict[str, Any]) -> SessionHeader:
    payload = {k: v for k, v in entry.items() if k != "type"}
    try:
        return SessionHeader(**payload)
    except TypeError as exc:
        raise TranscriptError(f"Malformed session header: {exc}") from exc


def iter_entries(path: Path) -> Iterator[dict[str, Any]]:
    """Raw entries, for tooling that wants more than messages."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                yield entry


def now_timestamp() -> str:
    return datetime.now(UTC).isoformat()
