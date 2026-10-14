"""Persisting a run, as an event subscriber.

The recorder is a subscriber like any other — it never calls into the agent, and
the runtime does not know it exists. That is what lets persistence be optional
(`--no-save`) without a branch anywhere in the loop.

It records three events and ignores the rest, since deltas and tool progress
describe a run in flight rather than its durable result:

- `message_end` — every message, as it completes.
- `agent_start` — only when the run's system prompt, tools, model or history
  differ from what the last `context` entry says, which opens a new segment.
  So a change made between runs is on record before the first request it
  affects, and an unchanged session writes nothing extra.
- `request_rewritten` — what a `prepare_request` hook sent instead.
"""

import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from lab_harness.agent.types import AgentEvent, AgentStart, MessageEnd, RequestRewritten
from lab_harness.config import Config, snapshot
from lab_harness.llm.types import AssistantMessage, Message, Model, ToolDeclaration
from lab_harness.session.store import new_session_id, session_path
from lab_harness.session.transcript import (
    Rewrite,
    SessionContext,
    SessionHeader,
    TranscriptWriter,
    now_timestamp,
)

_log = logging.getLogger(__name__)


class SessionRecorder:
    """Writes a session to a transcript as it happens."""

    def __init__(self, writer: TranscriptWriter, context: SessionContext) -> None:
        """`context` is the entry already written, which opened this segment."""
        self._writer = writer
        self._written = 0
        self._context = context
        # Messages in the agent's history: those the segment carried forward,
        # plus every one announced since. What the next run should start from.
        self._history = context.restored

    @property
    def path(self) -> Path:
        return self._writer.path

    @property
    def session_id(self) -> str:
        return self._writer.header.id

    async def __call__(self, event: AgentEvent) -> None:
        if isinstance(event, AgentStart):
            self._record_context(event)
        elif isinstance(event, RequestRewritten):
            self._record(self._writer.append_rewrite, Rewrite(event.messages, event.model))
        elif isinstance(event, MessageEnd):
            self._record_message(event)

    def _record_context(self, event: AgentStart) -> None:
        current = self._context
        if (event.system_prompt, event.tools, event.model, event.history) == (
            current.system_prompt,
            current.tools,
            current.model,
            self._history,
        ):
            return
        self._context = replace(
            current,
            timestamp=now_timestamp(),
            restored=event.history,
            system_prompt=event.system_prompt,
            tools=event.tools,
            model=event.model,
        )
        self._history = event.history
        _log.debug("Context changed: new segment in %s", self.path.name)
        self._record(self._writer.append_context, self._context)

    def _record_message(self, event: MessageEnd) -> None:
        self._history += 1
        self._record(self._write_message, event.message)

    def _write_message(self, message: Message) -> None:
        self._writer.append_message(message)
        self._written += 1
        if isinstance(message, AssistantMessage):
            # Usage is per-request, so it rides alongside the assistant
            # message that carries it rather than being derived later.
            self._writer.append_usage(message.usage, message.model)

    def _record[T](self, write: Callable[[T], None], entry: T) -> None:
        try:
            write(entry)
        except OSError:
            # A full disk must not kill a run in progress. The loop isolates
            # subscriber failures anyway; this logs the cause rather than
            # leaving a silent gap in the transcript.
            _log.exception("Could not write to session %s", self._writer.path)

    def close(self) -> None:
        _log.debug("Closing transcript %s after %d new message(s)", self.path.name, self._written)
        self._writer.close()


def start_recorder(
    *,
    cwd: str,
    model: Model,
    system_prompt: str,
    config: Config,
    restored: int,
    tools: tuple[ToolDeclaration, ...] = (),
    session_id: str | None = None,
    path: Path | None = None,
) -> SessionRecorder:
    """Open a transcript, record what this invocation runs with, and return a
    recorder writing to it.

    Pass `path` to append to an existing session (resume); otherwise a new
    session file is created. `restored` is how many of the session's messages
    the agent was handed — 0 for a new session. It is required, and passed
    rather than worked out from the file, because resume may have trimmed an
    interrupted turn: the count is what keeps that turn out of the history the
    next read rebuilds.
    """
    appending = path is not None
    if path is not None:
        identifier = path.stem
    else:
        identifier = session_id or new_session_id()
        path = session_path(cwd, identifier, config.session)

    header = SessionHeader(
        id=identifier,
        cwd=str(Path(cwd).expanduser().resolve()),
        model=model.id,
        timestamp=now_timestamp(),
        version=config.session.version,
    )
    writer = TranscriptWriter(path, header)
    writer.open()
    context = SessionContext(
        timestamp=now_timestamp(),
        restored=restored,
        system_prompt=system_prompt,
        tools=tools,
        model=model,
        config=snapshot(config),
    )
    writer.append_context(context)
    _log.info("%s transcript %s", "Appending to" if appending else "Writing", path)
    return SessionRecorder(writer, context)
