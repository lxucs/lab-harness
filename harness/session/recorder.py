"""Persisting a run, as an event subscriber.

The recorder is a subscriber like any other — it never calls into the agent, and
the runtime does not know it exists. That is what lets persistence be optional
(`--no-save`) without a branch anywhere in the loop.

It records `message_end` and ignores everything else: deltas and tool progress
describe a run in flight, not its durable result.
"""

import hashlib
import logging
from pathlib import Path

from harness.agent.types import AgentEvent, MessageEnd
from harness.config import SessionConfig
from harness.llm.types import AssistantMessage, Model
from harness.session.store import new_session_id, session_path
from harness.session.transcript import SessionHeader, TranscriptWriter, now_timestamp

_log = logging.getLogger(__name__)


def prompt_hash(system_prompt: str) -> str:
    """A short digest of the system prompt.

    Hashed rather than stored: a prompt can carry project context that does not
    belong in a file users may share or paste into a bug report.
    """
    return hashlib.sha256(system_prompt.encode()).hexdigest()[:16]


class SessionRecorder:
    """Writes a run's messages to a transcript as they complete."""

    def __init__(self, writer: TranscriptWriter) -> None:
        self._writer = writer

    @property
    def path(self) -> Path:
        return self._writer.path

    @property
    def session_id(self) -> str:
        return self._writer.header.id

    async def __call__(self, event: AgentEvent) -> None:
        if not isinstance(event, MessageEnd):
            return
        try:
            self._writer.append_message(event.message)
            if isinstance(event.message, AssistantMessage):
                # Usage is per-request, so it rides alongside the assistant
                # message that carries it rather than being derived later.
                self._writer.append_usage(event.message.usage, event.message.model)
        except OSError:
            # A full disk must not kill a run in progress. The loop isolates
            # subscriber failures anyway; this logs the cause rather than
            # leaving a silent gap in the transcript.
            _log.exception("Could not write to session %s", self._writer.path)

    def close(self) -> None:
        self._writer.close()


def start_recorder(
    *,
    cwd: str,
    model: Model,
    system_prompt: str,
    config: SessionConfig,
    session_id: str | None = None,
    path: Path | None = None,
) -> SessionRecorder:
    """Open a transcript and return a recorder writing to it.

    Pass `path` to append to an existing session (resume); otherwise a new
    session file is created.
    """
    appending = path is not None
    if path is not None:
        identifier = path.stem
    else:
        identifier = session_id or new_session_id()
        path = session_path(cwd, identifier, config)

    header = SessionHeader(
        id=identifier,
        cwd=str(Path(cwd).expanduser().resolve()),
        model=model.id,
        timestamp=now_timestamp(),
        version=config.version,
        system_prompt_hash=prompt_hash(system_prompt),
    )
    writer = TranscriptWriter(path, header)
    writer.open()
    _log.info("%s transcript %s", "Appending to" if appending else "Writing", path)
    return SessionRecorder(writer)
