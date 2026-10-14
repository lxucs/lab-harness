"""Rebuilding an agent's transcript from a session file.

Resume is a read, a validation, and a list assignment. Keeping it that small is
deliberate: the transcript is the same message list the agent already holds, so
there is no separate "restored" representation to drift out of sync.
"""

from dataclasses import dataclass
from pathlib import Path

from lab_harness.config import SessionConfig
from lab_harness.llm.types import AssistantMessage, Message, ToolResultMessage
from lab_harness.session.store import find_session, latest_session
from lab_harness.session.transcript import Transcript, read_transcript


@dataclass(frozen=True, slots=True)
class ResumedSession:
    path: Path
    #: The file as read, before trimming.
    transcript: Transcript
    #: The history to hand the agent: the transcript's, less any interrupted
    #: turn at its end.
    messages: tuple[Message, ...]
    #: Non-fatal problems worth telling the user about — a trimmed trailing
    #: turn, skipped entries, a changed system prompt.
    warnings: tuple[str, ...] = ()


class ResumeError(Exception):
    """No session to resume, or the file cannot be used."""


def resume(
    *,
    cwd: str,
    config: SessionConfig,
    session_id: str | None = None,
    system_prompt: str | None = None,
) -> ResumedSession:
    """Load the named session, or the most recent one for this directory.

    Pass the `system_prompt` the agent will run with to be warned when it
    differs from the one the session last recorded.
    """
    if session_id:
        path = find_session(cwd, session_id, config)
        if path is None:
            raise ResumeError(f"No session matching {session_id!r} for this directory.")
    else:
        path = latest_session(cwd, config)
        if path is None:
            raise ResumeError("No previous session for this directory.")

    transcript = read_transcript(path)
    if transcript.header.version > config.version:
        raise ResumeError(
            f"{path.name} was written by a newer version "
            f"(file v{transcript.header.version}, this build reads v{config.version})."
        )

    warnings: list[str] = []
    if transcript.skipped:
        warnings.append(f"{transcript.skipped} unreadable or unknown entrie(s) skipped.")
    if (
        system_prompt is not None
        and transcript.segments
        and transcript.segments[-1].context.system_prompt != system_prompt
    ):
        warnings.append("The system prompt has changed since this session was recorded.")

    messages, trimmed = trim_to_resumable(transcript.messages)
    if trimmed:
        warnings.append(
            f"Dropped {trimmed} trailing message(s) from an interrupted turn so the "
            f"conversation can continue."
        )

    return ResumedSession(
        path=path, transcript=transcript, messages=messages, warnings=tuple(warnings)
    )


def trim_to_resumable(messages: tuple[Message, ...]) -> tuple[tuple[Message, ...], int]:
    """Drop a trailing turn the provider would reject, returning how many.

    A session killed mid-turn can end with an assistant message requesting tool
    calls whose results were never written. Providers require every tool call to
    have a matching result in the next turn, so replaying that transcript fails
    the first request — the one case where resume would reliably break.

    Trimming the incomplete turn costs the user one re-ask; not trimming it
    costs them the whole session.
    """
    result = list(messages)

    while result:
        last = result[-1]

        # An assistant turn requesting tools with no results after it at all.
        if isinstance(last, AssistantMessage):
            if not last.tool_calls:
                break
            result.pop()
            continue

        if isinstance(last, ToolResultMessage):
            assistant_index = _last_assistant_index(result)
            if assistant_index is None:
                # Results with no requesting turn — a badly truncated file.
                result.pop()
                continue
            assistant = result[assistant_index]
            assert isinstance(assistant, AssistantMessage)
            answered = {
                m.tool_call_id
                for m in result[assistant_index + 1 :]
                if isinstance(m, ToolResultMessage)
            }
            if all(call.id in answered for call in assistant.tool_calls):
                break
            # A partially answered batch: drop the whole turn, not just the
            # tail. Providers require *every* call in a turn to have a result,
            # so an assistant message with only some of its results attached is
            # exactly as unsendable as one with none.
            del result[assistant_index:]
            continue

        break

    return tuple(result), len(messages) - len(result)


def _last_assistant_index(messages: list[Message]) -> int | None:
    """Index of the most recent assistant message, or None if there is none."""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], AssistantMessage):
            return index
    return None
