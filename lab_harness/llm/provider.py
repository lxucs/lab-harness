"""The provider contract.

A provider MUST NOT raise for request, model, or runtime failures. It
terminates the stream with a StreamError carrying an AssistantMessage whose
stop_reason is "error" or "aborted". This is why the agent loop has almost no
try/except around streaming.

Programming errors (a bad argument type, a bug in the adapter) may still raise
— the rule covers failures that are part of normal operation.
"""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Protocol

from lab_harness.llm.events import StreamEvent
from lab_harness.llm.types import AssistantMessage, Context, Model, ToolCallBlock


@dataclass(frozen=True, slots=True)
class StreamOptions:
    max_tokens: int | None = None
    temperature: float | None = None
    #: Reasoning effort for this request, in place of the model's own; ""
    #: asks for none. Sent only to a model that supports reasoning.
    reasoning: str | None = None
    #: Cooperative cancellation. Providers should stop and emit a
    #: StreamError(reason="aborted") when this is set.
    signal: asyncio.Event | None = None


def reasoning_effort(model: Model, opts: StreamOptions) -> str:
    """The effort a request asks for: its own, else the model's. Empty asks for none."""
    return opts.reasoning if opts.reasoning is not None else model.reasoning_effort


def interrupted(
    model: Model,
    provider: str,
    notice: str,
    partial: AssistantMessage | None = None,
) -> AssistantMessage:
    """The message a stream the user stopped leaves behind.

    What arrived before the stop is reported rather than thrown away: the words
    were on screen, and the tokens — cache reads and writes among them — were
    read and charged for. A tool call is dropped, though — it was cut off
    mid-argument, the loop will not execute it, and every provider rejects a
    turn that leaves a call unanswered.

    `partial` is the message the adapter had assembled; an abort before the
    stream produced anything has none to report.
    """
    if partial is None:
        return AssistantMessage(
            content=(),
            model=model.id,
            provider=provider,
            stop_reason="aborted",
            error_message=notice,
        )
    return replace(
        partial,
        content=tuple(b for b in partial.content if not isinstance(b, ToolCallBlock)),
        stop_reason="aborted",
        error_message=notice,
    )


class Provider(Protocol):
    """Streams assistant responses. See the module docstring for the contract."""

    name: str

    def stream(
        self,
        model: Model,
        ctx: Context,
        opts: StreamOptions | None = None,
    ) -> AsyncIterator[StreamEvent]: ...
