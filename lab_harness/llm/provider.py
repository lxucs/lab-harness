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
from dataclasses import dataclass
from typing import Protocol

from lab_harness.llm.events import StreamEvent
from lab_harness.llm.types import Context, Model


@dataclass(frozen=True, slots=True)
class StreamOptions:
    max_tokens: int | None = None
    temperature: float | None = None
    #: Reasoning effort, for models that support it.
    reasoning: str | None = None
    #: Cooperative cancellation. Providers should stop and emit a
    #: StreamError(reason="aborted") when this is set.
    signal: asyncio.Event | None = None


class Provider(Protocol):
    """Streams assistant responses. See the module docstring for the contract."""

    name: str

    def stream(
        self,
        model: Model,
        ctx: Context,
        opts: StreamOptions | None = None,
    ) -> AsyncIterator[StreamEvent]: ...
