"""Fake transports for the two HTTP provider adapters.

Shared because both suites drive a real `httpx.AsyncClient` over
`httpx.MockTransport`: the adapter's own request-building and SSE parsing run
for real, and only the socket is canned. The protocol-specific frame builders
stay in each suite, since that is the part under test.
"""

import json
from typing import Any

import httpx

from harness.llm.events import StreamDone, StreamError, StreamEvent
from harness.llm.types import AssistantMessage

#: Obviously fake. Real keys never appear in this repository.
TEST_KEY = "test-key-not-real"


class Recorder:
    """Captures the outgoing request and replays a canned SSE body."""

    def __init__(self, body: str = "", status: int = 200) -> None:
        self.body = body
        self.status = status
        self.request: httpx.Request | None = None
        self.payload: dict[str, Any] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.request = request
        if request.content:
            self.payload = json.loads(request.content)
        return httpx.Response(self.status, text=self.body)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


class Exploding:
    """A transport that fails the way a dropped connection does."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


def final(events: list[StreamEvent]) -> AssistantMessage:
    """The assembled message from the stream's terminal event."""
    last = events[-1]
    assert isinstance(last, StreamDone | StreamError)
    return last.message


def assert_blocks_open_before_deltas(events: list[StreamEvent]) -> None:
    """docs/events.md: a delta never precedes its block's start event.

    Presence is not enough. A start emitted at the *end* of the stream still
    "exists" while leaving a renderer holding deltas for a block it was never
    told about — which is exactly the bug this catches.
    """
    from harness.llm.events import (
        TextDelta,
        TextStart,
        ThinkingDelta,
        ThinkingStart,
        ToolCallDelta,
        ToolCallStart,
    )

    opened: set[int] = set()
    for event in events:
        if isinstance(event, ThinkingStart | TextStart | ToolCallStart):
            assert event.block_index not in opened, "a block was opened twice"
            opened.add(event.block_index)
        elif isinstance(event, ThinkingDelta | TextDelta | ToolCallDelta):
            assert event.block_index in opened, (
                f"block {event.block_index} streamed a delta before its start event"
            )
