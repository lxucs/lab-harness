"""A provider that replays scripted events. The backbone of the offline suite.

Build scripts with the `*_response` helpers rather than by hand — they produce
protocol-correct event sequences, so a test that passes against the fake is
testing the loop, not a malformed script.
"""

from collections.abc import AsyncIterator, Sequence
from itertools import count

from harness.llm.events import (
    StreamDone,
    StreamError,
    StreamEvent,
    StreamStart,
    TextDelta,
    TextEnd,
    TextStart,
    ToolCallEnd,
    ToolCallStart,
)
from harness.llm.provider import StreamOptions
from harness.llm.types import (
    AssistantMessage,
    Context,
    Model,
    TextBlock,
    ToolCallBlock,
    Usage,
)

FAKE_MODEL = Model(
    id="fake-model",
    name="Fake Model",
    provider="fake",
    context_window=200_000,
    max_tokens=8_192,
)

_DEFAULT_USAGE = Usage(input=10, output=5)


def text_response(
    value: str,
    *,
    model: str = FAKE_MODEL.id,
    provider: str = "fake",
    chunk_size: int = 0,
) -> list[StreamEvent]:
    """A complete turn that returns `value` as text.

    chunk_size=0 emits one delta; larger values split the text, which is how
    streaming-render tests get multiple updates.
    """
    chunks = (
        [value]
        if chunk_size <= 0
        else [value[i : i + chunk_size] for i in range(0, len(value), chunk_size)]
    )
    message = AssistantMessage(
        content=(TextBlock(text=value),),
        model=model,
        provider=provider,
        usage=_DEFAULT_USAGE,
        stop_reason="stop",
    )
    return [
        StreamStart(),
        TextStart(block_index=0),
        *(TextDelta(block_index=0, delta=c) for c in chunks),
        TextEnd(block_index=0, text=value),
        StreamDone(message=message, reason="stop"),
    ]


def tool_call_response(
    calls: Sequence[ToolCallBlock],
    *,
    preamble: str | None = None,
    model: str = FAKE_MODEL.id,
    provider: str = "fake",
) -> list[StreamEvent]:
    """A turn that requests one or more tool calls, optionally with leading text."""
    events: list[StreamEvent] = [StreamStart()]
    content: list[TextBlock | ToolCallBlock] = []
    index = count()

    if preamble is not None:
        i = next(index)
        events += [
            TextStart(block_index=i),
            TextDelta(block_index=i, delta=preamble),
            TextEnd(block_index=i, text=preamble),
        ]
        content.append(TextBlock(text=preamble))

    for call in calls:
        i = next(index)
        events += [ToolCallStart(block_index=i), ToolCallEnd(block_index=i, tool_call=call)]
        content.append(call)

    message = AssistantMessage(
        content=tuple(content),
        model=model,
        provider=provider,
        usage=_DEFAULT_USAGE,
        stop_reason="tool_use",
    )
    events.append(StreamDone(message=message, reason="tool_use"))
    return events


def error_response(
    error: str,
    *,
    model: str = FAKE_MODEL.id,
    provider: str = "fake",
    reason: str = "error",
) -> list[StreamEvent]:
    """A turn that fails. Note it terminates with StreamError, never an exception."""
    message = AssistantMessage(
        content=(),
        model=model,
        provider=provider,
        stop_reason="error" if reason == "error" else "aborted",
        error_message=error,
    )
    return [StreamError(message=message, reason=reason)]  # type: ignore[arg-type]


def truncated_tool_call_response(
    calls: Sequence[ToolCallBlock],
    *,
    model: str = FAKE_MODEL.id,
    provider: str = "fake",
) -> list[StreamEvent]:
    """Tool calls cut off by the output token limit.

    Arguments may parse and validate while being silently incomplete, so the
    loop must refuse to execute any of them.
    """
    events = tool_call_response(calls, model=model, provider=provider)
    done = events[-1]
    assert isinstance(done, StreamDone)
    message = AssistantMessage(
        content=done.message.content,
        model=model,
        provider=provider,
        usage=done.message.usage,
        stop_reason="length",
    )
    events[-1] = StreamDone(message=message, reason="length")
    return events


class FakeProvider:
    """Replays one script per request, in order."""

    name = "fake"

    def __init__(self, scripts: Sequence[Sequence[StreamEvent]]) -> None:
        self._scripts = [list(s) for s in scripts]
        self._index = 0
        #: Contexts the caller sent, in order. Assert against this.
        self.requests: list[Context] = []

    async def stream(
        self,
        model: Model,
        ctx: Context,
        opts: StreamOptions | None = None,
    ) -> AsyncIterator[StreamEvent]:
        if self._index >= len(self._scripts):
            raise AssertionError(
                f"FakeProvider script exhausted: request {self._index + 1} was made "
                f"but only {len(self._scripts)} script(s) were provided."
            )
        script = self._scripts[self._index]
        self._index += 1
        self.requests.append(ctx)

        for event in script:
            if opts is not None and opts.signal is not None and opts.signal.is_set():
                yield StreamError(
                    message=AssistantMessage(
                        content=(),
                        model=model.id,
                        provider=self.name,
                        stop_reason="aborted",
                        error_message="Operation aborted",
                    ),
                    reason="aborted",
                )
                return
            yield event
