import asyncio

import pytest

from lab_harness.llm.events import (
    StreamDone,
    StreamError,
    StreamEvent,
    StreamStart,
    TextDelta,
    ToolCallEnd,
)
from lab_harness.llm.fake import (
    FAKE_MODEL,
    FakeProvider,
    error_response,
    text_response,
    tool_call_response,
)
from lab_harness.llm.provider import StreamOptions
from lab_harness.llm.types import Context, TextBlock, ToolCallBlock


async def _drain(provider: FakeProvider, ctx: Context) -> list[StreamEvent]:
    return [event async for event in provider.stream(FAKE_MODEL, ctx)]


def _ctx(prompt: str = "sys") -> Context:
    return Context(system_prompt=prompt, messages=())


async def test_replays_scripted_text_response() -> None:
    provider = FakeProvider([text_response("hello")])
    events = await _drain(provider, _ctx())

    first = events[0]
    last = events[-1]
    assert isinstance(first, StreamStart)
    assert isinstance(last, StreamDone)
    first_content = last.message.content[0]
    assert isinstance(first_content, TextBlock)
    assert first_content.text == "hello"
    assert last.reason == "stop"


async def test_tool_call_response_sets_tool_use_stop_reason() -> None:
    call = ToolCallBlock(id="c1", name="read", arguments={"path": "a.py"})
    provider = FakeProvider([tool_call_response([call])])
    events = await _drain(provider, _ctx())

    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.reason == "tool_use"
    assert done.message.tool_calls == (call,)


async def test_error_response_terminates_with_stream_error() -> None:
    provider = FakeProvider([error_response("rate limited")])
    events = await _drain(provider, _ctx())

    last = events[-1]
    assert isinstance(last, StreamError)
    assert last.type == "error"
    assert last.message.stop_reason == "error"
    assert last.message.error_message == "rate limited"


async def test_raises_when_script_is_exhausted() -> None:
    """A loop making more requests than expected is a bug worth failing on."""
    provider = FakeProvider([text_response("only one")])
    await _drain(provider, _ctx())

    with pytest.raises(AssertionError, match="exhausted"):
        await _drain(provider, _ctx())


async def test_a_signal_mid_script_aborts_with_what_was_streamed() -> None:
    """As the adapters do: the words already sent are kept, a call is not.

    A call is dropped for the reason the adapters drop one — nothing will
    answer it, and a turn with an unanswered call cannot be sent again.
    """
    signal = asyncio.Event()
    call = ToolCallBlock(id="c1", name="read", arguments={"path": "a.py"})
    provider = FakeProvider([text_response("half an answer", chunk_size=4)])

    events: list[StreamEvent] = []
    async for event in provider.stream(FAKE_MODEL, _ctx(), StreamOptions(signal=signal)):
        events.append(event)
        if isinstance(event, TextDelta):
            signal.set()

    last = events[-1]
    assert isinstance(last, StreamError)
    assert last.reason == "aborted"
    assert last.message.stop_reason == "aborted"
    assert last.message.content == (TextBlock(text="half"),)

    # A call already announced is dropped, rather than left unanswered.
    signal = asyncio.Event()
    provider = FakeProvider([tool_call_response([call])])
    events = []
    async for event in provider.stream(FAKE_MODEL, _ctx(), StreamOptions(signal=signal)):
        events.append(event)
        if isinstance(event, ToolCallEnd):
            signal.set()

    last = events[-1]
    assert isinstance(last, StreamError)
    assert last.message.content == ()
