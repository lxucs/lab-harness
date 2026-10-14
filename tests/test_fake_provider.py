import pytest

from lab_harness.llm.events import StreamDone, StreamError, StreamEvent, StreamStart, TextDelta
from lab_harness.llm.fake import (
    FAKE_MODEL,
    FakeProvider,
    error_response,
    text_response,
    tool_call_response,
)
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


async def test_streams_text_in_deltas() -> None:
    provider = FakeProvider([text_response("abc", chunk_size=1)])
    events = await _drain(provider, _ctx())

    deltas = [e.delta for e in events if isinstance(e, TextDelta)]
    assert deltas == ["a", "b", "c"]


async def test_records_requests_for_assertions() -> None:
    provider = FakeProvider([text_response("one"), text_response("two")])
    await _drain(provider, _ctx("first"))
    await _drain(provider, _ctx("second"))

    assert [c.system_prompt for c in provider.requests] == ["first", "second"]


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
