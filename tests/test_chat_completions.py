"""The Chat Completions adapter.

Everything here is offline. `httpx.MockTransport` carries canned bytes, so the
real request-building and the real SSE parsing run — only the socket is fake.
The canned frames are the shapes these APIs actually stream, including
DeepSeek's `reasoning_content` and the usage-only final chunk.

Two rules are load-bearing and get their own tests: the adapter **never
raises**, and tool results stay **separate** messages (the Anthropic adapter
merges them, and getting this backwards is a 400 from every endpoint).
"""

import asyncio
import json
from typing import Any

import pytest

from harness.config import OpenAIConfig
from harness.llm.events import (
    StreamDone,
    StreamError,
    StreamEvent,
    StreamStart,
    TextDelta,
    TextEnd,
    TextStart,
    ThinkingDelta,
    ThinkingEnd,
    ThinkingStart,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)
from harness.llm.provider import StreamOptions
from harness.llm.providers.chat_completions import ChatCompletionsProvider
from harness.llm.types import (
    AssistantMessage,
    Context,
    ImageBlock,
    Message,
    Model,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolDeclaration,
    ToolResultMessage,
    UserMessage,
    text,
)
from tests.http_fakes import (
    TEST_KEY,
    Exploding,
    Recorder,
    assert_blocks_open_before_deltas,
    final,
)

MODEL = Model(
    id="deepseek-chat",
    name="DeepSeek Chat",
    provider="deepseek",
    context_window=128_000,
    max_tokens=8_192,
)

REASONING_MODEL = Model(
    id="deepseek-reasoner",
    name="DeepSeek Reasoner",
    provider="deepseek",
    context_window=128_000,
    max_tokens=8_192,
    supports_reasoning=True,
)


def sse(*chunks: dict[str, Any], done: bool = True) -> str:
    frames = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    if done:
        frames.append("data: [DONE]\n\n")
    return "".join(frames)


def delta(**fields: Any) -> dict[str, Any]:
    """One streaming chunk carrying a delta."""
    return {"choices": [{"index": 0, "delta": fields, "finish_reason": None}]}


def finish(reason: str = "stop") -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}


def provider(
    recorder: Recorder | Exploding,
    config: OpenAIConfig | None = None,
    name: str = "deepseek",
) -> ChatCompletionsProvider:
    config = config or OpenAIConfig(api_key=TEST_KEY)
    return ChatCompletionsProvider(config, name, client=recorder.client())


def ctx(
    messages: tuple[Message, ...] = (),
    system_prompt: str = "",
    tools: tuple[ToolDeclaration, ...] = (),
) -> Context:
    return Context(system_prompt=system_prompt, messages=messages, tools=tools)


async def collect(
    prov: ChatCompletionsProvider,
    context: Context | None = None,
    opts: StreamOptions | None = None,
    model: Model = MODEL,
) -> list[StreamEvent]:
    return [event async for event in prov.stream(model, context or ctx(), opts)]


# --- the request ----------------------------------------------------------


async def test_posts_to_the_chat_completions_path() -> None:
    recorder = Recorder(sse(delta(content="hi"), finish()))

    await collect(provider(recorder, OpenAIConfig(api_key=TEST_KEY, base_url="https://x.test/v1")))

    assert recorder.request is not None
    assert str(recorder.request.url) == "https://x.test/v1/chat/completions"


async def test_a_trailing_slash_on_base_url_does_not_double_up() -> None:
    recorder = Recorder(sse(finish()))

    await collect(provider(recorder, OpenAIConfig(api_key=TEST_KEY, base_url="https://x.test/v1/")))

    assert recorder.request is not None
    assert str(recorder.request.url) == "https://x.test/v1/chat/completions"


async def test_the_key_rides_as_a_bearer_token() -> None:
    recorder = Recorder(sse(finish()))

    await collect(provider(recorder))

    assert recorder.request is not None
    assert recorder.request.headers["authorization"] == f"Bearer {TEST_KEY}"


async def test_the_environment_beats_a_configured_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same precedence as every other credential in the project."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")
    recorder = Recorder(sse(finish()))

    await collect(
        provider(recorder, OpenAIConfig(api_key=TEST_KEY, api_key_env="DEEPSEEK_API_KEY"))
    )

    assert recorder.request is not None
    assert recorder.request.headers["authorization"] == "Bearer from-env"


async def test_the_environment_can_override_the_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://gateway.test/v1")
    recorder = Recorder(sse(finish()))

    await collect(
        provider(
            recorder,
            OpenAIConfig(
                api_key=TEST_KEY, base_url="https://x.test/v1", base_url_env="DEEPSEEK_BASE_URL"
            ),
        )
    )

    assert recorder.request is not None
    assert str(recorder.request.url).startswith("https://gateway.test/v1")


async def test_the_request_streams() -> None:
    recorder = Recorder(sse(finish()))

    await collect(provider(recorder))

    assert recorder.payload["stream"] is True
    assert recorder.payload["model"] == "deepseek-chat"


async def test_usage_is_requested_by_default() -> None:
    """Without stream_options the final chunk carries no usage at all."""
    recorder = Recorder(sse(finish()))

    await collect(provider(recorder))

    assert recorder.payload["stream_options"] == {"include_usage": True}


async def test_usage_can_be_turned_off_for_servers_that_reject_it() -> None:
    recorder = Recorder(sse(finish()))

    await collect(provider(recorder, OpenAIConfig(api_key=TEST_KEY, include_usage=False)))

    assert "stream_options" not in recorder.payload


async def test_tools_are_declared_in_the_function_envelope() -> None:
    recorder = Recorder(sse(finish()))
    declaration = ToolDeclaration(
        name="read", description="Read a file", parameters={"type": "object", "properties": {}}
    )

    await collect(provider(recorder), ctx(tools=(declaration,)))

    assert recorder.payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


async def test_no_tools_means_no_tools_key() -> None:
    recorder = Recorder(sse(finish()))

    await collect(provider(recorder))

    assert "tools" not in recorder.payload


async def test_temperature_is_forwarded() -> None:
    recorder = Recorder(sse(finish()))

    await collect(provider(recorder), opts=StreamOptions(temperature=0.2))

    assert recorder.payload["temperature"] == 0.2


async def test_reasoning_effort_only_for_a_model_that_supports_it() -> None:
    recorder = Recorder(sse(finish()))

    await collect(provider(recorder), opts=StreamOptions(reasoning="high"), model=MODEL)
    assert "reasoning_effort" not in recorder.payload

    await collect(provider(recorder), opts=StreamOptions(reasoning="high"), model=REASONING_MODEL)
    assert recorder.payload["reasoning_effort"] == "high"


# --- message translation --------------------------------------------------


async def test_the_system_prompt_becomes_the_first_message() -> None:
    """Chat Completions has no top-level system field; Anthropic's does."""
    recorder = Recorder(sse(finish()))

    await collect(
        provider(recorder),
        ctx(messages=(UserMessage(content=text("hi")),), system_prompt="Be brief."),
    )

    assert recorder.payload["messages"][0] == {"role": "system", "content": "Be brief."}


async def test_no_system_prompt_means_no_system_message() -> None:
    recorder = Recorder(sse(finish()))

    await collect(provider(recorder), ctx(messages=(UserMessage(content=text("hi")),)))

    assert [m["role"] for m in recorder.payload["messages"]] == ["user"]


async def test_a_text_only_user_turn_is_a_plain_string() -> None:
    """Endpoints that reject the parts array still accept a string."""
    recorder = Recorder(sse(finish()))

    await collect(provider(recorder), ctx(messages=(UserMessage(content=text("hello")),)))

    assert recorder.payload["messages"][0] == {"role": "user", "content": "hello"}


async def test_an_image_turn_becomes_the_parts_array() -> None:
    recorder = Recorder(sse(finish()))
    message = UserMessage(
        content=(TextBlock(text="what is this"), ImageBlock(data="QUJD", mime_type="image/png"))
    )

    await collect(provider(recorder), ctx(messages=(message,)))

    assert recorder.payload["messages"][0]["content"] == [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
    ]


async def test_tool_results_stay_separate_messages() -> None:
    """The Anthropic adapter merges these; Chat Completions requires the opposite."""
    recorder = Recorder(sse(finish()))
    messages = (
        ToolResultMessage(tool_call_id="c1", tool_name="read", content=text("one")),
        ToolResultMessage(tool_call_id="c2", tool_name="ls", content=text("two")),
    )

    await collect(provider(recorder), ctx(messages=messages))

    assert recorder.payload["messages"] == [
        {"role": "tool", "tool_call_id": "c1", "content": "one"},
        {"role": "tool", "tool_call_id": "c2", "content": "two"},
    ]


async def test_assistant_tool_calls_serialize_arguments_as_a_json_string() -> None:
    """Sending the object rather than the string is a 400 on most endpoints."""
    recorder = Recorder(sse(finish()))
    message = AssistantMessage(
        content=(ToolCallBlock(id="c1", name="read", arguments={"path": "a.py"}),),
        model="deepseek-chat",
        provider="deepseek",
    )

    await collect(provider(recorder), ctx(messages=(message,)))

    call = recorder.payload["messages"][0]["tool_calls"][0]
    assert call == {
        "id": "c1",
        "type": "function",
        "function": {"name": "read", "arguments": '{"path": "a.py"}'},
    }


async def test_thinking_blocks_are_not_replayed() -> None:
    """Vendors reject reasoning sent back; DeepSeek documents it explicitly."""
    recorder = Recorder(sse(finish()))
    message = AssistantMessage(
        content=(ThinkingBlock(thinking="secret reasoning"), TextBlock(text="the answer")),
        model="deepseek-chat",
        provider="deepseek",
    )

    await collect(provider(recorder), ctx(messages=(message,)))

    assert recorder.payload["messages"][0]["content"] == "the answer"
    assert "secret reasoning" not in json.dumps(recorder.payload)


async def test_an_empty_assistant_turn_is_dropped() -> None:
    """An errored turn has no content; an empty message is rejected upstream."""
    recorder = Recorder(sse(finish()))
    message = AssistantMessage(
        content=(), model="deepseek-chat", provider="deepseek", stop_reason="error"
    )

    await collect(provider(recorder), ctx(messages=(message, UserMessage(content=text("retry")))))

    assert [m["role"] for m in recorder.payload["messages"]] == ["user"]


# --- response translation -------------------------------------------------


async def test_text_streams_as_start_deltas_and_end() -> None:
    recorder = Recorder(sse(delta(content="Hel"), delta(content="lo"), finish()))

    events = await collect(provider(recorder))

    assert isinstance(events[0], StreamStart)
    assert isinstance(events[1], TextStart)
    assert [e.delta for e in events if isinstance(e, TextDelta)] == ["Hel", "lo"]
    ends = [e for e in events if isinstance(e, TextEnd)]
    assert [e.text for e in ends] == ["Hello"]
    assert final(events).content == (TextBlock(text="Hello"),)


async def test_reasoning_content_becomes_thinking() -> None:
    """DeepSeek's field name for reasoning deltas."""
    recorder = Recorder(
        sse(
            delta(reasoning_content="step "),
            delta(reasoning_content="one"),
            delta(content="done"),
            finish(),
        )
    )

    events = await collect(provider(recorder))

    assert any(isinstance(e, ThinkingStart) for e in events)
    assert [e.delta for e in events if isinstance(e, ThinkingDelta)] == ["step ", "one"]
    assert [e.thinking for e in events if isinstance(e, ThinkingEnd)] == ["step one"]
    assert final(events).content == (ThinkingBlock(thinking="step one"), TextBlock(text="done"))


async def test_the_reasoning_alias_is_also_accepted() -> None:
    """Several gateways relabel reasoning_content to reasoning."""
    recorder = Recorder(sse(delta(reasoning="thought"), finish()))

    events = await collect(provider(recorder))

    assert [e.delta for e in events if isinstance(e, ThinkingDelta)] == ["thought"]


async def test_a_tool_call_accumulates_across_frames() -> None:
    recorder = Recorder(
        sse(
            delta(
                tool_calls=[{"index": 0, "id": "c1", "function": {"name": "read", "arguments": ""}}]
            ),
            delta(tool_calls=[{"index": 0, "function": {"arguments": '{"path"'}}]),
            delta(tool_calls=[{"index": 0, "function": {"arguments": ': "a.py"}'}}]),
            finish("tool_calls"),
        )
    )

    events = await collect(provider(recorder))

    assert any(isinstance(e, ToolCallStart) for e in events)
    assert [e.delta for e in events if isinstance(e, ToolCallDelta)] == ['{"path"', ': "a.py"}']
    ends = [e for e in events if isinstance(e, ToolCallEnd)]
    assert len(ends) == 1
    assert ends[0].tool_call == ToolCallBlock(id="c1", name="read", arguments={"path": "a.py"})
    assert final(events).stop_reason == "tool_use"


async def test_two_tool_calls_keep_separate_blocks() -> None:
    recorder = Recorder(
        sse(
            delta(
                tool_calls=[{"index": 0, "id": "c1", "function": {"name": "ls", "arguments": "{}"}}]
            ),
            delta(
                tool_calls=[
                    {"index": 1, "id": "c2", "function": {"name": "read", "arguments": "{}"}}
                ]
            ),
            finish("tool_calls"),
        )
    )

    events = await collect(provider(recorder))

    calls = final(events).tool_calls
    assert [(c.id, c.name) for c in calls] == [("c1", "ls"), ("c2", "read")]


async def test_a_missing_index_is_treated_as_the_first_call() -> None:
    """Some endpoints omit index when there is only one call."""
    recorder = Recorder(
        sse(
            delta(tool_calls=[{"id": "c1", "function": {"name": "ls", "arguments": "{}"}}]),
            finish("tool_calls"),
        )
    )

    events = await collect(provider(recorder))

    assert len(final(events).tool_calls) == 1


async def test_partial_tool_json_is_never_salvaged() -> None:
    """A truncated object can parse into args that validate while missing fields."""
    recorder = Recorder(
        sse(
            delta(
                tool_calls=[
                    {"index": 0, "id": "c1", "function": {"name": "read", "arguments": '{"pa'}}
                ]
            ),
            finish("length"),
        )
    )

    events = await collect(provider(recorder))

    assert final(events).tool_calls[0].arguments == {}
    assert final(events).stop_reason == "length"


async def test_blocks_get_distinct_indices() -> None:
    """The wire gives no index for text, so the adapter assigns them."""
    recorder = Recorder(
        sse(
            delta(reasoning_content="think"),
            delta(content="say"),
            delta(
                tool_calls=[{"index": 0, "id": "c1", "function": {"name": "ls", "arguments": "{}"}}]
            ),
            finish("tool_calls"),
        )
    )

    events = await collect(provider(recorder))

    starts = [e for e in events if isinstance(e, ThinkingStart | TextStart | ToolCallStart)]
    indices = [e.block_index for e in starts]
    assert len(indices) == len(set(indices)), f"block indices collided: {indices}"


@pytest.mark.parametrize(
    ("wire", "expected"),
    [
        ("stop", "stop"),
        ("length", "length"),
        ("tool_calls", "tool_use"),
        ("function_call", "tool_use"),
        ("content_filter", "stop"),
        ("something_new", "stop"),
    ],
)
async def test_finish_reason_mapping(wire: str, expected: str) -> None:
    recorder = Recorder(sse(delta(content="x"), finish(wire)))

    events = await collect(provider(recorder))

    assert final(events).stop_reason == expected


async def test_usage_is_read_from_the_final_chunk() -> None:
    """It arrives in a chunk whose `choices` is empty, after finish_reason."""
    recorder = Recorder(
        sse(
            delta(content="hi"),
            finish(),
            {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20}},
        )
    )

    events = await collect(provider(recorder))

    usage = final(events).usage
    assert (usage.input, usage.output) == (100, 20)


async def test_cached_tokens_are_not_counted_twice() -> None:
    """prompt_tokens includes the cached portion, so input excludes it."""
    recorder = Recorder(
        sse(
            finish(),
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 40},
                },
            },
        )
    )

    events = await collect(provider(recorder))

    usage = final(events).usage
    assert (usage.input, usage.cache_read) == (60, 40)
    assert usage.total == 105


async def test_the_message_records_model_and_provider_profile() -> None:
    """The profile name, so a transcript says which endpoint answered."""
    recorder = Recorder(sse(delta(content="hi"), finish()))

    events = await collect(provider(recorder, name="my-gateway"))

    assert final(events).provider == "my-gateway"
    assert final(events).model == "deepseek-chat"


async def test_a_stream_that_ends_without_done_still_closes_its_blocks() -> None:
    """Not every server sends the [DONE] frame."""
    recorder = Recorder(sse(delta(content="hi"), finish(), done=False))

    events = await collect(provider(recorder))

    assert [e.text for e in events if isinstance(e, TextEnd)] == ["hi"]
    assert isinstance(events[-1], StreamDone)


async def test_keepalives_and_comments_are_ignored() -> None:
    body = ": ping\n\n" + sse(delta(content="hi"), finish())
    recorder = Recorder(body)

    events = await collect(provider(recorder))

    assert final(events).content == (TextBlock(text="hi"),)


async def test_a_malformed_frame_does_not_lose_the_rest_of_the_stream() -> None:
    body = "data: {not json\n\n" + sse(delta(content="hi"), finish())
    recorder = Recorder(body)

    events = await collect(provider(recorder))

    assert final(events).content == (TextBlock(text="hi"),)
    assert isinstance(events[-1], StreamDone)


# --- failures: the adapter never raises -----------------------------------


async def test_a_missing_credential_is_a_stream_error_not_a_raise() -> None:
    recorder = Recorder(sse(finish()))

    events = await collect(
        provider(recorder, OpenAIConfig(api_key="", api_key_env="NOT_SET_ANYWHERE"))
    )

    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    message = events[0].message.error_message or ""
    # Both places a key can go, named for the user.
    assert "NOT_SET_ANYWHERE" in message
    assert "config.local.toml" in message


async def test_the_credential_error_names_the_profile() -> None:
    recorder = Recorder(sse(finish()))

    events = await collect(
        provider(recorder, OpenAIConfig(api_key="", api_key_env="NOPE"), name="my-gateway")
    )

    assert "my-gateway" in (final(events).error_message or "")


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500, 503])
async def test_http_errors_become_stream_errors(status: int) -> None:
    recorder = Recorder(
        json.dumps({"error": {"message": "upstream said no"}}),
        status=status,
    )

    events = await collect(provider(recorder))

    assert isinstance(events[-1], StreamError)
    message = final(events).error_message or ""
    assert str(status) in message
    assert "upstream said no" in message
    assert final(events).stop_reason == "error"


async def test_a_top_level_message_envelope_is_understood() -> None:
    """Several compatible servers do not nest under `error`."""
    recorder = Recorder(json.dumps({"message": "rate limited"}), status=429)

    events = await collect(provider(recorder))

    assert "rate limited" in (final(events).error_message or "")


async def test_an_html_error_body_degrades_to_something_readable() -> None:
    """A base_url pointing at the wrong thing returns HTML, not JSON."""
    recorder = Recorder("<html><body>502 Bad Gateway</body></html>", status=502)

    events = await collect(provider(recorder))

    message = final(events).error_message or ""
    assert "502" in message
    assert len(message) < 300, "an HTML page must not be pasted into the transcript whole"


async def test_a_transport_failure_becomes_a_stream_error() -> None:
    events = await collect(provider(Exploding()))

    assert isinstance(events[-1], StreamError)
    assert "ConnectError" in (final(events).error_message or "")


async def test_an_abort_before_the_request_stops_immediately() -> None:
    signal = asyncio.Event()
    signal.set()
    recorder = Recorder(sse(delta(content="never"), finish()))

    events = await collect(provider(recorder), opts=StreamOptions(signal=signal))

    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert events[0].reason == "aborted"
    assert recorder.request is None, "no request should have been sent"


async def test_an_abort_mid_stream_terminates_the_turn() -> None:
    signal = asyncio.Event()
    recorder = Recorder(sse(delta(content="a"), delta(content="b"), delta(content="c"), finish()))
    stream = provider(recorder).stream(MODEL, ctx(), StreamOptions(signal=signal))

    events: list[StreamEvent] = []
    async for event in stream:
        events.append(event)
        if isinstance(event, TextDelta):
            # Fire the signal the way Ctrl-C does: part way through.
            signal.set()

    assert isinstance(events[-1], StreamError)
    assert events[-1].reason == "aborted"
    assert final(events).stop_reason == "aborted"


async def test_a_request_building_failure_is_reported_not_raised() -> None:
    """Arguments that cannot be serialized are a bug, but not a fatal one."""

    class Unserializable:
        pass

    recorder = Recorder(sse(finish()))
    message = AssistantMessage(
        content=(ToolCallBlock(id="c1", name="x", arguments={"bad": Unserializable()}),),
        model="deepseek-chat",
        provider="deepseek",
    )

    events = await collect(provider(recorder), ctx(messages=(message,)))

    assert isinstance(events[0], StreamError)
    assert "Could not build request" in (final(events).error_message or "")


async def test_every_block_is_opened_before_its_deltas() -> None:
    """docs/events.md: updates never appear before their start.

    This protocol gives no block boundaries at all, so the adapter invents
    them — which makes the ordering its responsibility rather than the
    server's.
    """
    recorder = Recorder(
        sse(
            delta(reasoning_content="think"),
            delta(content="say"),
            delta(
                tool_calls=[{"index": 0, "id": "c1", "function": {"name": "ls", "arguments": "{}"}}]
            ),
            finish("tool_calls"),
        )
    )

    events = await collect(provider(recorder))

    assert_blocks_open_before_deltas(events)
