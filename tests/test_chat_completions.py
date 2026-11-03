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
import logging
from dataclasses import replace
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from lab_harness.agent import Agent
from lab_harness.config import Config, OpenAIConfig, profile_defaults
from lab_harness.llm.events import (
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
from lab_harness.llm.provider import StreamOptions
from lab_harness.llm.providers import _http
from lab_harness.llm.providers.chat_completions import ChatCompletionsProvider
from lab_harness.llm.registry import resolve_model
from lab_harness.llm.types import (
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
from lab_harness.tools.base import ToolContext, ToolResult, ok, tool
from tests.entries import profile_entry, retrying_profile
from tests.http_fakes import (
    TEST_KEY,
    Exploding,
    Recorder,
    Scripted,
    assert_blocks_open_before_deltas,
    final,
)

#: Local fixtures, not catalog lookups: MODEL is the non-reasoning case on
#: purpose, whatever the catalog says about the id.
MODEL = Model(
    id="deepseek-flash",
    name="DeepSeek Flash",
    provider="deepseek",
    context_window=128_000,
    max_tokens=8_192,
)

REASONING_MODEL = Model(
    id="deepseek-v4-pro",
    name="DeepSeek V4 Pro",
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
    config = config or profile_entry(api_key=TEST_KEY)
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

    await collect(provider(recorder, profile_entry(api_key=TEST_KEY, base_url="https://x.test/v1")))

    assert recorder.request is not None
    assert str(recorder.request.url) == "https://x.test/v1/chat/completions"


async def test_a_trailing_slash_on_base_url_does_not_double_up() -> None:
    recorder = Recorder(sse(finish()))

    await collect(
        provider(recorder, profile_entry(api_key=TEST_KEY, base_url="https://x.test/v1/"))
    )

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
        provider(recorder, profile_entry(api_key=TEST_KEY, api_key_env="DEEPSEEK_API_KEY"))
    )

    assert recorder.request is not None
    assert recorder.request.headers["authorization"] == "Bearer from-env"


async def test_the_environment_can_override_the_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://gateway.test/v1")
    recorder = Recorder(sse(finish()))

    await collect(
        provider(
            recorder,
            profile_entry(
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
    assert recorder.payload["model"] == "deepseek-flash"


async def test_usage_is_requested_by_default() -> None:
    """Without stream_options the final chunk carries no usage at all."""
    recorder = Recorder(sse(finish()))

    await collect(provider(recorder))

    assert recorder.payload["stream_options"] == {"include_usage": True}


async def test_usage_can_be_turned_off_for_servers_that_reject_it() -> None:
    recorder = Recorder(sse(finish()))

    await collect(provider(recorder, profile_entry(api_key=TEST_KEY, include_usage=False)))

    assert "stream_options" not in recorder.payload


async def test_preserved_thinking_is_not_requested_by_default() -> None:
    """The `thinking` object is GLM's shape; another vendor reads it differently."""
    recorder = Recorder(sse(finish()))

    await collect(provider(recorder))

    assert "thinking" not in recorder.payload


async def test_preserved_thinking_sends_glms_documented_object() -> None:
    """Exactly Z.ai's example; `clear_thinking: false` is what keeps it."""
    recorder = Recorder(sse(finish()))

    await collect(provider(recorder, profile_entry(api_key=TEST_KEY, preserve_thinking=True)))

    assert recorder.payload["thinking"] == {"type": "enabled", "clear_thinking": False}


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


async def test_the_models_effort_is_sent_unless_a_request_overrides_it() -> None:
    recorder = Recorder(sse(finish()))
    model = replace(REASONING_MODEL, reasoning_effort="low")

    await collect(provider(recorder), model=model)
    assert recorder.payload["reasoning_effort"] == "low"

    await collect(provider(recorder), opts=StreamOptions(reasoning=""), model=model)
    assert "reasoning_effort" not in recorder.payload, "an empty override sends none"


# --- message translation --------------------------------------------------


async def test_the_system_prompt_becomes_the_first_message() -> None:
    """Chat Completions has no top-level system field; Anthropic's does."""
    recorder = Recorder(sse(finish()))

    await collect(
        provider(recorder),
        ctx(messages=(UserMessage(content=text("hi")),), system_prompt="Be brief."),
    )

    assert recorder.payload["messages"][0] == {"role": "system", "content": "Be brief."}


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
        model="deepseek-flash",
        provider="deepseek",
    )

    await collect(provider(recorder), ctx(messages=(message,)))

    call = recorder.payload["messages"][0]["tool_calls"][0]
    assert call == {
        "id": "c1",
        "type": "function",
        "function": {"name": "read", "arguments": '{"path": "a.py"}'},
    }


async def test_thinking_is_not_replayed_by_default() -> None:
    """A server never told to expect reasoning back may reject the unknown field."""
    recorder = Recorder(sse(finish()))
    message = AssistantMessage(
        content=(ThinkingBlock(thinking="secret reasoning"), TextBlock(text="the answer")),
        model="deepseek-flash",
        provider="deepseek",
    )

    await collect(provider(recorder), ctx(messages=(message,)))

    assert recorder.payload["messages"][0] == {"role": "assistant", "content": "the answer"}
    assert "secret reasoning" not in json.dumps(recorder.payload)


# --- reasoning replay -----------------------------------------------------
#
# DeepSeek returns a 400 for a request carrying `tools` unless every earlier
# turn's `reasoning_content` comes back; GLM asks for it unmodified. Request
# shapes follow the two vendors' published examples, not captured traffic.

REPLAYING = profile_entry(api_key=TEST_KEY, replay_reasoning_field="reasoning_content")


def thinking_turn(
    thinking: str, *blocks: TextBlock | ToolCallBlock, signature: str | None = None
) -> AssistantMessage:
    return AssistantMessage(
        content=(ThinkingBlock(thinking=thinking, signature=signature), *blocks),
        model="deepseek-flash",
        provider="deepseek",
    )


async def test_a_replaying_profile_sends_reasoning_back() -> None:
    recorder = Recorder(sse(finish()))
    message = thinking_turn("secret reasoning", TextBlock(text="the answer"))

    await collect(provider(recorder, REPLAYING), ctx(messages=(message,)))

    assert recorder.payload["messages"][0] == {
        "role": "assistant",
        "content": "the answer",
        "reasoning_content": "secret reasoning",
    }


async def test_replayed_reasoning_rides_beside_tool_calls() -> None:
    """The tool loop is where DeepSeek enforces it."""
    recorder = Recorder(sse(finish()))
    call = ToolCallBlock(id="c1", name="ls", arguments={})
    message = thinking_turn("list first", call)

    await collect(provider(recorder, REPLAYING), ctx(messages=(message,)))

    entry = recorder.payload["messages"][0]
    assert entry["content"] is None
    assert entry["reasoning_content"] == "list first"
    assert [c["id"] for c in entry["tool_calls"]] == ["c1"]


async def test_every_earlier_turn_carries_its_reasoning() -> None:
    """DeepSeek wants all previous turns, not only the latest tool call's."""
    recorder = Recorder(sse(finish()))
    messages = (
        UserMessage(content=text("q1")),
        thinking_turn("first thought", TextBlock(text="a1")),
        UserMessage(content=text("q2")),
        thinking_turn("second thought", TextBlock(text="a2")),
    )

    await collect(provider(recorder, REPLAYING), ctx(messages=messages))

    replayed = [m.get("reasoning_content") for m in recorder.payload["messages"]]
    assert replayed == [None, "first thought", None, "second thought"]


async def test_the_replay_field_is_the_configured_one() -> None:
    """`reasoning`, not the default `reasoning_content`, so the setting is what decides."""
    field = "reasoning"
    recorder = Recorder(sse(finish()))
    config = profile_entry(api_key=TEST_KEY, replay_reasoning_field=field)

    await collect(
        provider(recorder, config), ctx(messages=(thinking_turn("t", TextBlock(text="a")),))
    )

    assert recorder.payload["messages"][0][field] == "t"


async def test_signed_reasoning_from_another_provider_is_not_replayed() -> None:
    """A signature means Anthropic or Responses produced it, for their own replay."""
    recorder = Recorder(sse(finish()))
    message = thinking_turn("anthropic thinking", TextBlock(text="a"), signature="sig")

    await collect(provider(recorder, REPLAYING), ctx(messages=(message,)))

    assert "reasoning_content" not in recorder.payload["messages"][0]


async def test_a_turn_with_only_reasoning_is_still_dropped() -> None:
    """No text and no calls is an interrupted turn; an empty reply is rejected."""
    recorder = Recorder(sse(finish()))
    messages = (thinking_turn("cut off mid-thought"), UserMessage(content=text("retry")))

    await collect(provider(recorder, REPLAYING), ctx(messages=messages))

    assert [m["role"] for m in recorder.payload["messages"]] == ["user"]


async def test_streamed_reasoning_goes_back_byte_for_byte() -> None:
    """The server cached the reasoning as it generated it; any edit is a cache miss."""
    pieces = ["Step 1:\n", "  check → ", "a.py", " ", "été\n"]
    first = Recorder(
        sse(
            *(delta(reasoning_content=piece) for piece in pieces),
            delta(
                tool_calls=[{"index": 0, "id": "c1", "function": {"name": "ls", "arguments": "{}"}}]
            ),
            finish("tool_calls"),
        )
    )
    reply = final(await collect(provider(first, REPLAYING)))

    second = Recorder(sse(finish()))
    await collect(provider(second, REPLAYING), ctx(messages=(reply,)))

    assert second.payload["messages"][0]["reasoning_content"] == "".join(pieces)


# --- what was sent, rebuilt -------------------------------------------------


async def test_request_body_is_the_json_stream_posts() -> None:
    recorder = Recorder(sse(finish()))
    adapter = provider(recorder, REPLAYING)
    tools = (ToolDeclaration(name="ls", description="List.", parameters={"type": "object"}),)
    request = ctx(
        messages=(
            UserMessage(content=text("q1")),
            thinking_turn("t", TextBlock(text="a1")),
            UserMessage(content=text("q2")),
        ),
        system_prompt="Be brief.",
        tools=tools,
    )

    await collect(adapter, request)

    assert adapter.request_body(MODEL, request) == recorder.payload


async def test_reply_entries_are_what_the_next_request_carries() -> None:
    adapter = provider(Recorder(), REPLAYING)
    call = ToolCallBlock(id="c1", name="ls", arguments={"path": "."})
    reply = thinking_turn("list first", TextBlock(text="Looking."), call)

    sent = adapter.request_body(MODEL, ctx(messages=(UserMessage(content=text("q")), reply)))

    assert adapter.reply_entries(reply) == sent["messages"][1:]


async def test_a_reply_keeps_its_reasoning_when_the_profile_replays_none() -> None:
    """What the model generated, not what goes back: training wants the thought."""
    adapter = provider(Recorder())  # replay_reasoning_field is empty
    reply = thinking_turn("the thought", TextBlock(text="the answer"))

    (entry,) = adapter.reply_entries(reply)

    assert entry["reasoning_content"] == "the thought"
    sent = adapter.request_body(MODEL, ctx(messages=(reply,)))
    assert "reasoning_content" not in sent["messages"][0]


def test_an_empty_reply_has_no_entries() -> None:
    message = AssistantMessage(
        content=(), model="deepseek-flash", provider="deepseek", stop_reason="error"
    )

    assert provider(Recorder()).reply_entries(message) == []


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


async def test_a_frame_that_ends_the_reasoning_and_starts_the_answer_keeps_both() -> None:
    """The handover can share a frame. A live gateway sends the idle field as ""
    on every frame, as these do."""
    recorder = Recorder(
        sse(
            delta(reasoning_content="Let me give the", content=""),
            delta(reasoning_content=" answer.", content="Seven"),
            delta(reasoning_content="", content=" entries."),
            finish(),
        )
    )

    events = await collect(provider(recorder))

    assert final(events).content == (
        ThinkingBlock(thinking="Let me give the answer."),
        TextBlock(text="Seven entries."),
    )


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


@pytest.mark.parametrize(
    ("wire", "expected"),
    [
        ("stop", "stop"),
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


async def test_deepseek_usage_reads_its_cache_hits() -> None:
    """DeepSeek reports hits twice: its own field, and the OpenAI-style one.

    The shape is the final chunk from DeepSeek's API reference, not captured
    traffic, with the counts changed so a hit is actually exercised.
    """
    recorder = Recorder(
        sse(
            finish(),
            {
                "choices": [],
                "usage": {
                    "completion_tokens": 9,
                    "prompt_tokens": 17,
                    "total_tokens": 26,
                    "prompt_tokens_details": {"cached_tokens": 12},
                    "prompt_cache_hit_tokens": 12,
                    "prompt_cache_miss_tokens": 5,
                },
            },
        )
    )

    events = await collect(provider(recorder))

    usage = final(events).usage
    assert (usage.input, usage.cache_read, usage.output) == (5, 12, 9)


async def test_the_message_records_model_and_provider_profile() -> None:
    """The profile name, so a transcript says which endpoint answered."""
    recorder = Recorder(sse(delta(content="hi"), finish()))

    events = await collect(provider(recorder, name="my-gateway"))

    assert final(events).provider == "my-gateway"
    assert final(events).model == "deepseek-flash"


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
        provider(recorder, profile_entry(api_key="", api_key_env="NOT_SET_ANYWHERE"))
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
        provider(recorder, profile_entry(api_key="", api_key_env="NOPE"), name="my-gateway")
    )

    assert "my-gateway" in (final(events).error_message or "")


@pytest.mark.parametrize("status", [400, 500])
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


async def test_a_200_that_is_not_a_stream_is_an_error() -> None:
    """A gateway can refuse with a 200 — this body is one rejecting a model id.

    Read as a stream, it holds no frames, so the turn would end as a clean
    `stop` with nothing said: no answer, and no error to say why.
    """
    recorder = Recorder(json.dumps({"code": 1005, "msg": "invalid model"}))

    events = await collect(provider(recorder))

    assert isinstance(events[-1], StreamError)
    assert final(events).stop_reason == "error"
    assert "invalid model" in (final(events).error_message or "")


async def test_an_empty_200_is_an_error() -> None:
    events = await collect(provider(Recorder("")))

    assert isinstance(events[-1], StreamError)
    assert "200" in (final(events).error_message or "")


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


async def test_an_abort_keeps_what_the_stream_had_produced() -> None:
    """A cancelled answer is not an empty one: what was said stays in the turn."""
    signal = asyncio.Event()
    recorder = Recorder(sse(delta(content="a"), delta(content="b"), finish()))
    stream = provider(recorder).stream(MODEL, ctx(), StreamOptions(signal=signal))

    events: list[StreamEvent] = []
    async for event in stream:
        events.append(event)
        if isinstance(event, TextDelta):
            signal.set()

    message = final(events)
    assert message.stop_reason == "aborted"
    assert message.content == (TextBlock(text="a"),)


async def test_a_request_building_failure_is_reported_not_raised() -> None:
    """Arguments that cannot be serialized are a bug, but not a fatal one."""

    class Unserializable:
        pass

    recorder = Recorder(sse(finish()))
    message = AssistantMessage(
        content=(ToolCallBlock(id="c1", name="x", arguments={"bad": Unserializable()}),),
        model="deepseek-flash",
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


# --- through the loop -----------------------------------------------------


class _Scripted:
    """Serves one canned SSE body per request, recording every payload."""

    def __init__(self, *bodies: str) -> None:
        self._bodies = list(bodies)
        self.payloads: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.payloads.append(json.loads(request.content))
        return httpx.Response(200, text=self._bodies.pop(0))

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


class _NoParams(BaseModel):
    pass


@tool(name="ls", description="List nothing.")
async def _ls(p: _NoParams, ctx: ToolContext) -> ToolResult:
    return ok("a.py")


async def test_the_shipped_deepseek_profile_survives_a_tool_loop() -> None:
    """The case DeepSeek refuses: request two of a loop, reasoning missing.

    Real agent, real loop, the shipped profile — only the socket is canned.
    """
    transport = _Scripted(
        sse(
            delta(reasoning_content="I should look first."),
            delta(
                tool_calls=[{"index": 0, "id": "c1", "function": {"name": "ls", "arguments": "{}"}}]
            ),
            finish("tool_calls"),
        ),
        sse(delta(content="There is one file."), finish()),
    )
    profile = replace(Config().providers.openai["deepseek"], api_key=TEST_KEY)
    agent = Agent(
        model=resolve_model("deepseek-flash"),
        provider=ChatCompletionsProvider(profile, "deepseek", client=transport.client()),
        tools=(_ls,),
    )

    await agent.prompt("what is here?")

    assert len(transport.payloads) == 2
    assert "tools" in transport.payloads[1]
    turn = next(m for m in transport.payloads[1]["messages"] if m["role"] == "assistant")
    assert turn["reasoning_content"] == "I should look first."
    assert turn["tool_calls"][0]["id"] == "c1"


# --- the whole reply at once ----------------------------------------------


def completion(
    message: dict[str, Any],
    *,
    finish_reason: str = "stop",
    usage: dict[str, Any] | None = None,
) -> str:
    """A non-streamed reply, in the shape these endpoints actually return.

    `object`, `created` and the per-choice `logprobs` are part of the real
    payload and are ignored by the adapter. They stay here so the fixture is
    the document rather than a trimmed version of it — and so a reply carrying
    keys the adapter does not read, as a training server's token ids do, is
    what the tests run against.
    """
    body: dict[str, Any] = {
        "id": "chatcmpl-7",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "deepseek-flash",
        "choices": [
            {"index": 0, "message": message, "logprobs": None, "finish_reason": finish_reason}
        ],
    }
    if usage is not None:
        body["usage"] = usage
    return json.dumps(body)


def unstreamed(**settings: Any) -> OpenAIConfig:
    """A profile that asks for the whole reply in one response."""
    return profile_entry(api_key=TEST_KEY, stream=False, **settings)


class _Slow:
    """A transport that does not answer until it is released."""

    def __init__(self, body: str) -> None:
        self._body = body
        self.arrived = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.arrived.set()
        await self.release.wait()
        return httpx.Response(200, text=self._body)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


async def test_a_profile_can_ask_for_the_whole_reply_at_once() -> None:
    recorder = Recorder(completion({"role": "assistant", "content": "hi"}))

    events = await collect(provider(recorder, unstreamed()))

    assert recorder.payload["stream"] is False
    # Only valid on a streaming request; a server may refuse it on this one.
    assert "stream_options" not in recorder.payload
    assert final(events).content == (TextBlock(text="hi"),)


async def test_a_non_streamed_reply_arrives_as_one_delta_per_block() -> None:
    """The whole block in a single delta. The only difference a consumer sees."""
    recorder = Recorder(
        completion({"role": "assistant", "reasoning_content": "thinking", "content": "answer"})
    )

    events = await collect(provider(recorder, unstreamed()))

    assert_blocks_open_before_deltas(events)
    assert [e for e in events if isinstance(e, ThinkingDelta)] == [
        ThinkingDelta(block_index=0, delta="thinking")
    ]
    assert [e for e in events if isinstance(e, TextDelta)] == [
        TextDelta(block_index=1, delta="answer")
    ]
    assert final(events).content == (
        ThinkingBlock(thinking="thinking"),
        TextBlock(text="answer"),
    )


async def test_a_non_streamed_reply_ends_every_block_it_opened() -> None:
    """The end events are the stream's, not the wire's — nothing sends them."""
    recorder = Recorder(completion({"role": "assistant", "content": "answer"}))

    events = await collect(provider(recorder, unstreamed()))

    assert [type(e) for e in events] == [
        StreamStart,
        TextStart,
        TextDelta,
        TextEnd,
        StreamDone,
    ]


async def test_a_non_streamed_reply_keeps_parallel_tool_calls_apart() -> None:
    """A reply indexes none of its calls, and a missing index reads as 0.

    So without indices assigned on the way through, both calls accumulate into
    one block: a single call with the two names concatenated and the arguments
    run together. The regression this guards is silent and destructive.
    """
    recorder = Recorder(
        completion(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path": "a.py"}'},
                    },
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "ls", "arguments": "{}"},
                    },
                ],
            },
            finish_reason="tool_calls",
        )
    )

    message = final(await collect(provider(recorder, unstreamed())))

    assert message.stop_reason == "tool_use"
    assert message.tool_calls == (
        ToolCallBlock(id="c1", name="read", arguments={"path": "a.py"}),
        ToolCallBlock(id="c2", name="ls", arguments={}),
    )


async def test_a_non_streamed_reply_reports_its_usage() -> None:
    recorder = Recorder(
        completion(
            {"role": "assistant", "content": "hi"},
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 7,
                "total_tokens": 107,
                "prompt_tokens_details": {"cached_tokens": 60},
            },
        )
    )

    usage = final(await collect(provider(recorder, unstreamed()))).usage

    # prompt_tokens counts the cached portion, so subtracting keeps the total
    # from charging those tokens twice.
    assert (usage.input, usage.cache_read, usage.output) == (40, 60, 7)


async def test_a_non_streamed_error_status_becomes_an_error_turn() -> None:
    recorder = Recorder(json.dumps({"error": {"message": "model not found"}}), status=404)

    message = final(await collect(provider(recorder, unstreamed())))

    assert message.stop_reason == "error"
    assert message.error_message is not None
    assert "model not found" in message.error_message


async def test_a_non_streamed_body_that_is_not_json_becomes_an_error_turn() -> None:
    """A misconfigured base_url answers with a page, not a reply."""
    recorder = Recorder("<html>upstream error</html>")

    message = final(await collect(provider(recorder, unstreamed())))

    assert message.stop_reason == "error"
    assert message.error_message is not None
    assert "expected a reply" in message.error_message


async def test_a_gateway_refusing_with_a_200_becomes_an_error_turn() -> None:
    """Valid JSON, no `choices`: an answer-shaped nothing, which would
    otherwise end the turn as a clean stop carrying no answer and no error."""
    recorder = Recorder(json.dumps({"code": 1005, "msg": "invalid model"}))

    message = final(await collect(provider(recorder, unstreamed())))

    assert message.stop_reason == "error"
    assert message.error_message is not None
    assert "invalid model" in message.error_message


async def test_an_abort_mid_request_does_not_wait_for_the_whole_reply() -> None:
    """Nothing arrives until the reply is complete, so the signal is raced.

    The transport is never released: were the signal only read before the
    request, this would hang instead of ending the turn.
    """
    transport = _Slow(completion({"role": "assistant", "content": "too late"}))
    signal = asyncio.Event()
    prov = ChatCompletionsProvider(unstreamed(), "deepseek", client=transport.client())

    async def run() -> list[StreamEvent]:
        return [event async for event in prov.stream(MODEL, ctx(), StreamOptions(signal=signal))]

    task = asyncio.create_task(run())
    await asyncio.wait_for(transport.arrived.wait(), timeout=5)
    signal.set()
    events = await asyncio.wait_for(task, timeout=5)

    message = final(events)
    assert message.stop_reason == "aborted"
    # Nothing had arrived, so there is nothing to report alongside the stop.
    assert message.content == ()


async def test_the_loop_runs_a_tool_call_from_a_non_streamed_reply() -> None:
    """The loop is told nothing about any of this: same events, same run."""
    transport = Scripted(
        completion(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}
                ],
            },
            finish_reason="tool_calls",
        ),
        completion({"role": "assistant", "content": "There is one file."}),
    )
    agent = Agent(
        model=MODEL,
        provider=ChatCompletionsProvider(unstreamed(), "deepseek", client=transport.client()),
        tools=(_ls,),
    )

    produced = await agent.prompt("what is here?")

    assert [message.role for message in produced] == ["assistant", "tool_result", "assistant"]
    assert len(transport.payloads) == 2
    assert transport.payloads[1]["stream"] is False
    # The tool result went back under the call that asked for it.
    assert transport.payloads[1]["messages"][-1]["tool_call_id"] == "c1"


# --- retries --------------------------------------------------------------


class _Failing:
    """Fails with a given status the first `times`, then serves the body."""

    def __init__(self, body: str, *, status: int, times: int, retry_after: str = "") -> None:
        self._body = body
        self._status = status
        self._times = times
        self._retry_after = retry_after
        self.attempts = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.attempts += 1
        if self.attempts > self._times:
            return httpx.Response(200, text=self._body)
        headers = {"retry-after": self._retry_after} if self._retry_after else {}
        return httpx.Response(self._status, text="slow down", headers=headers)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


def test_retry_defaults() -> None:
    """The shipped numbers, which every other test turns off. See tests/entries.py."""
    shipped = profile_defaults()

    assert shipped.max_retries == 5
    assert shipped.retry_initial_seconds == 1.0
    assert shipped.retry_max_seconds == 60.0


async def test_a_rate_limited_request_is_sent_again() -> None:
    transport = _Failing(sse(delta(content="hi"), finish()), status=429, times=2)
    prov = ChatCompletionsProvider(
        retrying_profile(api_key=TEST_KEY), "deepseek", client=transport.client()
    )

    events = await collect(prov)

    assert transport.attempts == 3
    assert final(events).content == (TextBlock(text="hi"),)


async def test_the_reply_records_how_many_times_it_was_sent_again() -> None:
    """A turn is one message however many attempts it took, so the count on it
    is the only record that the endpoint was being flaky."""
    transport = _Failing(sse(delta(content="hi"), finish()), status=429, times=2)
    prov = ChatCompletionsProvider(
        retrying_profile(api_key=TEST_KEY), "deepseek", client=transport.client()
    )

    message = final(await collect(prov))

    assert message.retries == 2
    assert message.stop_reason == "stop"


async def test_a_first_time_reply_records_no_retries() -> None:
    message = final(await collect(provider(Recorder(sse(delta(content="hi"), finish())))))

    assert message.retries == 0


async def test_a_turn_that_gave_up_records_every_attempt_it_made() -> None:
    transport = _Failing(sse(finish()), status=503, times=99)
    prov = ChatCompletionsProvider(
        retrying_profile(api_key=TEST_KEY), "deepseek", client=transport.client()
    )

    message = final(await collect(prov))

    assert (message.stop_reason, message.retries) == ("error", 5)


async def test_retries_stop_at_the_configured_count() -> None:
    """Six attempts for five retries, then the error turn the contract wants."""
    transport = _Failing(sse(finish()), status=503, times=99)
    prov = ChatCompletionsProvider(
        retrying_profile(api_key=TEST_KEY), "deepseek", client=transport.client()
    )

    message = final(await collect(prov))

    assert transport.attempts == 6
    assert message.stop_reason == "error"


async def test_a_request_the_server_refused_is_not_sent_again() -> None:
    """A 400 is the request's own fault: sending it again only delays the error."""
    transport = _Failing(sse(finish()), status=400, times=99)
    prov = ChatCompletionsProvider(
        retrying_profile(api_key=TEST_KEY), "deepseek", client=transport.client()
    )

    message = final(await collect(prov))

    assert transport.attempts == 1
    assert message.stop_reason == "error"


async def test_a_dropped_connection_is_sent_again() -> None:
    """A transport failure says nothing about whether the request was good."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, text=sse(delta(content="hi"), finish()))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = ChatCompletionsProvider(retrying_profile(api_key=TEST_KEY), "deepseek", client=client)

    events = await collect(prov)

    assert attempts == 3
    assert final(events).content == (TextBlock(text="hi"),)


async def test_a_stream_that_fails_part_way_through_is_not_sent_again() -> None:
    """The consumer holds the first half; a retry would hand it the whole reply.

    The body ends mid-frame after a delta the consumer has already seen, so the
    turn fails rather than repeating text that is on screen.
    """
    sent = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent
        sent += 1
        return httpx.Response(200, text=f"data: {json.dumps(delta(content='half'))}\n\ndata: {{")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = ChatCompletionsProvider(retrying_profile(api_key=TEST_KEY), "deepseek", client=client)

    events = await collect(prov)

    assert sent == 1
    # The delta reached the consumer, so the stream had opened.
    assert any(isinstance(event, TextDelta) for event in events)


async def test_a_retry_after_the_server_asked_for_wins_under_the_cap() -> None:
    """`backoff` is the policy, so the schedule is checked directly.

    Timing assertions on a real sleep would be flaky; these are the decisions
    that matter, and the adapter tests above cover the wiring.
    """
    config = retrying_profile(api_key=TEST_KEY, retry_initial_seconds=1.0, retry_max_seconds=60.0)
    too_many = httpx.Response(429, headers={"retry-after": "45"})
    unhurried = httpx.Response(503)

    # Doubling, from retry_initial_seconds.
    assert [_http.backoff(config, n, unhurried) for n in range(4)] == [1.0, 2.0, 4.0, 8.0]
    # retry_max_seconds caps a long doubling and a long Retry-After alike.
    assert _http.backoff(replace(config, max_retries=20), 9, unhurried) == 60.0
    assert _http.backoff(replace(config, retry_max_seconds=5.0), 0, too_many) == 5.0
    # The server's own number beats ours when it is longer.
    assert _http.backoff(config, 0, too_many) == 45.0
    # An unparseable header leaves the doubling in charge.
    assert _http.backoff(config, 0, httpx.Response(429, headers={"retry-after": "Wed, 21 Oct"}))
    # None means a transport failure, which is always worth another attempt.
    assert _http.backoff(config, 0, None) == 1.0
    # Past the count, and a status that will not improve: no retry at all.
    assert _http.backoff(config, config.max_retries, unhurried) is None
    assert _http.backoff(config, 0, httpx.Response(400)) is None


async def test_an_abort_during_a_backoff_ends_the_turn() -> None:
    """A run stopped mid-backoff should not sit out the rest of the doubling.

    The backoff here is 30 seconds and the signal fires once the first attempt
    has failed. Slept through rather than raced, this test would take that long.
    """
    arrived = asyncio.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        arrived.set()
        return httpx.Response(503, text="unavailable")

    signal = asyncio.Event()
    config = retrying_profile(api_key=TEST_KEY, retry_initial_seconds=30.0, retry_max_seconds=30.0)
    prov = ChatCompletionsProvider(
        config, "deepseek", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )

    async def stop_once_it_fails() -> None:
        await arrived.wait()
        signal.set()

    waker = asyncio.create_task(stop_once_it_fails())
    events = await asyncio.wait_for(collect(prov, opts=StreamOptions(signal=signal)), timeout=5)
    await waker

    assert final(events).stop_reason == "aborted"


async def test_a_non_streamed_request_is_retried_too() -> None:
    body = completion({"role": "assistant", "content": "hi"})
    transport = _Failing(body, status=500, times=2)
    config = retrying_profile(api_key=TEST_KEY, stream=False)
    prov = ChatCompletionsProvider(config, "deepseek", client=transport.client())

    message = final(await collect(prov))

    assert transport.attempts == 3
    assert message.content == (TextBlock(text="hi"),)
    assert message.retries == 2


# --- what is logged -------------------------------------------------------


async def test_the_log_names_the_response_and_its_finish_reason_as_sent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """content_filter maps to "stop", so the log is where the real reason survives."""
    first = {"id": "chatcmpl-42", "model": "deepseek-v4-flash", **delta(content="hi")}
    body = sse(first, {"id": "chatcmpl-42", **finish("content_filter")})

    with caplog.at_level(logging.DEBUG, logger="lab_harness.llm.providers.chat_completions"):
        await collect(provider(Recorder(body)))

    assert "Response chatcmpl-42 from deepseek-v4-flash" in caplog.text
    # Every chunk repeats the id; the line is written once.
    assert caplog.text.count("chatcmpl-42") == 1
    assert "finish_reason=content_filter" in caplog.text


async def test_the_log_counts_the_turns_whose_reasoning_goes_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """DeepSeek 400s a tool loop missing it, so how many carried it is the first question."""
    config = profile_entry(api_key=TEST_KEY, replay_reasoning_field="reasoning_content")
    reasoned = AssistantMessage(
        content=(ThinkingBlock(thinking="because"), TextBlock(text="a")),
        model=MODEL.id,
        provider="deepseek",
    )
    plain = AssistantMessage(content=(TextBlock(text="b"),), model=MODEL.id, provider="deepseek")
    user = UserMessage(content=text("go"))

    with caplog.at_level(logging.DEBUG, logger="lab_harness.llm.providers.chat_completions"):
        await collect(provider(Recorder(sse(finish())), config), ctx((user, reasoned, plain, user)))

    assert "Reasoning replayed on 1 turn(s) as reasoning_content" in caplog.text
