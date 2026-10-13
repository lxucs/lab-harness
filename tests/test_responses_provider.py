"""The OpenAI Responses adapter.

Offline, through `httpx.MockTransport`, so the real request-building and the
real SSE parsing run against canned frames.

The section that earns its keep is **reasoning round-trip**. A reasoning model
thinks, calls a tool, and has to resume thinking from where it left off; the
only thing that makes that possible is replaying the encrypted blob verbatim.
If those tests pass but the replay is wrong, every multi-step task with a
reasoning model silently gets worse — nothing errors, the model just forgets.
"""

import asyncio
import json
from typing import Any

import pytest

from harness.config import OpenAIConfig
from harness.llm.events import (
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
from harness.llm.providers.responses import ResponsesProvider
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
    id="gpt-5",
    name="GPT-5",
    provider="openai",
    context_window=400_000,
    max_tokens=128_000,
)

REASONING_MODEL = Model(
    id="gpt-5",
    name="GPT-5",
    provider="openai",
    context_window=400_000,
    max_tokens=128_000,
    supports_reasoning=True,
)

#: What a reasoning item's replay token looks like once packed into
#: `ThinkingBlock.signature`. Opaque to the core; this module owns the shape.
ENCRYPTED = "gAAAAABn-encrypted-blob"


def sse(*events: dict[str, Any]) -> str:
    """Responses streams typed events; the type is inside the payload."""
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def item_added(index: int, item: dict[str, Any]) -> dict[str, Any]:
    return {"type": "response.output_item.added", "output_index": index, "item": item}


def item_done(index: int, item: dict[str, Any]) -> dict[str, Any]:
    return {"type": "response.output_item.done", "output_index": index, "item": item}


def text_delta(index: int, value: str) -> dict[str, Any]:
    return {"type": "response.output_text.delta", "output_index": index, "delta": value}


def summary_delta(index: int, value: str) -> dict[str, Any]:
    return {
        "type": "response.reasoning_summary_text.delta",
        "output_index": index,
        "delta": value,
    }


def args_delta(index: int, value: str) -> dict[str, Any]:
    return {
        "type": "response.function_call_arguments.delta",
        "output_index": index,
        "delta": value,
    }


def completed(usage: dict[str, Any] | None = None) -> dict[str, Any]:
    response: dict[str, Any] = {"status": "completed"}
    if usage is not None:
        response["usage"] = usage
    return {"type": "response.completed", "response": response}


def reasoning_item(
    identifier: str = "rs_1", summary: str = "", encrypted: str = ENCRYPTED
) -> dict[str, Any]:
    item: dict[str, Any] = {"type": "reasoning", "id": identifier}
    if encrypted:
        item["encrypted_content"] = encrypted
    item["summary"] = [{"type": "summary_text", "text": summary}] if summary else []
    return item


def provider(
    recorder: Recorder | Exploding,
    config: OpenAIConfig | None = None,
    name: str = "openai",
) -> ResponsesProvider:
    config = config or OpenAIConfig(api="responses", api_key=TEST_KEY)
    return ResponsesProvider(config, name, client=recorder.client())


def ctx(
    messages: tuple[Message, ...] = (),
    system_prompt: str = "",
    tools: tuple[ToolDeclaration, ...] = (),
) -> Context:
    return Context(system_prompt=system_prompt, messages=messages, tools=tools)


async def collect(
    prov: ResponsesProvider,
    context: Context | None = None,
    opts: StreamOptions | None = None,
    model: Model = MODEL,
) -> list[StreamEvent]:
    return [event async for event in prov.stream(model, context or ctx(), opts)]


def signed(thinking: str = "I considered it", identifier: str = "rs_1") -> ThinkingBlock:
    """A thinking block carrying a replay token, as the adapter produces one."""
    return ThinkingBlock(
        thinking=thinking,
        signature=json.dumps({"id": identifier, "encrypted_content": ENCRYPTED}),
    )


# --- the request ----------------------------------------------------------


async def test_posts_to_the_responses_path() -> None:
    recorder = Recorder(sse(completed()))

    await collect(
        provider(recorder, OpenAIConfig(api_key=TEST_KEY, base_url="https://api.openai.com/v1"))
    )

    assert recorder.request is not None
    assert str(recorder.request.url) == "https://api.openai.com/v1/responses"


async def test_the_system_prompt_is_top_level_instructions() -> None:
    """Like the Messages API, unlike Chat Completions."""
    recorder = Recorder(sse(completed()))

    await collect(provider(recorder), ctx(system_prompt="Be brief."))

    assert recorder.payload["instructions"] == "Be brief."
    assert all(i.get("role") != "system" for i in recorder.payload["input"])


async def test_the_conversation_is_input_not_messages() -> None:
    recorder = Recorder(sse(completed()))

    await collect(provider(recorder), ctx(messages=(UserMessage(content=text("hi")),)))

    assert "messages" not in recorder.payload
    assert recorder.payload["input"] == [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
    ]


async def test_tool_declarations_are_flat() -> None:
    """Chat Completions nests all of this under a "function" key."""
    recorder = Recorder(sse(completed()))
    declaration = ToolDeclaration(
        name="read", description="Read a file", parameters={"type": "object", "properties": {}}
    )

    await collect(provider(recorder), ctx(tools=(declaration,)))

    assert recorder.payload["tools"] == [
        {
            "type": "function",
            "name": "read",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {}},
        }
    ]


async def test_the_token_ceiling_uses_the_responses_name() -> None:
    recorder = Recorder(sse(completed()))

    await collect(provider(recorder), opts=StreamOptions(max_tokens=99))

    assert recorder.payload["max_output_tokens"] == 99
    assert "max_tokens" not in recorder.payload


async def test_the_server_is_told_not_to_store_the_thread() -> None:
    """The harness owns its transcript; a server-side thread --resume cannot see."""
    recorder = Recorder(sse(completed()))

    await collect(provider(recorder))

    assert recorder.payload["store"] is False


async def test_encrypted_reasoning_is_requested_for_a_reasoning_model() -> None:
    recorder = Recorder(sse(completed()))

    await collect(provider(recorder), model=REASONING_MODEL)

    assert recorder.payload["include"] == ["reasoning.encrypted_content"]
    assert recorder.payload["reasoning"]["summary"] == "auto"


async def test_a_non_reasoning_model_asks_for_none_of_it() -> None:
    recorder = Recorder(sse(completed()))

    await collect(provider(recorder), model=MODEL)

    assert "reasoning" not in recorder.payload
    assert "include" not in recorder.payload


async def test_reasoning_effort_is_forwarded() -> None:
    recorder = Recorder(sse(completed()))

    await collect(provider(recorder), opts=StreamOptions(reasoning="high"), model=REASONING_MODEL)

    assert recorder.payload["reasoning"]["effort"] == "high"


async def test_the_summary_can_be_turned_off() -> None:
    recorder = Recorder(sse(completed()))

    await collect(
        provider(recorder, OpenAIConfig(api_key=TEST_KEY, reasoning_summary="")),
        model=REASONING_MODEL,
    )

    assert "summary" not in recorder.payload.get("reasoning", {})


async def test_storing_server_side_drops_the_encrypted_request() -> None:
    """The two are alternatives: stored threads are referenced, not replayed."""
    recorder = Recorder(sse(completed()))

    await collect(
        provider(recorder, OpenAIConfig(api_key=TEST_KEY, store=True)), model=REASONING_MODEL
    )

    assert recorder.payload["store"] is True
    assert "include" not in recorder.payload


# --- items out ------------------------------------------------------------


async def test_a_tool_result_is_a_sibling_item() -> None:
    """Not nested in a message, as it would be in the other two protocols."""
    recorder = Recorder(sse(completed()))
    message = ToolResultMessage(tool_call_id="call_1", tool_name="read", content=text("file body"))

    await collect(provider(recorder), ctx(messages=(message,)))

    assert recorder.payload["input"] == [
        {"type": "function_call_output", "call_id": "call_1", "output": "file body"}
    ]


async def test_an_assistant_turn_flattens_into_its_items() -> None:
    """Reasoning, then text, then calls — reasoning must precede its output."""
    recorder = Recorder(sse(completed()))
    message = AssistantMessage(
        content=(
            signed("thought about it"),
            TextBlock(text="here goes"),
            ToolCallBlock(id="call_1", name="read", arguments={"path": "a.py"}),
        ),
        model="gpt-5",
        provider="openai",
    )

    await collect(provider(recorder), ctx(messages=(message,)))

    kinds = [i["type"] for i in recorder.payload["input"]]
    assert kinds == ["reasoning", "message", "function_call"]


async def test_tool_call_arguments_go_back_as_a_json_string() -> None:
    recorder = Recorder(sse(completed()))
    message = AssistantMessage(
        content=(ToolCallBlock(id="call_1", name="read", arguments={"path": "a.py"}),),
        model="gpt-5",
        provider="openai",
    )

    await collect(provider(recorder), ctx(messages=(message,)))

    assert recorder.payload["input"][0] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "read",
        "arguments": '{"path": "a.py"}',
    }


async def test_an_image_becomes_an_input_image_part() -> None:
    recorder = Recorder(sse(completed()))
    message = UserMessage(
        content=(TextBlock(text="what is this"), ImageBlock(data="QUJD", mime_type="image/png"))
    )

    await collect(provider(recorder), ctx(messages=(message,)))

    assert recorder.payload["input"][0]["content"] == [
        {"type": "input_text", "text": "what is this"},
        {"type": "input_image", "image_url": "data:image/png;base64,QUJD"},
    ]


# --- reasoning round-trip: the point of this adapter ----------------------


async def test_a_reasoning_item_is_replayed_with_its_ciphertext() -> None:
    """The whole reason this protocol exists here.

    Drop the blob and the model restarts its reasoning from nothing after
    every tool call — silently, with no error to notice.
    """
    recorder = Recorder(sse(completed()))
    message = AssistantMessage(
        content=(signed("I weighed the options", "rs_abc"),), model="gpt-5", provider="openai"
    )

    await collect(provider(recorder), ctx(messages=(message,)))

    item = recorder.payload["input"][0]
    assert item["type"] == "reasoning"
    assert item["id"] == "rs_abc"
    assert item["encrypted_content"] == ENCRYPTED


async def test_the_summary_rides_back_in_the_shape_it_arrived_in() -> None:
    recorder = Recorder(sse(completed()))
    message = AssistantMessage(
        content=(signed("I weighed the options"),), model="gpt-5", provider="openai"
    )

    await collect(provider(recorder), ctx(messages=(message,)))

    assert recorder.payload["input"][0]["summary"] == [
        {"type": "summary_text", "text": "I weighed the options"}
    ]


async def test_a_signature_less_thinking_block_is_dropped() -> None:
    """A resumed DeepSeek session: plaintext reasoning, no replay token.

    Sending a reasoning item the server cannot decrypt is a 400. Dropping it
    only costs the model some context.
    """
    recorder = Recorder(sse(completed()))
    message = AssistantMessage(
        content=(ThinkingBlock(thinking="from deepseek", signature=None), TextBlock(text="hi")),
        model="deepseek-v4-pro",
        provider="deepseek",
    )

    await collect(provider(recorder), ctx(messages=(message,)))

    kinds = [i["type"] for i in recorder.payload["input"]]
    assert kinds == ["message"], "a reasoning item without a token must not be sent"


async def test_an_unparseable_signature_is_dropped_rather_than_sent() -> None:
    """An Anthropic signature is an opaque string, not our JSON envelope."""
    recorder = Recorder(sse(completed()))
    message = AssistantMessage(
        content=(ThinkingBlock(thinking="x", signature="anthropic-style-sig"),),
        model="gpt-5",
        provider="openai",
    )

    await collect(provider(recorder), ctx(messages=(message,)))

    assert recorder.payload["input"] == []


async def test_the_round_trip_survives_a_full_cycle() -> None:
    """Stream a reasoning turn, then feed the result straight back.

    This is the test that would catch a packing/unpacking mismatch, which
    nothing else would: each half can look right on its own.
    """
    first = Recorder(
        sse(
            item_added(0, {"type": "reasoning", "id": "rs_9"}),
            summary_delta(0, "weighing it"),
            item_done(0, reasoning_item("rs_9", "weighing it")),
            completed(),
        )
    )
    events = await collect(provider(first), model=REASONING_MODEL)
    produced = final(events)

    second = Recorder(sse(completed()))
    await collect(provider(second), ctx(messages=(produced,)), model=REASONING_MODEL)

    item = second.payload["input"][0]
    assert item["id"] == "rs_9"
    assert item["encrypted_content"] == ENCRYPTED


# --- streaming ------------------------------------------------------------


async def test_text_streams_as_start_deltas_and_end() -> None:
    recorder = Recorder(
        sse(
            item_added(0, {"type": "message", "role": "assistant"}),
            text_delta(0, "Hel"),
            text_delta(0, "lo"),
            completed(),
        )
    )

    events = await collect(provider(recorder))

    assert isinstance(events[0], StreamStart)
    assert any(isinstance(e, TextStart) for e in events)
    assert [e.delta for e in events if isinstance(e, TextDelta)] == ["Hel", "lo"]
    assert [e.text for e in events if isinstance(e, TextEnd)] == ["Hello"]
    assert final(events).content == (TextBlock(text="Hello"),)


async def test_a_reasoning_summary_streams_as_thinking() -> None:
    recorder = Recorder(
        sse(
            item_added(0, {"type": "reasoning", "id": "rs_1"}),
            summary_delta(0, "first "),
            summary_delta(0, "thought"),
            item_done(0, reasoning_item("rs_1", "first thought")),
            completed(),
        )
    )

    events = await collect(provider(recorder), model=REASONING_MODEL)

    assert any(isinstance(e, ThinkingStart) for e in events)
    assert [e.delta for e in events if isinstance(e, ThinkingDelta)] == ["first ", "thought"]
    assert [e.thinking for e in events if isinstance(e, ThinkingEnd)] == ["first thought"]


async def test_a_reasoning_item_with_no_summary_still_produces_a_block() -> None:
    """The ciphertext is the part that matters; the summary is optional.

    Dropping the block would lose the model's place between tool calls.
    """
    recorder = Recorder(
        sse(
            item_added(0, {"type": "reasoning", "id": "rs_1"}),
            item_done(0, reasoning_item("rs_1", summary="")),
            completed(),
        )
    )

    events = await collect(provider(recorder), model=REASONING_MODEL)

    blocks = [b for b in final(events).content if isinstance(b, ThinkingBlock)]
    assert len(blocks) == 1
    assert blocks[0].thinking == ""
    assert blocks[0].signature is not None
    assert any(isinstance(e, ThinkingStart) for e in events)
    assert any(isinstance(e, ThinkingEnd) for e in events)


async def test_a_tool_call_accumulates_across_frames() -> None:
    recorder = Recorder(
        sse(
            item_added(0, {"type": "function_call", "call_id": "call_1", "name": "read"}),
            args_delta(0, '{"path"'),
            args_delta(0, ': "a.py"}'),
            item_done(
                0,
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read",
                    "arguments": '{"path": "a.py"}',
                },
            ),
            completed(),
        )
    )

    events = await collect(provider(recorder))

    assert any(isinstance(e, ToolCallStart) for e in events)
    assert [e.delta for e in events if isinstance(e, ToolCallDelta)] == ['{"path"', ': "a.py"}']
    ends = [e for e in events if isinstance(e, ToolCallEnd)]
    assert ends[0].tool_call == ToolCallBlock(id="call_1", name="read", arguments={"path": "a.py"})
    assert final(events).stop_reason == "tool_use"


async def test_the_final_item_wins_over_dropped_argument_deltas() -> None:
    """A stream that lost a delta would otherwise hand the model half an object."""
    recorder = Recorder(
        sse(
            item_added(0, {"type": "function_call", "call_id": "call_1", "name": "read"}),
            args_delta(0, '{"pa'),
            item_done(
                0,
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read",
                    "arguments": '{"path": "a.py"}',
                },
            ),
            completed(),
        )
    )

    events = await collect(provider(recorder))

    assert final(events).tool_calls[0].arguments == {"path": "a.py"}


async def test_partial_tool_json_is_never_salvaged() -> None:
    recorder = Recorder(
        sse(
            item_added(0, {"type": "function_call", "call_id": "call_1", "name": "read"}),
            args_delta(0, '{"pa'),
            completed(),
        )
    )

    events = await collect(provider(recorder))

    assert final(events).tool_calls[0].arguments == {}


async def test_items_keep_distinct_block_indices() -> None:
    recorder = Recorder(
        sse(
            item_added(0, {"type": "reasoning", "id": "rs_1"}),
            summary_delta(0, "think"),
            item_added(1, {"type": "message", "role": "assistant"}),
            text_delta(1, "say"),
            item_added(2, {"type": "function_call", "call_id": "call_1", "name": "ls"}),
            args_delta(2, "{}"),
            completed(),
        )
    )

    events = await collect(provider(recorder), model=REASONING_MODEL)

    starts = [e for e in events if isinstance(e, ThinkingStart | TextStart | ToolCallStart)]
    indices = [e.block_index for e in starts]
    assert indices == sorted(set(indices))
    assert len(indices) == 3


async def test_an_unknown_item_type_is_ignored() -> None:
    """A new item type must not break the turn's text."""
    recorder = Recorder(
        sse(
            item_added(0, {"type": "web_search_call", "id": "ws_1"}),
            item_added(1, {"type": "message", "role": "assistant"}),
            text_delta(1, "answer"),
            completed(),
        )
    )

    events = await collect(provider(recorder))

    assert final(events).content == (TextBlock(text="answer"),)


async def test_usage_uses_the_responses_field_names() -> None:
    recorder = Recorder(
        sse(
            completed(
                {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "input_tokens_details": {"cached_tokens": 40},
                    "output_tokens_details": {"reasoning_tokens": 30},
                }
            )
        )
    )

    events = await collect(provider(recorder))

    usage = final(events).usage
    # input_tokens includes the cached portion; subtracting avoids double count.
    assert (usage.input, usage.output, usage.cache_read) == (60, 50, 40)
    assert usage.total == 150


@pytest.mark.parametrize(
    ("reason", "expected"),
    [("max_output_tokens", "length"), ("content_filter", "stop"), ("something_new", "stop")],
)
async def test_incomplete_reasons_map_to_stop_reasons(reason: str, expected: str) -> None:
    recorder = Recorder(
        sse(
            {
                "type": "response.incomplete",
                "response": {"status": "incomplete", "incomplete_details": {"reason": reason}},
            }
        )
    )

    events = await collect(provider(recorder))

    assert final(events).stop_reason == expected


# --- failures: the adapter never raises -----------------------------------


async def test_a_missing_credential_is_a_stream_error() -> None:
    recorder = Recorder(sse(completed()))

    events = await collect(provider(recorder, OpenAIConfig(api_key="", api_key_env="NOT_SET")))

    assert isinstance(events[0], StreamError)
    assert "NOT_SET" in (final(events).error_message or "")


@pytest.mark.parametrize("status", [400, 401, 429, 500])
async def test_http_errors_become_stream_errors(status: int) -> None:
    recorder = Recorder(json.dumps({"error": {"message": "nope"}}), status=status)

    events = await collect(provider(recorder))

    assert isinstance(events[-1], StreamError)
    assert "nope" in (final(events).error_message or "")


async def test_an_error_event_mid_stream_ends_the_turn() -> None:
    """A Responses stream can start 200 OK and then fail."""
    recorder = Recorder(
        sse(
            item_added(0, {"type": "message", "role": "assistant"}),
            text_delta(0, "partial"),
            {"type": "error", "message": "model exploded"},
        )
    )

    events = await collect(provider(recorder))

    assert isinstance(events[-1], StreamError)
    assert "model exploded" in (final(events).error_message or "")


async def test_a_failed_response_event_ends_the_turn() -> None:
    recorder = Recorder(
        sse(
            {
                "type": "response.failed",
                "response": {"status": "failed", "error": {"message": "server gave up"}},
            }
        )
    )

    events = await collect(provider(recorder))

    assert isinstance(events[-1], StreamError)
    assert "server gave up" in (final(events).error_message or "")


async def test_a_transport_failure_becomes_a_stream_error() -> None:
    events = await collect(provider(Exploding()))

    assert isinstance(events[-1], StreamError)
    assert "ConnectError" in (final(events).error_message or "")


async def test_an_abort_before_the_request_stops_immediately() -> None:
    signal = asyncio.Event()
    signal.set()
    recorder = Recorder(sse(completed()))

    events = await collect(provider(recorder), opts=StreamOptions(signal=signal))

    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert events[0].reason == "aborted"
    assert recorder.request is None


async def test_an_abort_mid_stream_terminates_the_turn() -> None:
    signal = asyncio.Event()
    recorder = Recorder(
        sse(
            item_added(0, {"type": "message", "role": "assistant"}),
            text_delta(0, "a"),
            text_delta(0, "b"),
            completed(),
        )
    )
    stream = provider(recorder).stream(MODEL, ctx(), StreamOptions(signal=signal))

    events: list[StreamEvent] = []
    async for event in stream:
        events.append(event)
        if isinstance(event, TextDelta):
            signal.set()

    assert isinstance(events[-1], StreamError)
    assert events[-1].reason == "aborted"


async def test_a_malformed_frame_does_not_lose_the_stream() -> None:
    body = "data: {not json\n\n" + sse(
        item_added(0, {"type": "message", "role": "assistant"}),
        text_delta(0, "hi"),
        completed(),
    )
    recorder = Recorder(body)

    events = await collect(provider(recorder))

    assert final(events).content == (TextBlock(text="hi"),)


# --- protocol ordering -----------------------------------------------------


async def test_every_block_is_opened_before_its_deltas() -> None:
    """docs/events.md: updates never appear before their start.

    Presence is not enough — a start emitted at the end of the stream still
    "exists" while leaving a renderer with deltas for a block it was never
    told about.
    """
    recorder = Recorder(
        sse(
            item_added(0, {"type": "reasoning", "id": "rs_1"}),
            summary_delta(0, "think"),
            item_added(1, {"type": "message", "role": "assistant"}),
            text_delta(1, "say"),
            item_added(2, {"type": "function_call", "call_id": "call_1", "name": "ls"}),
            args_delta(2, "{}"),
            completed(),
        )
    )

    events = await collect(provider(recorder), model=REASONING_MODEL)

    assert_blocks_open_before_deltas(events)


async def test_a_block_is_closed_exactly_once() -> None:
    recorder = Recorder(
        sse(
            item_added(0, {"type": "message", "role": "assistant"}),
            text_delta(0, "hi"),
            item_done(0, {"type": "message", "role": "assistant"}),
            completed(),
        )
    )

    events = await collect(provider(recorder))

    assert len([e for e in events if isinstance(e, TextEnd)]) == 1
