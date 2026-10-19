"""The Anthropic adapter, against a fake SDK client — and, to see what reaches
the wire, the real SDK over a canned transport. Never touches the network.

The contract under test is the one the whole loop depends on: this provider
must not raise. Every failure path has to arrive as a StreamError.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import Any

import anthropic
import pytest
from anthropic.types import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
)

from lab_harness.config import Config
from lab_harness.llm.events import (
    StreamDone,
    StreamError,
    StreamStart,
    TextDelta,
    TextEnd,
    ThinkingDelta,
    ToolCallDelta,
    ToolCallEnd,
)
from lab_harness.llm.provider import StreamOptions
from lab_harness.llm.providers.anthropic import AnthropicProvider
from lab_harness.llm.registry import resolve_model
from lab_harness.llm.types import (
    AssistantMessage,
    Context,
    ProviderBlock,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolDeclaration,
    ToolResultMessage,
    UserMessage,
    text,
)

MODEL = resolve_model("claude-sonnet-5")


# --- fake SDK -------------------------------------------------------------


def _msg_start(input_tokens: int = 10) -> RawMessageStartEvent:
    return RawMessageStartEvent.model_validate(
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": MODEL.id,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 2,
                    "cache_creation_input_tokens": 3,
                },
            },
        }
    )


def _block_start(index: int, block: dict[str, Any]) -> RawContentBlockStartEvent:
    return RawContentBlockStartEvent.model_validate(
        {"type": "content_block_start", "index": index, "content_block": block}
    )


def _block_delta(index: int, delta: dict[str, Any]) -> RawContentBlockDeltaEvent:
    return RawContentBlockDeltaEvent.model_validate(
        {"type": "content_block_delta", "index": index, "delta": delta}
    )


def _block_stop(index: int) -> RawContentBlockStopEvent:
    return RawContentBlockStopEvent.model_validate({"type": "content_block_stop", "index": index})


def _msg_delta(stop_reason: str, output_tokens: int = 7) -> RawMessageDeltaEvent:
    return RawMessageDeltaEvent.model_validate(
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        }
    )


class _FakeStream:
    def __init__(self, events: Sequence[object]) -> None:
        self._events = events

    async def __aenter__(self) -> "_FakeStream":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def __aiter__(self) -> AsyncIterator[object]:
        for event in self._events:
            yield event


class _FakeMessages:
    def __init__(self, events: Sequence[object], error: Exception | None = None) -> None:
        self._events = events
        self._error = error
        self.requests: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> _FakeStream:
        self.requests.append(kwargs)
        if self._error is not None:
            raise self._error
        return _FakeStream(self._events)


class _FakeClient:
    def __init__(self, events: Sequence[object] = (), error: Exception | None = None) -> None:
        self.messages = _FakeMessages(events, error)


#: Claude's adapter settings, as the catalog ships them.
SHIPPED = Config().providers.anthropic


def _provider(events: Sequence[object] = (), error: Exception | None = None) -> AnthropicProvider:
    return AnthropicProvider(SHIPPED, client=_FakeClient(events, error))  # type: ignore[arg-type]


async def _collect(provider: AnthropicProvider, ctx: Context, opts: StreamOptions | None = None):
    return [event async for event in provider.stream(MODEL, ctx, opts)]


def _ctx(**kwargs: Any) -> Context:
    return Context(
        system_prompt=kwargs.get("system_prompt", "sys"),
        messages=kwargs.get("messages", (UserMessage(content=text("hi")),)),
        tools=kwargs.get("tools", ()),
    )


# --- happy path -----------------------------------------------------------


async def test_a_text_response_becomes_the_full_event_sequence() -> None:
    provider = _provider(
        [
            _msg_start(),
            _block_start(0, {"type": "text", "text": ""}),
            _block_delta(0, {"type": "text_delta", "text": "Hel"}),
            _block_delta(0, {"type": "text_delta", "text": "lo"}),
            _block_stop(0),
            _msg_delta("end_turn"),
        ]
    )

    events = await _collect(provider, _ctx())

    assert isinstance(events[0], StreamStart)
    assert [e.delta for e in events if isinstance(e, TextDelta)] == ["Hel", "lo"]
    assert any(isinstance(e, TextEnd) and e.text == "Hello" for e in events)

    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.message.content == (TextBlock(text="Hello"),)
    assert done.message.stop_reason == "stop"


async def test_usage_is_assembled_from_both_ends_of_the_stream() -> None:
    provider = _provider([_msg_start(input_tokens=42), _msg_delta("end_turn", output_tokens=9)])

    events = await _collect(provider, _ctx())

    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.message.usage.input == 42
    assert done.message.usage.output == 9
    assert done.message.usage.cache_read == 2
    assert done.message.usage.cache_write == 3


async def test_tool_calls_are_assembled_from_partial_json() -> None:
    provider = _provider(
        [
            _msg_start(),
            _block_start(0, {"type": "tool_use", "id": "tu_1", "name": "read", "input": {}}),
            _block_delta(0, {"type": "input_json_delta", "partial_json": '{"path":'}),
            _block_delta(0, {"type": "input_json_delta", "partial_json": ' "a.py"}'}),
            _block_stop(0),
            _msg_delta("tool_use"),
        ]
    )

    events = await _collect(provider, _ctx())

    assert [e.delta for e in events if isinstance(e, ToolCallDelta)] == ['{"path":', ' "a.py"}']
    end = next(e for e in events if isinstance(e, ToolCallEnd))
    assert end.tool_call == ToolCallBlock(id="tu_1", name="read", arguments={"path": "a.py"})

    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.message.stop_reason == "tool_use"


async def test_malformed_tool_json_yields_empty_arguments_not_a_salvage() -> None:
    """A truncated object can parse into arguments that validate but lie."""
    provider = _provider(
        [
            _msg_start(),
            _block_start(0, {"type": "tool_use", "id": "tu_1", "name": "read", "input": {}}),
            _block_delta(0, {"type": "input_json_delta", "partial_json": '{"path": "a.p'}),
            _block_stop(0),
            _msg_delta("max_tokens"),
        ]
    )

    events = await _collect(provider, _ctx())

    end = next(e for e in events if isinstance(e, ToolCallEnd))
    assert end.tool_call.arguments == {}
    done = events[-1]
    assert isinstance(done, StreamDone)
    # The loop refuses to execute any call from a length-stopped turn.
    assert done.message.stop_reason == "length"


async def test_thinking_blocks_round_trip_with_their_signature() -> None:
    provider = _provider(
        [
            _msg_start(),
            _block_start(0, {"type": "thinking", "thinking": "", "signature": ""}),
            _block_delta(0, {"type": "thinking_delta", "thinking": "hmm"}),
            _block_delta(0, {"type": "signature_delta", "signature": "sig-1"}),
            _block_stop(0),
            _msg_delta("end_turn"),
        ]
    )

    events = await _collect(provider, _ctx())

    assert [e.delta for e in events if isinstance(e, ThinkingDelta)] == ["hmm"]
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.message.content == (ThinkingBlock(thinking="hmm", signature="sig-1"),)


async def test_unmodeled_server_blocks_survive_as_provider_blocks() -> None:
    """This is what ProviderBlock exists for — no new core type per server tool."""
    provider = _provider(
        [
            _msg_start(),
            _block_start(
                0,
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_1",
                    "content": [],
                },
            ),
            _block_stop(0),
            _msg_delta("end_turn"),
        ]
    )

    events = await _collect(provider, _ctx())

    done = events[-1]
    assert isinstance(done, StreamDone)
    block = done.message.content[0]
    assert isinstance(block, ProviderBlock)
    assert block.provider_type == "web_search_tool_result"
    assert block.data["tool_use_id"] == "srvtoolu_1"


@pytest.mark.parametrize(
    ("wire", "expected"),
    [
        ("stop_sequence", "stop"),
        ("pause_turn", "stop"),
        ("refusal", "stop"),
    ],
)
async def test_stop_reasons_map_to_core_values(wire: str, expected: str) -> None:
    provider = _provider([_msg_start(), _msg_delta(wire)])

    events = await _collect(provider, _ctx())

    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.message.stop_reason == expected


# --- the never-raise contract ---------------------------------------------


async def test_an_api_error_terminates_the_stream_instead_of_raising() -> None:
    error = anthropic.APIStatusError(
        "rate limited",
        response=_fake_response(429),
        body={"error": {"message": "Too many requests"}},
    )
    provider = _provider(error=error)

    events = await _collect(provider, _ctx())

    assert len(events) == 1
    failure = events[0]
    assert isinstance(failure, StreamError)
    assert failure.message.stop_reason == "error"
    assert "Too many requests" in (failure.message.error_message or "")


async def test_an_unexpected_exception_also_becomes_a_stream_error() -> None:
    provider = _provider(error=ValueError("something odd"))

    events = await _collect(provider, _ctx())

    failure = events[0]
    assert isinstance(failure, StreamError)
    assert "something odd" in (failure.message.error_message or "")


async def test_missing_credentials_report_through_the_stream_not_the_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction must not raise; the loop learns about it as an error turn."""
    config = SHIPPED
    monkeypatch.delenv(config.api_key_env, raising=False)
    monkeypatch.delenv(config.auth_token_env, raising=False)

    provider = AnthropicProvider(config)
    events = await _collect(provider, _ctx())

    failure = events[0]
    assert isinstance(failure, StreamError)
    assert config.api_key_env in (failure.message.error_message or "")


async def test_an_already_set_signal_aborts_before_any_request() -> None:
    import asyncio

    signal = asyncio.Event()
    signal.set()
    client = _FakeClient([_msg_start(), _msg_delta("end_turn")])
    provider = AnthropicProvider(SHIPPED, client=client)  # type: ignore[arg-type]

    events = await _collect(provider, _ctx(), StreamOptions(signal=signal))

    failure = events[0]
    assert isinstance(failure, StreamError)
    assert failure.reason == "aborted"
    assert client.messages.requests == []


async def test_a_signal_set_mid_stream_stops_it() -> None:
    """A Ctrl-C partway through a response ends the stream as aborted."""
    signal = asyncio.Event()

    class _SettingStream(_FakeStream):
        """Sets the signal after the first SDK event, before the rest arrive."""

        async def __aiter__(self) -> AsyncIterator[object]:
            yield _msg_start()
            signal.set()
            yield _block_start(0, {"type": "text", "text": ""})
            yield _msg_delta("end_turn")

    client = _FakeClient()
    client.messages.stream = lambda **kwargs: _SettingStream(())  # type: ignore[method-assign]
    provider = AnthropicProvider(SHIPPED, client=client)  # type: ignore[arg-type]

    events = await _collect(provider, _ctx(), StreamOptions(signal=signal))

    failure = events[-1]
    assert isinstance(failure, StreamError)
    assert failure.reason == "aborted"
    assert failure.message.stop_reason == "aborted"
    # The text block after the abort point never made it through.
    assert not any(isinstance(e, TextDelta) for e in events)


async def test_a_translation_failure_becomes_an_error_not_a_crash() -> None:
    """A request we cannot build is still a failed turn, not an exception."""

    class Unbuildable:
        role = "assistant"

        @property
        def content(self) -> tuple[object, ...]:
            raise RuntimeError("translation exploded")

    provider = _provider([_msg_start(), _msg_delta("end_turn")])
    ctx = Context(system_prompt="s", messages=(Unbuildable(),), tools=())  # type: ignore[arg-type]

    events = await _collect(provider, ctx)

    failure = events[0]
    assert isinstance(failure, StreamError)
    assert "Could not build request" in (failure.message.error_message or "")


# --- request translation --------------------------------------------------


async def test_system_prompt_goes_to_the_top_level_field() -> None:
    client = _FakeClient([_msg_start(), _msg_delta("end_turn")])
    provider = AnthropicProvider(SHIPPED, client=client)  # type: ignore[arg-type]

    await _collect(provider, _ctx(system_prompt="be helpful"))

    assert client.messages.requests[0]["system"] == "be helpful"


async def test_tool_declarations_become_input_schema() -> None:
    client = _FakeClient([_msg_start(), _msg_delta("end_turn")])
    provider = AnthropicProvider(SHIPPED, client=client)  # type: ignore[arg-type]
    declaration = ToolDeclaration(
        name="read", description="Read a file", parameters={"type": "object", "properties": {}}
    )

    await _collect(provider, _ctx(tools=(declaration,)))

    tools = client.messages.requests[0]["tools"]
    assert tools == [
        {"name": "read", "description": "Read a file", "input_schema": declaration.parameters}
    ]


async def test_consecutive_tool_results_merge_into_one_user_turn() -> None:
    """The API requires every tool_result for a turn in a single user message."""
    client = _FakeClient([_msg_start(), _msg_delta("end_turn")])
    provider = AnthropicProvider(SHIPPED, client=client)  # type: ignore[arg-type]
    messages = (
        UserMessage(content=text("go")),
        AssistantMessage(
            content=(
                ToolCallBlock(id="t1", name="read", arguments={}),
                ToolCallBlock(id="t2", name="ls", arguments={}),
            ),
            model=MODEL.id,
            provider="anthropic",
            stop_reason="tool_use",
        ),
        ToolResultMessage(tool_call_id="t1", tool_name="read", content=text("one")),
        ToolResultMessage(tool_call_id="t2", tool_name="ls", content=text("two")),
    )

    await _collect(provider, _ctx(messages=messages))

    wire = client.messages.requests[0]["messages"]
    assert [m["role"] for m in wire] == ["user", "assistant", "user"]
    assert len(wire[-1]["content"]) == 2
    assert [b["tool_use_id"] for b in wire[-1]["content"]] == ["t1", "t2"]


async def test_an_error_placeholder_message_is_not_replayed() -> None:
    """Empty content is rejected by the API; an aborted turn has none."""
    client = _FakeClient([_msg_start(), _msg_delta("end_turn")])
    provider = AnthropicProvider(SHIPPED, client=client)  # type: ignore[arg-type]
    messages = (
        UserMessage(content=text("go")),
        AssistantMessage(content=(), model=MODEL.id, provider="anthropic", stop_reason="aborted"),
    )

    await _collect(provider, _ctx(messages=messages))

    wire = client.messages.requests[0]["messages"]
    assert [m["role"] for m in wire] == ["user"]


async def test_signature_less_thinking_blocks_are_dropped() -> None:
    """The API rejects a replayed thinking block without its signature."""
    client = _FakeClient([_msg_start(), _msg_delta("end_turn")])
    provider = AnthropicProvider(SHIPPED, client=client)  # type: ignore[arg-type]
    messages = (
        AssistantMessage(
            content=(ThinkingBlock(thinking="hmm"), TextBlock(text="answer")),
            model=MODEL.id,
            provider="anthropic",
        ),
    )

    await _collect(provider, _ctx(messages=messages))

    content = client.messages.requests[0]["messages"][0]["content"]
    assert [b["type"] for b in content] == ["text"]


async def test_provider_blocks_are_replayed_verbatim() -> None:
    client = _FakeClient([_msg_start(), _msg_delta("end_turn")])
    provider = AnthropicProvider(SHIPPED, client=client)  # type: ignore[arg-type]
    messages = (
        AssistantMessage(
            content=(ProviderBlock(provider_type="web_search_tool_use", data={"id": "s1"}),),
            model=MODEL.id,
            provider="anthropic",
        ),
    )

    await _collect(provider, _ctx(messages=messages))

    content = client.messages.requests[0]["messages"][0]["content"]
    assert content == [{"type": "web_search_tool_use", "id": "s1"}]


async def test_max_tokens_comes_from_the_model_not_a_literal() -> None:
    client = _FakeClient([_msg_start(), _msg_delta("end_turn")])
    provider = AnthropicProvider(SHIPPED, client=client)  # type: ignore[arg-type]

    await _collect(provider, _ctx())

    assert client.messages.requests[0]["max_tokens"] == MODEL.max_tokens


HAIKU = resolve_model("claude-haiku-4-5-20251001")


async def _request_for(
    model_id: str, opts: StreamOptions | None = None, **config: Any
) -> dict[str, Any]:
    client = _FakeClient([_msg_start(), _msg_delta("end_turn")])
    provider = AnthropicProvider(replace(SHIPPED, **config), client=client)  # type: ignore[arg-type]
    _ = [event async for event in provider.stream(resolve_model(model_id), _ctx(), opts)]
    return client.messages.requests[0]


@pytest.mark.parametrize("model_id", ["claude-fable-5", "claude-opus-5"])
async def test_adaptive_models_ask_for_summarized_thinking(model_id: str) -> None:
    """All think unasked, but return empty thinking text unless asked for it."""
    request = await _request_for(model_id)

    assert request["thinking"] == {"type": "adaptive", "display": "summarized"}


async def test_an_adaptive_model_is_never_sent_a_budget() -> None:
    """The 5-series rejects `budget_tokens` with a 400; effort is the dial."""
    request = await _request_for("claude-sonnet-5", StreamOptions(reasoning="high"))

    assert request["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert request["output_config"] == {"effort": "high"}


async def test_an_empty_display_sends_no_preference() -> None:
    request = await _request_for("claude-opus-5", thinking_display="")

    assert request["thinking"] == {"type": "adaptive"}


async def test_thinking_drops_temperature() -> None:
    """The API rejects temperature alongside thinking, of either kind."""
    adaptive = await _request_for("claude-sonnet-5", StreamOptions(temperature=0.7))
    budget = await _request_for(
        HAIKU.id, StreamOptions(reasoning="high", temperature=0.7), thinking_budget_tokens=2_048
    )

    assert "temperature" not in adaptive
    assert budget["thinking"] == {"type": "enabled", "budget_tokens": 2_048}
    assert "temperature" not in budget


async def test_a_budget_model_thinks_only_when_asked() -> None:
    """Haiku 4.5 predates adaptive thinking and does not think by default."""
    request = await _request_for(HAIKU.id, StreamOptions(temperature=0.7))

    assert "thinking" not in request
    assert request["temperature"] == 0.7


@pytest.mark.parametrize("model_id", ["claude-haiku-4-5", "claude-sonnet-4-20250514"])
async def test_budget_models_match_by_id_prefix(model_id: str) -> None:
    """So a dated id and its alias take the same request shape."""
    unasked = await _request_for(model_id)
    asked = await _request_for(
        model_id, StreamOptions(reasoning="high"), thinking_budget_tokens=1_024
    )

    assert "thinking" not in unasked, "a pre-4.6 model may not support thinking at all"
    assert asked["thinking"] == {"type": "enabled", "budget_tokens": 1_024}
    assert "output_config" not in asked


async def test_the_models_effort_is_the_request_for_either_kind() -> None:
    """Set in config rather than per request: effort for adaptive, a budget for the rest."""
    adaptive = replace(resolve_model("claude-sonnet-5"), reasoning_effort="low")
    budget = replace(HAIKU, reasoning_effort="low")
    client = _FakeClient([_msg_start(), _msg_delta("end_turn")])
    provider = AnthropicProvider(SHIPPED, client=client)  # type: ignore[arg-type]

    for model in (adaptive, budget):
        _ = [event async for event in provider.stream(model, _ctx())]

    assert client.messages.requests[0]["output_config"] == {"effort": "low"}
    assert client.messages.requests[1]["thinking"]["type"] == "enabled"


async def test_a_model_without_reasoning_gets_no_thinking_request() -> None:
    from dataclasses import replace

    client = _FakeClient([_msg_start(), _msg_delta("end_turn")])
    provider = AnthropicProvider(SHIPPED, client=client)  # type: ignore[arg-type]
    plain = replace(MODEL, supports_reasoning=False)

    _ = [event async for event in provider.stream(plain, _ctx(), StreamOptions(reasoning="high"))]

    assert "thinking" not in client.messages.requests[0]


# --- what was sent, rebuilt -------------------------------------------------


async def test_request_body_is_the_json_the_sdk_posts() -> None:
    """Through the real SDK, over a canned transport: what reaches the wire."""
    import httpx2

    events = (
        _msg_start(),
        _block_start(0, {"type": "text", "text": ""}),
        _block_delta(0, {"type": "text_delta", "text": "Hi"}),
        _block_stop(0),
        _msg_delta("end_turn"),
        RawMessageStopEvent(type="message_stop"),
    )
    body = "".join(f"event: {e.type}\ndata: {e.model_dump_json()}\n\n" for e in events)
    posted: list[dict[str, Any]] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        posted.append(json.loads(request.content))
        return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = anthropic.AsyncAnthropic(
        api_key="test-key-not-real",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(answer)),
        max_retries=0,
    )
    provider = AnthropicProvider(SHIPPED, client=client)
    earlier = AssistantMessage(
        content=(ThinkingBlock(thinking="t", signature="sig"), TextBlock(text="a1")),
        model=MODEL.id,
        provider="anthropic",
    )
    tools = (ToolDeclaration(name="ls", description="List.", parameters={"type": "object"}),)
    ctx = _ctx(
        messages=(UserMessage(content=text("q1")), earlier, UserMessage(content=text("q2"))),
        tools=tools,
    )

    streamed = await _collect(provider, ctx)

    assert isinstance(streamed[-1], StreamDone)
    assert provider.request_body(MODEL, ctx) == posted[0]


def test_reply_entries_are_what_the_next_request_carries() -> None:
    provider = _provider()
    reply = AssistantMessage(
        content=(
            ThinkingBlock(thinking="t", signature="sig"),
            TextBlock(text="Looking."),
            ToolCallBlock(id="c1", name="ls", arguments={"path": "."}),
        ),
        model=MODEL.id,
        provider="anthropic",
    )

    sent = provider.request_body(MODEL, _ctx(messages=(UserMessage(content=text("q")), reply)))

    assert provider.reply_entries(reply) == sent["messages"][1:]


def _fake_response(status: int) -> Any:
    # httpx2 is what the SDK vendors; constructing an APIStatusError needs the
    # same Response class it does.
    import httpx2

    return httpx2.Response(
        status_code=status, request=httpx2.Request("POST", "https://example.test")
    )


# --- what is logged -------------------------------------------------------


async def test_the_log_names_the_response_and_its_stop_reason_as_sent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Refusal maps to "stop", so the log is where the real reason survives."""
    provider = _provider([_msg_start(), _msg_delta("refusal")])

    with caplog.at_level(logging.DEBUG, logger="lab_harness.llm.providers.anthropic"):
        await _collect(provider, _ctx())

    assert (
        f"Request to {MODEL.id}: max_tokens={MODEL.max_tokens} messages=1 system=3" in caplog.text
    )
    assert 'thinking={"type":"adaptive","display":"summarized"}' in caplog.text
    assert f"Response msg_1 from {MODEL.id}" in caplog.text
    assert "stop_reason=refusal" in caplog.text
