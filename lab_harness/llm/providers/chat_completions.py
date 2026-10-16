"""The Chat Completions protocol.

OpenAI has moved on to the Responses API (`responses.py`), but this dialect is
what almost everyone else speaks: DeepSeek, GLM, Groq, OpenRouter, Together,
vLLM and Ollama. It is the lingua franca, not the legacy path, which is why it
is named for the protocol rather than for a vendor.

A profile picks its protocol with `api = "chat_completions"` — the default.

**This module must never raise** for request, model, or runtime failures — see
`lab_harness/llm/provider.py`. Every failure path ends in a `StreamError`, built by
the shared `_http.failure`.

Reasoning here is *plaintext*, and whether it goes back is the vendor's rule,
not ours. DeepSeek returns a 400 for a request carrying tools unless every
earlier turn's `reasoning_content` is sent back; GLM asks for it unmodified,
and keeps it only when asked to (`preserve_thinking`); a server that never
asked for it may reject the field. So it is a per-profile setting,
`replay_reasoning_field`, on for the shipped DeepSeek and GLM profiles and off
otherwise. `ThinkingBlock.signature` stays None either way — there is
no replay token, only the text. The Responses API is the opposite case; see
docs/providers.md.
"""

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from lab_harness.config import OpenAIConfig, profile_defaults
from lab_harness.llm.events import (
    StreamDone,
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
from lab_harness.llm.providers._trace import request_shape
from lab_harness.llm.types import (
    AssistantMessage,
    ContentBlock,
    Context,
    ImageBlock,
    Message,
    Model,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
)

#: Chat Completions finish_reason -> our StopReason. Anything unlisted means
#: "stop", including `content_filter`: the turn really did end, and the model
#: saying why is content rather than a transport failure.
_FINISH_REASONS: dict[str, StopReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "stop",
}

#: Delta fields carrying reasoning text, in the order they are checked.
#: `reasoning_content` is DeepSeek's; `reasoning` is what several gateways
#: (OpenRouter among them) relabel it to.
_REASONING_FIELDS = ("reasoning_content", "reasoning")

#: GLM's request for Preserved Thinking, exactly as Z.ai documents it.
#: `clear_thinking: false` is the part that matters; `type: "enabled"` is part
#: of the documented shape, and also switches thinking on for a hybrid model.
_PRESERVED_THINKING = {"type": "enabled", "clear_thinking": False}

_log = logging.getLogger(__name__)


class ChatCompletionsProvider:
    """Streams from an OpenAI-compatible Chat Completions endpoint.

    `name` is the *profile* name, not the vendor's — it is what lands in
    `AssistantMessage.provider`, so a transcript records which endpoint
    answered rather than the dialect it spoke.
    """

    def __init__(
        self,
        config: OpenAIConfig | None = None,
        name: str = "openai",
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config if config is not None else profile_defaults()
        self._client = client
        self.name = name
        _http.log_sources(name, self._config)

    async def stream(
        self,
        model: Model,
        ctx: Context,
        opts: StreamOptions | None = None,
    ) -> AsyncIterator[StreamEvent]:
        options = opts if opts is not None else StreamOptions()
        cfg = self._config

        if (stop := _http.precondition(model, self.name, cfg, options)) is not None:
            yield stop
            return

        try:
            request = self._build_request(model, ctx, options)
        except Exception as exc:
            # Translation bugs surface as an error turn rather than taking the
            # run down. The loop cannot repair them either way.
            yield _http.failure(model, self.name, f"Could not build request: {exc}")
            return

        url = _http.endpoint(cfg, "chat/completions")
        headers = _http.headers(_http.credential(cfg))

        _log.debug("Request to %s via %s: %s", model.id, self.name, request_shape(request))
        if field := cfg.replay_reasoning_field:
            replayed = sum(field in m for m in request["messages"])
            _log.debug("Reasoning replayed on %d turn(s) as %s", replayed, field)

        accumulator = _Accumulator(model=model, provider=self.name)
        try:
            async with (
                _http.client(self._client, cfg) as client,
                client.stream("POST", url, json=request, headers=headers) as response,
            ):
                if response.status_code >= 400:
                    # A streamed response has not read its body yet, and the
                    # error detail lives in it.
                    yield _http.failure(
                        model, self.name, _http.describe(response, await response.aread())
                    )
                    return
                yield StreamStart()
                async for line in response.aiter_lines():
                    if _http.aborted(options):
                        yield _http.failure(
                            model, self.name, _http.ABORTED_NOTICE, reason="aborted"
                        )
                        return
                    payload = _http.frame(line)
                    if payload is None:
                        continue
                    if isinstance(payload, _http.Done):
                        break
                    for translated in accumulator.feed(payload):
                        yield translated
        except Exception as exc:
            # httpx.HTTPError and anything else alike: the contract is the
            # same, and a failure part-way through a stream is still a failed
            # turn rather than a partial one.
            yield _http.failure(model, self.name, f"{type(exc).__name__}: {exc}")
            return

        for closing in accumulator.close():
            yield closing
        yield StreamDone(message=accumulator.message(), reason=accumulator.stop_reason)

    def _build_request(self, model: Model, ctx: Context, opts: StreamOptions) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": model.id,
            "messages": _to_wire_messages(
                ctx.system_prompt, ctx.messages, self._config.replay_reasoning_field
            ),
            "stream": True,
            "max_tokens": opts.max_tokens or model.max_tokens or self._config.default_max_tokens,
        }
        if self._config.include_usage:
            # Without this the final chunk carries no usage at all, and token
            # accounting silently reads zero for every turn.
            request["stream_options"] = {"include_usage": True}
        if self._config.preserve_thinking:
            request["thinking"] = dict(_PRESERVED_THINKING)
        if ctx.tools:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in ctx.tools
            ]
        if opts.temperature is not None:
            request["temperature"] = opts.temperature
        if opts.reasoning is not None and model.supports_reasoning:
            request["reasoning_effort"] = opts.reasoning
        return request


# --- request translation --------------------------------------------------


def _to_wire_messages(
    system_prompt: str, messages: tuple[Message, ...], reasoning_field: str
) -> list[dict[str, Any]]:
    """Core messages -> Chat Completions `messages`.

    Two differences from the Anthropic adapter, both forced by the wire format:
    the system prompt is a *message* here rather than a top-level field, and
    tool results stay separate rather than merging into one user turn.

    `reasoning_field` names where an assistant turn's reasoning goes back, or
    is empty to send none — see the module docstring.
    """
    wire: list[dict[str, Any]] = []
    if system_prompt:
        wire.append({"role": "system", "content": system_prompt})
    for message in messages:
        if message.role == "system":
            # Already carried above; a stray one in the transcript is not
            # replayed, matching the Anthropic adapter.
            continue
        if isinstance(message, ToolResultMessage):
            wire.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": _flatten(message.content) or "",
                }
            )
            continue
        if isinstance(message, UserMessage):
            wire.append({"role": "user", "content": _user_content(message.content)})
            continue
        if isinstance(message, AssistantMessage):
            entry = _assistant_entry(message, reasoning_field)
            if entry is not None:
                wire.append(entry)
    return wire


def _assistant_entry(message: AssistantMessage, reasoning_field: str) -> dict[str, Any] | None:
    """An assistant turn, or None when there is nothing to replay.

    An errored or aborted turn has no content and no calls; sending it would
    be an empty assistant message, which some endpoints reject outright. That
    holds even when it has reasoning: a thought cut off before any answer is
    not a turn the server can take back.
    """
    text_parts = _flatten(message.content)
    calls = [b for b in message.content if isinstance(b, ToolCallBlock)]
    if not text_parts and not calls:
        return None
    entry: dict[str, Any] = {"role": "assistant", "content": text_parts or None}
    if reasoning_field and (reasoning := _plaintext_reasoning(message.content)):
        entry[reasoning_field] = reasoning
    if calls:
        entry["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                # Arguments go back as a JSON *string*, which is how they
                # arrived. Sending the object is a 400 on most endpoints.
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in calls
        ]
    return entry


def _user_content(content: tuple[ContentBlock, ...]) -> Any:
    """A plain string when there are no images, the parts array when there are.

    Endpoints that do not accept images mostly do not accept the parts array
    either, so a text-only turn stays a string.
    """
    if not any(isinstance(b, ImageBlock) for b in content):
        return _flatten(content)
    parts: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextBlock):
            if block.text:
                parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
                }
            )
    return parts


def _flatten(content: tuple[ContentBlock, ...]) -> str:
    """Text blocks joined; everything else dropped.

    Reasoning is not text and goes back separately, if at all — see
    `_plaintext_reasoning`. ProviderBlocks belong to whichever provider
    produced them and have no meaning here.
    """
    return "".join(b.text for b in content if isinstance(b, TextBlock))


def _plaintext_reasoning(content: tuple[ContentBlock, ...]) -> str:
    """The reasoning this dialect produced, exactly as it streamed.

    Exactly, because the server cached it as generated: an edited replay is a
    cache miss at best. Signed blocks are skipped — a signature means Anthropic
    or Responses produced the block for their own replay, and its text is
    another model's thinking (or only a summary of it).
    """
    return "".join(
        b.thinking for b in content if isinstance(b, ThinkingBlock) and b.signature is None
    )


# --- response translation -------------------------------------------------


@dataclass
class _PartialText:
    text: str = ""


@dataclass
class _PartialThinking:
    thinking: str = ""


@dataclass
class _PartialToolCall:
    id: str = ""
    name: str = ""
    #: Arguments arrive as JSON text split across frames, so they accumulate
    #: as a string and are parsed once, at the end.
    json_args: str = ""

    def finish(self) -> ToolCallBlock:
        return ToolCallBlock(
            id=self.id, name=self.name, arguments=_http.tool_arguments(self.json_args)
        )


_Partial = _PartialText | _PartialThinking | _PartialToolCall


class _Accumulator:
    """Assembles one streamed response, emitting StreamEvents as it goes.

    Chat Completions reports no block boundaries — no start, no stop, and no
    index for text. This assigns a `block_index` on first sight of each stream
    and closes every open block at the end, so the event protocol holds even
    though the wire format does not supply it.
    """

    def __init__(self, *, model: Model, provider: str) -> None:
        self._model = model
        self._provider = provider
        self._blocks: dict[int, _Partial] = {}
        self._order: list[int] = []
        self._next_index = 0
        self._text_index: int | None = None
        self._thinking_index: int | None = None
        #: wire tool_calls[].index -> our block_index. Not the same numbering:
        #: the wire counts tool calls, we count blocks of every kind.
        self._tool_indices: dict[int, int] = {}
        self._closed = False
        #: Logged on first sight. Every chunk repeats it.
        self._response_id: str | None = None
        self.usage = Usage()
        self.stop_reason: StopReason = "stop"

    def _open(self, state: _Partial) -> int:
        index = self._next_index
        self._next_index += 1
        self._blocks[index] = state
        self._order.append(index)
        return index

    def feed(self, chunk: dict[str, Any]) -> list[StreamEvent]:
        events: list[StreamEvent] = []

        if self._response_id is None and isinstance(chunk.get("id"), str):
            self._response_id = chunk["id"]
            _log.debug("Response %s from %s", self._response_id, chunk.get("model"))

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            # Arrives in a final chunk whose `choices` is usually empty, which
            # is why usage is read before the choices are looked at.
            self.usage = _to_usage(usage)

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return events
        choice = choices[0]
        if not isinstance(choice, dict):
            return events

        delta = choice.get("delta")
        if isinstance(delta, dict):
            events.extend(self._delta(delta))

        finish = choice.get("finish_reason")
        if isinstance(finish, str):
            # As sent: the mapping folds content_filter into "stop".
            _log.debug("Response finish_reason=%s", finish)
            self.stop_reason = _FINISH_REASONS.get(finish, "stop")
        return events

    def _delta(self, delta: dict[str, Any]) -> list[StreamEvent]:
        events: list[StreamEvent] = []

        for field in _REASONING_FIELDS:
            value = delta.get(field)
            if isinstance(value, str) and value:
                if self._thinking_index is None:
                    self._thinking_index = self._open(_PartialThinking())
                    events.append(ThinkingStart(block_index=self._thinking_index))
                state = self._blocks[self._thinking_index]
                assert isinstance(state, _PartialThinking)
                state.thinking += value
                events.append(ThinkingDelta(block_index=self._thinking_index, delta=value))
                break

        content = delta.get("content")
        if isinstance(content, str) and content:
            if self._text_index is None:
                self._text_index = self._open(_PartialText())
                events.append(TextStart(block_index=self._text_index))
            state = self._blocks[self._text_index]
            assert isinstance(state, _PartialText)
            state.text += content
            events.append(TextDelta(block_index=self._text_index, delta=content))

        calls = delta.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if isinstance(call, dict):
                    events.extend(self._tool_delta(call))
        return events

    def _tool_delta(self, call: dict[str, Any]) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        wire_index = call.get("index")
        if not isinstance(wire_index, int):
            # Some endpoints omit the index when there is only one call.
            wire_index = 0

        block_index = self._tool_indices.get(wire_index)
        if block_index is None:
            block_index = self._open(_PartialToolCall())
            self._tool_indices[wire_index] = block_index
            events.append(ToolCallStart(block_index=block_index))

        state = self._blocks[block_index]
        assert isinstance(state, _PartialToolCall)

        identifier = call.get("id")
        if isinstance(identifier, str) and identifier:
            state.id = identifier
        function = call.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str) and name:
                # Appended, not assigned: a few endpoints split the name across
                # frames the way they split arguments.
                state.name += name if not state.name.endswith(name) else ""
            arguments = function.get("arguments")
            if isinstance(arguments, str) and arguments:
                state.json_args += arguments
                events.append(ToolCallDelta(block_index=block_index, delta=arguments))
        return events

    def close(self) -> list[StreamEvent]:
        """End events for every open block.

        The wire format has no block-stop event, so the end of the stream is
        the only place these can be emitted. Idempotent, because the stream can
        finish either on `[DONE]` or by the body simply ending.
        """
        if self._closed:
            return []
        self._closed = True
        events: list[StreamEvent] = []
        for index in self._order:
            state = self._blocks[index]
            if isinstance(state, _PartialText):
                events.append(TextEnd(block_index=index, text=state.text))
            elif isinstance(state, _PartialThinking):
                events.append(ThinkingEnd(block_index=index, thinking=state.thinking))
            else:
                events.append(ToolCallEnd(block_index=index, tool_call=state.finish()))
        return events

    def message(self) -> AssistantMessage:
        content: list[ContentBlock] = []
        for index in self._order:
            state = self._blocks[index]
            if isinstance(state, _PartialText):
                if state.text:
                    content.append(TextBlock(text=state.text))
            elif isinstance(state, _PartialThinking):
                # No signature: this dialect has no replay token. Whether the
                # text goes back is the profile's `replay_reasoning_field`.
                content.append(ThinkingBlock(thinking=state.thinking))
            else:
                content.append(state.finish())
        return AssistantMessage(
            content=tuple(content),
            model=self._model.id,
            provider=self._provider,
            usage=self.usage,
            stop_reason=self.stop_reason,
        )


def _to_usage(raw: dict[str, Any]) -> Usage:
    details = raw.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else 0
    total_input = raw.get("prompt_tokens") or 0
    cache_read = cached or 0
    return Usage(
        # prompt_tokens includes the cached portion, so subtracting keeps
        # `Usage.total` from counting those tokens twice.
        input=max(total_input - cache_read, 0),
        output=raw.get("completion_tokens") or 0,
        cache_read=cache_read,
    )


__all__ = ["ChatCompletionsProvider"]
