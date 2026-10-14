"""Anthropic Messages adapter.

Translates core `Message` objects to the Messages API wire format and the SDK's
stream back into `StreamEvent`s.

**This module must never raise** for request, model, or runtime failures — see
`harness/llm/provider.py`. Every failure path here ends in a `StreamError`, and
`_failure` is the single place that builds one. The agent loop's near-total lack
of try/except around streaming is paid for right here.

Provider-native content the core does not model — server-side `web_search`
blocks, for instance — rides through as `ProviderBlock` rather than growing a
new core type.
"""

import json
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import anthropic
from anthropic.types import InputJSONDelta as SDKInputJSONDelta
from anthropic.types import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
)
from anthropic.types import SignatureDelta as SDKSignatureDelta
from anthropic.types import TextBlock as SDKTextBlock
from anthropic.types import TextDelta as SDKTextDelta
from anthropic.types import ThinkingBlock as SDKThinkingBlock
from anthropic.types import ThinkingDelta as SDKThinkingDelta
from anthropic.types import ToolUseBlock as SDKToolUseBlock

from harness.config import AnthropicConfig
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
from harness.llm.providers._trace import request_shape
from harness.llm.types import (
    AssistantMessage,
    ContentBlock,
    Context,
    ImageBlock,
    Message,
    Model,
    ProviderBlock,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
)

#: Messages API stop_reason -> our StopReason. Anything unlisted means "stop".
_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_use",
    "pause_turn": "stop",
    "refusal": "stop",
}

_ABORTED_NOTICE = "Operation aborted"

_log = logging.getLogger(__name__)


class AnthropicProvider:
    """Streams from the Anthropic Messages API.

    Credentials come from the environment, never from config files or
    arguments — see docs/contributing.md. The env var *names* are configurable;
    their values are not committed anywhere.
    """

    name = "anthropic"

    def __init__(
        self,
        config: AnthropicConfig | None = None,
        *,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self._config = config if config is not None else AnthropicConfig()
        self._client = client
        self._client_error: str | None = None
        if client is None:
            try:
                self._client = self._build_client()
            except Exception as exc:
                # Deferred, not raised: a missing key is a request failure like
                # any other, and the contract says the stream reports it.
                self._client_error = str(exc)

    def _build_client(self) -> anthropic.AsyncAnthropic:
        cfg = self._config
        token = os.environ.get(cfg.auth_token_env)
        key = os.environ.get(cfg.api_key_env)
        if not token and not key:
            raise RuntimeError(
                f"No Anthropic credentials. Set {cfg.api_key_env} or {cfg.auth_token_env}."
            )
        kwargs: dict[str, object] = {}
        if token:
            # auth_token wins: a gateway bearer token and a raw key can both be
            # present, and the token is the more specific intent.
            kwargs["auth_token"] = token
        else:
            kwargs["api_key"] = key
        # The variable's name, never its value. Same for base_url, which is a
        # private endpoint often enough to treat as sensitive.
        _log.debug("Credentials taken from %s", cfg.auth_token_env if token else cfg.api_key_env)
        base_url = os.environ.get(cfg.base_url_env)
        if base_url:
            _log.debug("Endpoint overridden by %s", cfg.base_url_env)
            kwargs["base_url"] = base_url
        return anthropic.AsyncAnthropic(**kwargs)  # type: ignore[arg-type]

    async def stream(
        self,
        model: Model,
        ctx: Context,
        opts: StreamOptions | None = None,
    ) -> AsyncIterator[StreamEvent]:
        options = opts if opts is not None else StreamOptions()

        if self._client is None:
            yield _failure(model, self.name, self._client_error or "Client unavailable")
            return
        if _aborted(options):
            yield _failure(model, self.name, _ABORTED_NOTICE, reason="aborted")
            return

        try:
            request = self._build_request(model, ctx, options)
        except Exception as exc:
            # Translation bugs surface as an error turn rather than taking the
            # run down. The loop cannot repair them either way.
            yield _failure(model, self.name, f"Could not build request: {exc}")
            return

        # Shape, not content — the transcript already holds the conversation.
        _log.debug("Request to %s: %s", model.id, request_shape(request))

        accumulator = _Accumulator(model=model, provider=self.name)
        try:
            async with self._client.messages.stream(**request) as stream:
                yield StreamStart()
                async for event in stream:
                    if _aborted(options):
                        yield _failure(model, self.name, _ABORTED_NOTICE, reason="aborted")
                        return
                    for translated in accumulator.feed(event):
                        yield translated
        except anthropic.APIError as exc:
            yield _failure(model, self.name, _describe(exc))
            return
        except Exception as exc:
            yield _failure(model, self.name, f"{type(exc).__name__}: {exc}")
            return

        yield StreamDone(message=accumulator.message(), reason=accumulator.stop_reason)

    def _build_request(self, model: Model, ctx: Context, opts: StreamOptions) -> dict[str, Any]:
        # dict[str, Any] rather than a typed kwargs mapping: this is a wire
        # payload assembled key by key and splatted into the SDK, which types
        # each parameter individually. Narrowing it here buys nothing the
        # SDK's own validation does not already do.
        request: dict[str, Any] = {
            "model": model.id,
            "max_tokens": opts.max_tokens or model.max_tokens or self._config.default_max_tokens,
            "messages": _to_wire_messages(ctx.messages),
        }
        if ctx.system_prompt:
            request["system"] = ctx.system_prompt
        if ctx.tools:
            request["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in ctx.tools
            ]
        if opts.temperature is not None:
            request["temperature"] = opts.temperature
        if model.supports_reasoning:
            thinking = self._thinking(model, opts)
            if thinking is not None:
                request["thinking"] = thinking
                # The API rejects temperature alongside thinking.
                request.pop("temperature", None)
            if opts.reasoning is not None and not self._takes_budget(model):
                # Adaptive thinking has no budget; effort is how deep it goes.
                request["output_config"] = {"effort": opts.reasoning}
        return request

    def _thinking(self, model: Model, opts: StreamOptions) -> dict[str, Any] | None:
        """The `thinking` field for a reasoning model, or None to send none.

        Two request shapes, per model. Adaptive is sent whether or not
        reasoning was requested, because the models that take it think by
        default anyway, and only an explicit request can ask for `display`.
        """
        if self._takes_budget(model):
            if opts.reasoning is None:
                return None
            return {"type": "enabled", "budget_tokens": self._config.thinking_budget_tokens}
        thinking: dict[str, Any] = {"type": "adaptive"}
        if self._config.thinking_display:
            thinking["display"] = self._config.thinking_display
        return thinking

    def _takes_budget(self, model: Model) -> bool:
        return model.id.startswith(self._config.budget_thinking_models)


# --- request translation --------------------------------------------------


def _to_wire_messages(messages: tuple[Message, ...]) -> list[dict[str, object]]:
    """Core messages -> Messages API `messages`.

    System messages are dropped: the API carries the prompt in a top-level
    `system` field, which `_build_request` fills from `Context.system_prompt`.

    Consecutive tool results are merged into one user message because the API
    requires every `tool_result` for a turn to arrive in a single user turn —
    the loop emits them as separate messages, so the merge happens here.
    """
    wire: list[dict[str, object]] = []
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "tool_result":
            block = _tool_result_block(message)
            if wire and wire[-1]["role"] == "user" and _is_tool_result_turn(wire[-1]):
                content = wire[-1]["content"]
                assert isinstance(content, list)
                content.append(block)
            else:
                wire.append({"role": "user", "content": [block]})
            continue
        content = [b for b in (_to_wire_block(block) for block in message.content) if b is not None]
        if not content:
            # The API rejects empty content. An assistant turn with none is an
            # error or abort placeholder, which has no place in the next request.
            continue
        wire.append(
            {
                "role": "user" if isinstance(message, UserMessage) else "assistant",
                "content": content,
            }
        )
    return wire


def _is_tool_result_turn(entry: dict[str, object]) -> bool:
    content = entry.get("content")
    return (
        isinstance(content, list)
        and len(content) > 0
        and isinstance(content[0], dict)
        and content[0].get("type") == "tool_result"
    )


def _tool_result_block(message: ToolResultMessage) -> dict[str, object]:
    content = [b for b in (_to_wire_block(block) for block in message.content) if b is not None]
    return {
        "type": "tool_result",
        "tool_use_id": message.tool_call_id,
        "content": content or [{"type": "text", "text": ""}],
        "is_error": message.is_error,
    }


def _to_wire_block(block: ContentBlock) -> dict[str, object] | None:
    """Core block -> wire block. None means 'omit from the request'."""
    if isinstance(block, TextBlock):
        # The API rejects empty text blocks.
        return {"type": "text", "text": block.text} if block.text else None
    if isinstance(block, ThinkingBlock):
        # Without the signature the API rejects the replayed block, so a
        # signature-less thinking block is dropped rather than failing the turn.
        if block.signature is None:
            return None
        return {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
    if isinstance(block, ImageBlock):
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": block.mime_type, "data": block.data},
        }
    if isinstance(block, ToolCallBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.arguments,
        }
    if isinstance(block, ProviderBlock):
        # Round-trip provider-native content verbatim. This is the whole point
        # of ProviderBlock: server-tool blocks must come back unchanged.
        return {"type": block.provider_type, **block.data}
    return None


# --- response translation -------------------------------------------------


@dataclass
class _PartialText:
    text: str = ""


@dataclass
class _PartialThinking:
    thinking: str = ""
    signature: str | None = None


@dataclass
class _PartialToolCall:
    id: str
    name: str
    json: str = ""

    def finish(self) -> ToolCallBlock:
        try:
            arguments = json.loads(self.json) if self.json.strip() else {}
        except json.JSONDecodeError:
            # Never salvage partial JSON. A truncated object can parse into
            # arguments that validate while silently missing fields; the loop
            # already refuses to execute calls from a truncated response, and
            # handing it a half-built dict would defeat that.
            arguments = {}
        return ToolCallBlock(
            id=self.id,
            name=self.name,
            arguments=arguments if isinstance(arguments, dict) else {},
        )


@dataclass
class _PartialProvider:
    provider_type: str
    data: dict[str, object]


_Partial = _PartialText | _PartialThinking | _PartialToolCall | _PartialProvider


class _Accumulator:
    """Assembles one streamed response, emitting StreamEvents as it goes.

    The SDK reports deltas; the loop needs both deltas and a final assembled
    `AssistantMessage`. This holds the partial state between the two.

    Per-kind dataclasses rather than a dict of `object`: the branches below
    narrow by type, so a field that does not exist on a block is a type error
    here rather than a KeyError in front of a user.
    """

    def __init__(self, *, model: Model, provider: str) -> None:
        self._model = model
        self._provider = provider
        self._blocks: dict[int, _Partial] = {}
        self._order: list[int] = []
        self.usage = Usage()
        self.stop_reason: StopReason = "stop"

    def feed(self, event: object) -> list[StreamEvent]:
        if isinstance(event, RawMessageStartEvent):
            # The id is what Anthropic support asks for; the model is the one
            # that answered, which an alias or a gateway can make different.
            _log.debug("Response %s from %s", event.message.id, event.message.model)
            self.usage = _to_usage(event.message.usage)
            return []
        if isinstance(event, RawContentBlockStartEvent):
            return self._start_block(event)
        if isinstance(event, RawContentBlockDeltaEvent):
            return self._delta(event)
        if isinstance(event, RawContentBlockStopEvent):
            return self._stop_block(event.index)
        if isinstance(event, RawMessageDeltaEvent):
            if event.delta.stop_reason:
                # As sent: the mapping folds refusal and pause_turn into "stop".
                _log.debug("Response stop_reason=%s", event.delta.stop_reason)
                self.stop_reason = _STOP_REASONS.get(event.delta.stop_reason, "stop")
            # The final usage arrives here; input counts came with message_start.
            self.usage = Usage(
                input=self.usage.input,
                output=event.usage.output_tokens,
                cache_read=self.usage.cache_read,
                cache_write=self.usage.cache_write,
            )
            return []
        return []

    def _start_block(self, event: RawContentBlockStartEvent) -> list[StreamEvent]:
        index = event.index
        block = event.content_block
        self._order.append(index)

        if isinstance(block, SDKTextBlock):
            self._blocks[index] = _PartialText()
            return [TextStart(block_index=index)]
        if isinstance(block, SDKThinkingBlock):
            self._blocks[index] = _PartialThinking()
            return [ThinkingStart(block_index=index)]
        if isinstance(block, SDKToolUseBlock):
            self._blocks[index] = _PartialToolCall(id=block.id, name=block.name)
            return [ToolCallStart(block_index=index)]
        # Server-tool results and anything else the core does not model. Kept
        # verbatim so the next request can replay it.
        _log.debug("Keeping a %s block verbatim", block.type)
        self._blocks[index] = _PartialProvider(provider_type=block.type, data=_dump(block))
        return []

    def _delta(self, event: RawContentBlockDeltaEvent) -> list[StreamEvent]:
        index = event.index
        state = self._blocks.get(index)
        delta = event.delta

        if isinstance(delta, SDKTextDelta) and isinstance(state, _PartialText):
            state.text += delta.text
            return [TextDelta(block_index=index, delta=delta.text)]
        if isinstance(delta, SDKThinkingDelta) and isinstance(state, _PartialThinking):
            state.thinking += delta.thinking
            return [ThinkingDelta(block_index=index, delta=delta.thinking)]
        if isinstance(delta, SDKSignatureDelta) and isinstance(state, _PartialThinking):
            state.signature = (state.signature or "") + delta.signature
            return []
        if isinstance(delta, SDKInputJSONDelta) and isinstance(state, _PartialToolCall):
            state.json += delta.partial_json
            return [ToolCallDelta(block_index=index, delta=delta.partial_json)]
        return []

    def _stop_block(self, index: int) -> list[StreamEvent]:
        state = self._blocks.get(index)
        if isinstance(state, _PartialText):
            return [TextEnd(block_index=index, text=state.text)]
        if isinstance(state, _PartialThinking):
            return [ThinkingEnd(block_index=index, thinking=state.thinking)]
        if isinstance(state, _PartialToolCall):
            return [ToolCallEnd(block_index=index, tool_call=state.finish())]
        return []

    def message(self) -> AssistantMessage:
        content: list[ContentBlock] = []
        for index in self._order:
            state = self._blocks[index]
            if isinstance(state, _PartialText):
                if state.text:
                    content.append(TextBlock(text=state.text))
            elif isinstance(state, _PartialThinking):
                content.append(ThinkingBlock(thinking=state.thinking, signature=state.signature))
            elif isinstance(state, _PartialToolCall):
                content.append(state.finish())
            else:
                content.append(ProviderBlock(provider_type=state.provider_type, data=state.data))
        return AssistantMessage(
            content=tuple(content),
            model=self._model.id,
            provider=self._provider,
            usage=self.usage,
            stop_reason=self.stop_reason,
        )


def _to_usage(raw: object) -> Usage:
    return Usage(
        input=getattr(raw, "input_tokens", 0) or 0,
        output=getattr(raw, "output_tokens", 0) or 0,
        cache_read=getattr(raw, "cache_read_input_tokens", 0) or 0,
        cache_write=getattr(raw, "cache_creation_input_tokens", 0) or 0,
    )


def _dump(block: object) -> dict[str, object]:
    """Best-effort plain-dict view of an SDK block, minus its `type` tag.

    `type` is dropped because `ProviderBlock` carries it separately in
    `provider_type`; re-adding it on the way out is `_to_wire_block`'s job.
    """
    dump = getattr(block, "model_dump", None)
    data = dump(mode="json") if callable(dump) else {}
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if k != "type"}
    return {}


def _describe(exc: anthropic.APIError) -> str:
    """A message the user can act on, without leaking credentials.

    SDK exception strings can embed request details; the status and the API's
    own message are the parts worth surfacing.
    """
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    detail = ""
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            detail = str(error.get("message", ""))
    if not detail:
        detail = str(exc)
    return f"{type(exc).__name__}{f' ({status})' if status else ''}: {detail}"


def _aborted(opts: StreamOptions) -> bool:
    return opts.signal is not None and opts.signal.is_set()


def _failure(
    model: Model,
    provider: str,
    error: str,
    *,
    reason: str = "error",
) -> StreamError:
    """The one place a failure becomes a stream event. Nothing here raises."""
    if reason == "aborted":
        _log.debug("Stream aborted for %s", model.id)
    else:
        _log.warning("%s request failed for %s: %s", provider, model.id, error)
    return StreamError(
        message=AssistantMessage(
            content=(),
            model=model.id,
            provider=provider,
            stop_reason="aborted" if reason == "aborted" else "error",
            error_message=error,
        ),
        reason="aborted" if reason == "aborted" else "error",
    )
