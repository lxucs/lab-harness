"""The OpenAI Responses protocol.

OpenAI's current API, and the only one of the three this harness speaks that
can carry a reasoning model's chain of thought across a tool call. A profile
opts in with `api = "responses"`.

Three things make it a different shape from Chat Completions, not a tweak of it:

- The conversation is a list of **items** (`message`, `function_call`,
  `function_call_output`, `reasoning`) rather than messages with roles, so a
  tool call and its result are siblings instead of nested in message bodies.
- The system prompt is top-level `instructions`, as in the Messages API.
- Reasoning is a first-class item that comes back **encrypted** and must be
  replayed verbatim. That is what `ThinkingBlock.signature` is for.

On reasoning specifically — this is the case `chat_completions.py` cannot
cover. A reasoning model thinks, calls a tool, and then has to keep thinking
from where it left off. OpenAI does not return the raw chain; it returns

  * a human-readable **summary** (`reasoning.summary`), which we put in
    `ThinkingBlock.thinking` so a terminal can render it, and
  * an opaque **`encrypted_content`** blob, which we pack into
    `ThinkingBlock.signature` and hand straight back on the next request.

We ask for the encrypted form and set `store: false` because this harness owns
its transcript. The alternative — letting OpenAI store the conversation and
referring to it by `previous_response_id` — would put the authoritative history
on someone else's server, where `--resume` and the session files could not see
it.

**This module must never raise** for request, model, or runtime failures — see
`lab_harness/llm/provider.py`.
"""

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
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

#: `response.incomplete` reasons -> our StopReason. A response that ran out of
#: room is `length`; anything else that stopped early is a plain stop.
_INCOMPLETE_REASONS: dict[str, StopReason] = {
    "max_output_tokens": "length",
    "content_filter": "stop",
}

_log = logging.getLogger(__name__)


class ResponsesProvider:
    """Streams from the OpenAI Responses API.

    `name` is the *profile* name, so a transcript records which endpoint
    answered rather than the protocol it spoke.
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
            yield _http.failure(model, self.name, f"Could not build request: {exc}")
            return

        url = _http.endpoint(cfg, "responses")

        _log.debug("Request to %s via %s: %s", model.id, self.name, request_shape(request))

        accumulator = _Accumulator(model=model, provider=self.name)
        try:
            async with (
                _http.client(self._client, cfg) as client,
                client.stream(
                    "POST", url, json=request, headers=_http.headers(_http.credential(cfg))
                ) as response,
            ):
                if response.status_code >= 400:
                    yield _http.failure(
                        model, self.name, _http.describe(response, await response.aread())
                    )
                    return
                yield StreamStart()
                frames = _http.Frames()
                async for line in response.aiter_lines():
                    if _http.aborted(options):
                        yield _http.failure(
                            model, self.name, _http.ABORTED_NOTICE, reason="aborted"
                        )
                        return
                    payload = frames.parse(line)
                    if payload is None:
                        continue
                    if isinstance(payload, _http.Done):
                        break
                    if (fatal := _fatal(payload)) is not None:
                        yield _http.failure(model, self.name, fatal)
                        return
                    for translated in accumulator.feed(payload):
                        yield translated
                if (error := frames.not_a_stream(response)) is not None:
                    yield _http.failure(model, self.name, error)
                    return
        except Exception as exc:
            yield _http.failure(model, self.name, f"{type(exc).__name__}: {exc}")
            return

        for closing in accumulator.close():
            yield closing
        yield StreamDone(message=accumulator.message(), reason=accumulator.stop_reason)

    def _build_request(self, model: Model, ctx: Context, opts: StreamOptions) -> dict[str, Any]:
        cfg = self._config
        request: dict[str, Any] = {
            "model": model.id,
            "input": _to_wire_items(ctx.messages),
            "stream": True,
            "max_output_tokens": (opts.max_tokens or model.max_tokens or cfg.default_max_tokens),
            # The harness owns the transcript; see the module docstring.
            "store": cfg.store,
        }
        if ctx.system_prompt:
            request["instructions"] = ctx.system_prompt
        if ctx.tools:
            # Flat, unlike Chat Completions, which nests all of this under a
            # "function" key.
            request["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in ctx.tools
            ]
        if opts.temperature is not None:
            request["temperature"] = opts.temperature
        if model.supports_reasoning:
            reasoning: dict[str, Any] = {}
            if opts.reasoning is not None:
                reasoning["effort"] = opts.reasoning
            if cfg.reasoning_summary:
                # Without this the thinking is invisible: the raw chain is
                # never returned, and the summary is the only readable form.
                reasoning["summary"] = cfg.reasoning_summary
            if reasoning:
                request["reasoning"] = reasoning
            if not cfg.store:
                # Only meaningful when the server is not storing the thread —
                # otherwise the chain is referenced by id instead.
                request["include"] = ["reasoning.encrypted_content"]
        return request


def _fatal(payload: dict[str, Any]) -> str | None:
    """An error reported *inside* the stream rather than by HTTP status.

    A Responses stream can begin 200 OK and then fail — a content filter, a
    model error mid-generation. Those arrive as events, and the contract says
    they end the turn rather than being silently dropped.
    """
    kind = payload.get("type")
    if kind == "error":
        message = payload.get("message")
        return str(message) if message else "Stream reported an error"
    if kind == "response.failed":
        response = payload.get("response")
        if isinstance(response, dict):
            error = response.get("error")
            if isinstance(error, dict):
                return str(error.get("message", "Response failed"))
        return "Response failed"
    return None


# --- request translation --------------------------------------------------


def _to_wire_items(messages: tuple[Message, ...]) -> list[dict[str, Any]]:
    """Core messages -> Responses `input` items.

    One message can become several items: an assistant turn that thought and
    then called two tools is a reasoning item plus two function_call items,
    all siblings. That flattening is the main difference from the other two
    adapters, and it is why reasoning can be replayed at all — a reasoning
    item is addressable rather than buried inside a message body.
    """
    items: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            # Carried as top-level `instructions`.
            continue
        if isinstance(message, ToolResultMessage):
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": _flatten(message.content) or "",
                }
            )
            continue
        if isinstance(message, UserMessage):
            items.append({"type": "message", "role": "user", "content": _user_content(message)})
            continue
        if isinstance(message, AssistantMessage):
            items.extend(_assistant_items(message))
    return items


def _assistant_items(message: AssistantMessage) -> list[dict[str, Any]]:
    """One assistant turn, flattened into its items.

    Reasoning comes first, because the API requires a reasoning item to
    precede the output it produced.
    """
    items: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, ThinkingBlock):
            item = _reasoning_item(block)
            if item is not None:
                items.append(item)
    body = _flatten(message.content)
    if body:
        items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": body}],
            }
        )
    for block in message.content:
        if isinstance(block, ToolCallBlock):
            items.append(
                {
                    "type": "function_call",
                    "call_id": block.id,
                    "name": block.name,
                    # A JSON *string*, the way it arrived.
                    "arguments": json.dumps(block.arguments),
                }
            )
    return items


def _reasoning_item(block: ThinkingBlock) -> dict[str, Any] | None:
    """Rebuild the reasoning item from the token we stashed in `signature`.

    A block with no signature is dropped rather than reconstructed. That
    covers two real cases: a resumed session whose thinking came from DeepSeek
    (plaintext, no replay token) and a response fetched without
    `encrypted_content`. Sending a reasoning item the server cannot decrypt is
    a 400; dropping it only costs the model some context.
    """
    if block.signature is None:
        return None
    try:
        token = json.loads(block.signature)
    except json.JSONDecodeError:
        return None
    if not isinstance(token, dict) or "id" not in token:
        return None
    item: dict[str, Any] = {"type": "reasoning", "id": token["id"]}
    if token.get("encrypted_content"):
        item["encrypted_content"] = token["encrypted_content"]
    # The summary is echoed back in the shape it arrived in. An empty list is
    # valid and means "it had no summary", which is different from omitting it.
    item["summary"] = [{"type": "summary_text", "text": block.thinking}] if block.thinking else []
    return item


def _user_content(message: UserMessage) -> list[dict[str, Any]]:
    """Always the parts array — this protocol has no bare-string shorthand."""
    parts: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            if block.text:
                parts.append({"type": "input_text", "text": block.text})
        elif isinstance(block, ImageBlock):
            parts.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{block.mime_type};base64,{block.data}",
                }
            )
    return parts or [{"type": "input_text", "text": ""}]


def _flatten(content: tuple[ContentBlock, ...]) -> str:
    """Text blocks joined. Thinking is handled separately, as its own item."""
    return "".join(b.text for b in content if isinstance(b, TextBlock))


# --- response translation -------------------------------------------------


@dataclass
class _PartialText:
    text: str = ""


@dataclass
class _PartialThinking:
    thinking: str = ""
    item_id: str = ""
    encrypted: str = ""

    def signature(self) -> str | None:
        """The replay token, packed into the one opaque field the core has.

        `ThinkingBlock.signature` is documented as an opaque provider replay
        token, and that is exactly what this is — the core never looks inside.
        Packing the id alongside the ciphertext keeps the reasoning item
        reconstructable without an OpenAI-shaped field in `llm/types.py`.
        """
        if not self.item_id:
            return None
        return json.dumps({"id": self.item_id, "encrypted_content": self.encrypted})


@dataclass
class _PartialToolCall:
    id: str = ""
    name: str = ""
    json_args: str = ""

    def finish(self) -> ToolCallBlock:
        return ToolCallBlock(
            id=self.id, name=self.name, arguments=_http.tool_arguments(self.json_args)
        )


_Partial = _PartialText | _PartialThinking | _PartialToolCall


@dataclass
class _Slot:
    """One output item, and the block index we gave it."""

    state: _Partial
    block_index: int = 0
    opened: bool = False
    closed: bool = False
    pending: list[StreamEvent] = field(default_factory=list)


class _Accumulator:
    """Assembles one streamed response.

    Keyed by the API's `output_index`, which is stable for the life of a
    response — unlike Chat Completions, this protocol does number its items,
    so the block indices here mirror the server's rather than being invented.
    """

    def __init__(self, *, model: Model, provider: str) -> None:
        self._model = model
        self._provider = provider
        self._slots: dict[int, _Slot] = {}
        self._order: list[int] = []
        self.usage = Usage()
        self.stop_reason: StopReason = "stop"

    def feed(self, payload: dict[str, Any]) -> list[StreamEvent]:
        kind = payload.get("type")
        if not isinstance(kind, str):
            return []

        if kind == "response.created":
            # The id is what OpenAI support asks for; the model is the one
            # that answered, which an alias can make different.
            if isinstance(created := payload.get("response"), dict):
                _log.debug("Response %s from %s", created.get("id"), created.get("model"))
            return []
        if kind == "response.output_item.added":
            return self._item_added(payload)
        if kind == "response.output_text.delta":
            return self._text_delta(payload)
        if kind == "response.reasoning_summary_text.delta":
            return self._thinking_delta(payload)
        if kind == "response.function_call_arguments.delta":
            return self._tool_delta(payload)
        if kind == "response.output_item.done":
            return self._item_done(payload)
        if kind in ("response.completed", "response.incomplete"):
            return self._completed(payload, incomplete=kind == "response.incomplete")
        return []

    def _slot(self, payload: dict[str, Any]) -> _Slot | None:
        index = payload.get("output_index")
        return self._slots.get(index) if isinstance(index, int) else None

    def _item_added(self, payload: dict[str, Any]) -> list[StreamEvent]:
        index = payload.get("output_index")
        item = payload.get("item")
        if not isinstance(index, int) or not isinstance(item, dict):
            return []
        kind = item.get("type")
        state: _Partial
        if kind == "reasoning":
            state = _PartialThinking(item_id=str(item.get("id") or ""))
        elif kind == "function_call":
            state = _PartialToolCall(
                id=str(item.get("call_id") or ""), name=str(item.get("name") or "")
            )
        elif kind == "message":
            state = _PartialText()
        else:
            # An item type the core does not model. Ignored rather than
            # guessed at; the turn's text and calls are unaffected.
            _log.debug("Ignoring unknown output item type %r", kind)
            return []
        slot = _Slot(state=state, block_index=len(self._order))
        self._slots[index] = slot
        self._order.append(index)
        return self._open(slot)

    def _open(self, slot: _Slot) -> list[StreamEvent]:
        """Emit the block's start event, once.

        Reasoning items are announced on arrival rather than on their first
        delta, because a reasoning item with no summary never produces one and
        would otherwise never be announced at all.
        """
        if slot.opened:
            return []
        slot.opened = True
        if isinstance(slot.state, _PartialText):
            return [TextStart(block_index=slot.block_index)]
        if isinstance(slot.state, _PartialThinking):
            return [ThinkingStart(block_index=slot.block_index)]
        return [ToolCallStart(block_index=slot.block_index)]

    def _text_delta(self, payload: dict[str, Any]) -> list[StreamEvent]:
        slot = self._slot(payload)
        value = payload.get("delta")
        if slot is None or not isinstance(value, str) or not value:
            return []
        if not isinstance(slot.state, _PartialText):
            return []
        slot.state.text += value
        return [TextDelta(block_index=slot.block_index, delta=value)]

    def _thinking_delta(self, payload: dict[str, Any]) -> list[StreamEvent]:
        slot = self._slot(payload)
        value = payload.get("delta")
        if slot is None or not isinstance(value, str) or not value:
            return []
        if not isinstance(slot.state, _PartialThinking):
            return []
        slot.state.thinking += value
        return [ThinkingDelta(block_index=slot.block_index, delta=value)]

    def _tool_delta(self, payload: dict[str, Any]) -> list[StreamEvent]:
        slot = self._slot(payload)
        value = payload.get("delta")
        if slot is None or not isinstance(value, str) or not value:
            return []
        if not isinstance(slot.state, _PartialToolCall):
            return []
        slot.state.json_args += value
        return [ToolCallDelta(block_index=slot.block_index, delta=value)]

    def _item_done(self, payload: dict[str, Any]) -> list[StreamEvent]:
        """The completed item, which is where the encrypted blob arrives.

        `encrypted_content` is never streamed in pieces — it shows up whole,
        here, after the summary deltas. Without reading this event the replay
        token would be lost and every tool call would restart the model's
        reasoning from nothing.
        """
        slot = self._slot(payload)
        item = payload.get("item")
        if slot is None or not isinstance(item, dict):
            return []
        if isinstance(slot.state, _PartialThinking):
            encrypted = item.get("encrypted_content")
            if isinstance(encrypted, str):
                slot.state.encrypted = encrypted
            identifier = item.get("id")
            if isinstance(identifier, str) and identifier:
                slot.state.item_id = identifier
            if not slot.state.thinking:
                # Some configurations return a summary only on the final item.
                slot.state.thinking = _summary_text(item)
        elif isinstance(slot.state, _PartialToolCall):
            # Belt and braces: the final item carries the authoritative
            # arguments, and a stream that dropped a delta would otherwise
            # hand the model a truncated object.
            arguments = item.get("arguments")
            if isinstance(arguments, str) and arguments:
                slot.state.json_args = arguments
            call_id = item.get("call_id")
            if isinstance(call_id, str) and call_id:
                slot.state.id = call_id
        return []

    def _completed(self, payload: dict[str, Any], *, incomplete: bool) -> list[StreamEvent]:
        response = payload.get("response")
        if not isinstance(response, dict):
            return []
        usage = response.get("usage")
        if isinstance(usage, dict):
            self.usage = _to_usage(usage)
        details = response.get("incomplete_details")
        reason = details.get("reason") if isinstance(details, dict) else None
        # As sent: the mapping folds content_filter into "stop".
        _log.debug("Response status=%s reason=%s", response.get("status"), reason)
        if incomplete:
            self.stop_reason = _INCOMPLETE_REASONS.get(str(reason), "stop")
        elif any(isinstance(s.state, _PartialToolCall) for s in self._slots.values()):
            # The protocol has no "stop because tools" status; the presence of
            # a function_call item is what says so.
            self.stop_reason = "tool_use"
        return []

    def close(self) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        for index in self._order:
            slot = self._slots[index]
            if slot.closed:
                continue
            slot.closed = True
            events.extend(self._open(slot))
            if isinstance(slot.state, _PartialText):
                events.append(TextEnd(block_index=slot.block_index, text=slot.state.text))
            elif isinstance(slot.state, _PartialThinking):
                events.append(
                    ThinkingEnd(block_index=slot.block_index, thinking=slot.state.thinking)
                )
            else:
                events.append(
                    ToolCallEnd(block_index=slot.block_index, tool_call=slot.state.finish())
                )
        return events

    def message(self) -> AssistantMessage:
        content: list[ContentBlock] = []
        for index in self._order:
            state = self._slots[index].state
            if isinstance(state, _PartialText):
                if state.text:
                    content.append(TextBlock(text=state.text))
            elif isinstance(state, _PartialThinking):
                # Kept even when the summary is empty: the signature is the
                # part that matters, and dropping the block would lose the
                # model's place between tool calls.
                content.append(ThinkingBlock(thinking=state.thinking, signature=state.signature()))
            else:
                content.append(state.finish())
        return AssistantMessage(
            content=tuple(content),
            model=self._model.id,
            provider=self._provider,
            usage=self.usage,
            stop_reason=self.stop_reason,
        )


def _summary_text(item: dict[str, Any]) -> str:
    summary = item.get("summary")
    if not isinstance(summary, list):
        return ""
    return "".join(str(part.get("text", "")) for part in summary if isinstance(part, dict))


def _to_usage(raw: dict[str, Any]) -> Usage:
    """Responses names its token counts differently from Chat Completions."""
    input_details = raw.get("input_tokens_details")
    cached = input_details.get("cached_tokens") if isinstance(input_details, dict) else 0
    cache_read = cached or 0
    total_input = raw.get("input_tokens") or 0
    return Usage(
        # input_tokens includes the cached portion; subtracting keeps
        # `Usage.total` from counting those tokens twice.
        input=max(total_input - cache_read, 0),
        # output_tokens already includes reasoning tokens. They are reported
        # separately in output_tokens_details.reasoning_tokens, which the core
        # `Usage` has no field for — the total stays correct either way.
        output=raw.get("output_tokens") or 0,
        cache_read=cache_read,
    )


__all__ = ["ResponsesProvider"]
