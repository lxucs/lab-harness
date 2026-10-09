"""Tagged-union serialization.

`dataclasses.asdict` handles nesting and gives plain dicts. Deserialization
dispatches on the `type` / `role` discriminator each type carries.
"""

from dataclasses import asdict, is_dataclass
from typing import Any

from harness.llm import events as _events
from harness.llm import types as _types

_BLOCKS: dict[str, type] = {
    "text": _types.TextBlock,
    "thinking": _types.ThinkingBlock,
    "image": _types.ImageBlock,
    "tool_call": _types.ToolCallBlock,
    "provider": _types.ProviderBlock,
}

_MESSAGES: dict[str, type] = {
    "system": _types.SystemMessage,
    "user": _types.UserMessage,
    "assistant": _types.AssistantMessage,
    "tool_result": _types.ToolResultMessage,
}

_EVENTS: dict[str, type] = {
    "start": _events.StreamStart,
    "text_start": _events.TextStart,
    "text_delta": _events.TextDelta,
    "text_end": _events.TextEnd,
    "thinking_start": _events.ThinkingStart,
    "thinking_delta": _events.ThinkingDelta,
    "thinking_end": _events.ThinkingEnd,
    "toolcall_start": _events.ToolCallStart,
    "toolcall_delta": _events.ToolCallDelta,
    "toolcall_end": _events.ToolCallEnd,
    "done": _events.StreamDone,
    "error": _events.StreamError,
}


def to_dict(obj: Any) -> dict[str, Any]:
    """Convert a protocol dataclass to a JSON-safe dict.

    Raises TypeError for non-dataclasses, which is how the round-trip test
    catches a live object sneaking onto an event.
    """
    if not is_dataclass(obj) or isinstance(obj, type):
        raise TypeError(f"Not a protocol dataclass: {type(obj).__name__}")
    return asdict(obj)


def block_from_dict(data: dict[str, Any]) -> _types.ContentBlock:
    payload = dict(data)
    tag = payload.pop("type")
    cls = _BLOCKS.get(tag)
    if cls is None:
        raise ValueError(f"Unknown content block type: {tag!r}")
    return cls(**payload)  # type: ignore[return-value]


def message_from_dict(data: dict[str, Any]) -> _types.Message:
    payload = dict(data)
    tag = payload.pop("role")
    cls = _MESSAGES.get(tag)
    if cls is None:
        raise ValueError(f"Unknown message role: {tag!r}")
    if isinstance(payload.get("content"), list):
        payload["content"] = tuple(block_from_dict(b) for b in payload["content"])
    if "usage" in payload and isinstance(payload["usage"], dict):
        payload["usage"] = _types.Usage(**payload["usage"])
    return cls(**payload)  # type: ignore[return-value]


def event_from_dict(data: dict[str, Any]) -> _events.StreamEvent:
    payload = dict(data)
    tag = payload.pop("type")
    cls = _EVENTS.get(tag)
    if cls is None:
        raise ValueError(f"Unknown stream event type: {tag!r}")
    if isinstance(payload.get("message"), dict):
        payload["message"] = message_from_dict(payload["message"])
    if isinstance(payload.get("tool_call"), dict):
        payload["tool_call"] = block_from_dict(payload["tool_call"])
    return cls(**payload)  # type: ignore[return-value]
