"""Provider stream protocol.

    start -> {text,thinking,toolcall}_{start,delta,end}* -> done | error

`start` always precedes updates. A stream may terminate directly with `error`
when request setup fails. Updates and `done` never appear before `start`.
"""

from dataclasses import dataclass
from typing import Literal

from harness.llm.types import AssistantMessage, StopReason, ToolCallBlock


@dataclass(frozen=True, slots=True)
class StreamStart:
    type: Literal["start"] = "start"


@dataclass(frozen=True, slots=True)
class TextStart:
    block_index: int
    type: Literal["text_start"] = "text_start"


@dataclass(frozen=True, slots=True)
class TextDelta:
    block_index: int
    delta: str
    type: Literal["text_delta"] = "text_delta"


@dataclass(frozen=True, slots=True)
class TextEnd:
    block_index: int
    text: str
    type: Literal["text_end"] = "text_end"


@dataclass(frozen=True, slots=True)
class ThinkingStart:
    block_index: int
    type: Literal["thinking_start"] = "thinking_start"


@dataclass(frozen=True, slots=True)
class ThinkingDelta:
    block_index: int
    delta: str
    type: Literal["thinking_delta"] = "thinking_delta"


@dataclass(frozen=True, slots=True)
class ThinkingEnd:
    block_index: int
    thinking: str
    type: Literal["thinking_end"] = "thinking_end"


@dataclass(frozen=True, slots=True)
class ToolCallStart:
    block_index: int
    type: Literal["toolcall_start"] = "toolcall_start"


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    block_index: int
    delta: str
    type: Literal["toolcall_delta"] = "toolcall_delta"


@dataclass(frozen=True, slots=True)
class ToolCallEnd:
    block_index: int
    tool_call: ToolCallBlock
    type: Literal["toolcall_end"] = "toolcall_end"


@dataclass(frozen=True, slots=True)
class StreamDone:
    message: AssistantMessage
    reason: StopReason
    type: Literal["done"] = "done"


@dataclass(frozen=True, slots=True)
class StreamError:
    message: AssistantMessage
    reason: Literal["error", "aborted"]
    type: Literal["error"] = "error"


StreamEvent = (
    StreamStart
    | TextStart
    | TextDelta
    | TextEnd
    | ThinkingStart
    | ThinkingDelta
    | ThinkingEnd
    | ToolCallStart
    | ToolCallDelta
    | ToolCallEnd
    | StreamDone
    | StreamError
)
