"""Agent-level events, tool contracts, and hook signatures.

Every AgentEvent is a frozen dataclass of JSON primitives — the same rule as
the stream protocol, for the same reason: cli/ consumes these, and a second
consumer in another language must be possible.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from harness.config import Config
from harness.llm.provider import Provider
from harness.llm.types import (
    AssistantMessage,
    ContentBlock,
    Message,
    Model,
    ToolCallBlock,
    ToolDeclaration,
    ToolResultMessage,
)

# --- tools ----------------------------------------------------------------

ExecutionMode = Literal["parallel", "sequential"]


@dataclass(frozen=True, slots=True)
class ToolResult:
    content: tuple[ContentBlock, ...]
    is_error: bool = False
    #: Structured data for renderers. Never sent to the model.
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolContext:
    """What a tool receives. Tools never reach for globals."""

    cwd: str
    call_id: str
    signal: asyncio.Event | None = None
    on_update: Callable[[ToolResult], None] | None = None


class AgentTool(Protocol):
    name: str
    description: str
    parameters: dict[str, object]
    execution_mode: ExecutionMode

    async def execute(self, args: dict[str, object], ctx: ToolContext) -> ToolResult: ...


# --- events ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AgentStart:
    type: Literal["agent_start"] = "agent_start"


@dataclass(frozen=True, slots=True)
class AgentEnd:
    messages: tuple[Message, ...]
    type: Literal["agent_end"] = "agent_end"


@dataclass(frozen=True, slots=True)
class TurnStart:
    type: Literal["turn_start"] = "turn_start"


@dataclass(frozen=True, slots=True)
class TurnEnd:
    message: AssistantMessage
    tool_results: tuple[ToolResultMessage, ...]
    type: Literal["turn_end"] = "turn_end"


@dataclass(frozen=True, slots=True)
class MessageStart:
    message: Message
    type: Literal["message_start"] = "message_start"


@dataclass(frozen=True, slots=True)
class MessageUpdate:
    """A streaming delta. Carries the delta, not a full snapshot — consumers
    needing the whole in-flight message read Agent.state.streaming_message."""

    block_index: int
    delta: str
    kind: Literal["text", "thinking"] = "text"
    type: Literal["message_update"] = "message_update"


@dataclass(frozen=True, slots=True)
class MessageEnd:
    message: Message
    type: Literal["message_end"] = "message_end"


@dataclass(frozen=True, slots=True)
class ToolExecutionStart:
    tool_call_id: str
    tool_name: str
    arguments: dict[str, object]
    type: Literal["tool_execution_start"] = "tool_execution_start"


@dataclass(frozen=True, slots=True)
class ToolExecutionUpdate:
    tool_call_id: str
    tool_name: str
    content: tuple[ContentBlock, ...]
    type: Literal["tool_execution_update"] = "tool_execution_update"


@dataclass(frozen=True, slots=True)
class ToolExecutionEnd:
    tool_call_id: str
    tool_name: str
    content: tuple[ContentBlock, ...]
    is_error: bool
    duration_ms: int
    type: Literal["tool_execution_end"] = "tool_execution_end"


AgentEvent = (
    AgentStart
    | AgentEnd
    | TurnStart
    | TurnEnd
    | MessageStart
    | MessageUpdate
    | MessageEnd
    | ToolExecutionStart
    | ToolExecutionUpdate
    | ToolExecutionEnd
)

EventSink = Callable[[AgentEvent], Awaitable[None]]

# --- hooks ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BeforeToolCallContext:
    tool_call: ToolCallBlock
    messages: tuple[Message, ...]


@dataclass(frozen=True, slots=True)
class BeforeToolCallResult:
    block: bool = False
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class AfterToolCallContext:
    tool_call: ToolCallBlock
    result: ToolResult


@dataclass(frozen=True, slots=True)
class AfterToolCallResult:
    content: tuple[ContentBlock, ...] | None = None
    is_error: bool | None = None


@dataclass(frozen=True, slots=True)
class PrepareRequestContext:
    messages: tuple[Message, ...]
    model: Model


@dataclass(frozen=True, slots=True)
class RequestUpdate:
    """Replacement state for the request being prepared. This is the seam
    context compaction will use."""

    messages: tuple[Message, ...] | None = None
    model: Model | None = None


# --- loop configuration ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class AgentContext:
    system_prompt: str
    messages: tuple[Message, ...]
    tools: tuple[AgentTool, ...] = ()


@dataclass(frozen=True, slots=True)
class LoopConfig:
    model: Model
    provider: Provider
    config: Config
    cwd: str = "."
    before_tool_call: (
        Callable[[BeforeToolCallContext], Awaitable[BeforeToolCallResult | None]] | None
    ) = None
    after_tool_call: (
        Callable[[AfterToolCallContext], Awaitable[AfterToolCallResult | None]] | None
    ) = None
    prepare_request: Callable[[PrepareRequestContext], Awaitable[RequestUpdate | None]] | None = (
        None
    )
    signal: asyncio.Event | None = None


def declarations_for(tools: tuple[AgentTool, ...]) -> tuple[ToolDeclaration, ...]:
    """Project executable tools into what the model is told about them."""
    return tuple(
        ToolDeclaration(name=t.name, description=t.description, parameters=t.parameters)
        for t in tools
    )
