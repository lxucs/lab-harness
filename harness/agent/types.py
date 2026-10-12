"""Agent-level events, tool contracts, and hook signatures.

Every AgentEvent is a frozen dataclass of JSON primitives — the same rule as
the stream protocol, for the same reason: cli/ consumes these, and a second
consumer in another language must be possible.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

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
from harness.tools.base import AgentTool, ToolResult

# The tool contract lives one layer down, in tools/base.py, so that tools never
# import from agent/. `AgentTool` and `ToolResult` are used below; these two are
# re-exported (the `as` form is what marks that deliberate) because callers
# reach for the whole contract through agent/.
from harness.tools.base import ExecutionMode as ExecutionMode
from harness.tools.base import ToolContext as ToolContext

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


@dataclass(frozen=True, slots=True)
class FinishTurnContext:
    """A turn that is about to end the run.

    Reached only on a *clean* ending — the model asked for no tools and did
    not error. An errored or aborted turn ends the run without consulting the
    hook, because "try again" is a decision for whoever handles the error.
    """

    message: AssistantMessage
    messages: tuple[Message, ...]
    #: 0-based index of the turn that just finished, so a hook can decide
    #: based on how much of the budget is already spent.
    turn: int


@dataclass(frozen=True, slots=True)
class FinishTurnResult:
    """Whether to keep going, and with what.

    `None` — or a result whose `continue_with` is None — ends the run, which
    makes having no hook and having one that declines indistinguishable.
    """

    #: Content for a user message appended to the transcript before the next
    #: request. The loop announces it like any other message, so a subscriber
    #: and the session file both see it.
    continue_with: tuple[ContentBlock, ...] | None = None


# --- loop configuration ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class AgentContext:
    system_prompt: str
    messages: tuple[Message, ...]
    tools: tuple[AgentTool, ...] = ()
    #: Messages already in `messages` that subscribers have not seen yet —
    #: normally the user turn that triggered this run. The loop announces them
    #: after `agent_start` so that a subscriber sees the whole conversation.
    #: Without this, a session writer persists an unresumable transcript with
    #: no user messages in it. They are announced, not appended: the caller
    #: already holds them.
    unannounced: tuple[Message, ...] = ()


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
    finish_turn: Callable[[FinishTurnContext], Awaitable[FinishTurnResult | None]] | None = None
    signal: asyncio.Event | None = None


def declarations_for(tools: tuple[AgentTool, ...]) -> tuple[ToolDeclaration, ...]:
    """Project executable tools into what the model is told about them."""
    return tuple(
        ToolDeclaration(name=t.name, description=t.description, parameters=t.parameters)
        for t in tools
    )
