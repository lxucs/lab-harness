"""The round-trip test that keeps the 'events are plain data' rule from decaying.

If this fails because someone attached a live object to an event, the fix is to
remove the object — not to special-case it here.
"""

import json

import pytest

from harness.agent.types import (
    AgentEnd,
    AgentEvent,
    AgentStart,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolExecutionUpdate,
    TurnEnd,
    TurnStart,
)
from harness.llm.events import (
    StreamDone,
    StreamError,
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
from harness.llm.serde import block_from_dict, event_from_dict, message_from_dict, to_dict
from harness.llm.types import (
    AssistantMessage,
    ImageBlock,
    ProviderBlock,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
    text,
)

BLOCKS = [
    TextBlock(text="hello"),
    ThinkingBlock(thinking="hmm", signature="sig-abc"),
    ImageBlock(data="aGk=", mime_type="image/png"),
    ToolCallBlock(id="call_1", name="read", arguments={"path": "a.py", "limit": 10}),
    ProviderBlock(provider_type="web_search_tool_result", data={"results": []}),
]

MESSAGES = [
    SystemMessage(content="You are a harness."),
    UserMessage(content=text("hi")),
    AssistantMessage(
        content=(TextBlock(text="sure"), ToolCallBlock(id="c1", name="ls", arguments={})),
        model="m",
        provider="p",
        usage=Usage(input=10, output=5),
        stop_reason="tool_use",
    ),
    AssistantMessage(
        content=(),
        model="m",
        provider="p",
        stop_reason="error",
        error_message="boom",
    ),
    ToolResultMessage(tool_call_id="c1", tool_name="ls", content=text("a.py"), duration_ms=12),
]

_ASSISTANT = AssistantMessage(content=text("done"), model="m", provider="p")

EVENTS = [
    StreamStart(),
    TextStart(block_index=0),
    TextDelta(block_index=0, delta="he"),
    TextEnd(block_index=0, text="hello"),
    ThinkingStart(block_index=1),
    ThinkingDelta(block_index=1, delta="..."),
    ThinkingEnd(block_index=1, thinking="..."),
    ToolCallStart(block_index=2),
    ToolCallDelta(block_index=2, delta='{"p":'),
    ToolCallEnd(block_index=2, tool_call=ToolCallBlock(id="c1", name="read", arguments={"p": "a"})),
    StreamDone(message=_ASSISTANT, reason="stop"),
    StreamError(message=_ASSISTANT, reason="error"),
]


@pytest.mark.parametrize("block", BLOCKS, ids=lambda b: b.type)
def test_block_roundtrip(block: object) -> None:
    assert block_from_dict(json.loads(json.dumps(to_dict(block)))) == block


def _msg_id(m: object) -> str:
    return f"{m.role}-{m.stop_reason if isinstance(m, AssistantMessage) else ''}"  # type: ignore[union-attr]


@pytest.mark.parametrize("message", MESSAGES, ids=_msg_id)
def test_message_roundtrip(message: object) -> None:
    assert message_from_dict(json.loads(json.dumps(to_dict(message)))) == message


@pytest.mark.parametrize("event", EVENTS, ids=lambda e: e.type)
def test_event_roundtrip(event: object) -> None:
    assert event_from_dict(json.loads(json.dumps(to_dict(event)))) == event


def test_every_event_type_is_covered() -> None:
    """Guard against adding an event type without adding it to EVENTS above.

    Without this, a new event type silently escapes the round-trip check and
    the 'plain data' rule decays one event at a time.
    """
    from typing import get_args

    from harness.llm import events as events_module

    union_members = set(get_args(events_module.StreamEvent))
    covered = {type(e) for e in EVENTS}
    assert covered == union_members


def test_every_block_type_is_covered() -> None:
    """Guard against adding a content block without a round-trip test case."""
    from typing import get_args

    from harness.llm import types as types_module

    union_members = set(get_args(types_module.ContentBlock))
    covered = {type(b) for b in BLOCKS}
    assert covered == union_members


def test_every_message_type_is_covered() -> None:
    """Guard against adding a message type without a round-trip test case."""
    from typing import get_args

    from harness.llm import types as types_module

    union_members = set(get_args(types_module.Message))
    covered = {type(m) for m in MESSAGES}
    assert covered == union_members


AGENT_EVENTS = [
    AgentStart(),
    AgentEnd(messages=(UserMessage(content=text("hi")),)),
    TurnStart(),
    TurnEnd(message=_ASSISTANT, tool_results=()),
    MessageStart(message=UserMessage(content=text("hi"))),
    MessageUpdate(block_index=0, delta="a", kind="text"),
    MessageEnd(message=_ASSISTANT),
    ToolExecutionStart(tool_call_id="c1", tool_name="read", arguments={"p": "a"}),
    ToolExecutionUpdate(tool_call_id="c1", tool_name="read", content=text("partial")),
    ToolExecutionEnd(
        tool_call_id="c1", tool_name="read", content=text("done"), is_error=False, duration_ms=5
    ),
]


@pytest.mark.parametrize("event", AGENT_EVENTS, ids=lambda e: e.type)
def test_agent_event_is_json_serializable(event: AgentEvent) -> None:
    """Agent events must be plain data — this is what keeps cli/ swappable."""
    encoded = json.dumps(to_dict(event))
    assert json.loads(encoded)["type"] == event.type


def test_every_agent_event_type_is_covered() -> None:
    from typing import get_args

    from harness.agent import types as agent_types

    union_members = set(get_args(agent_types.AgentEvent))
    assert {type(e) for e in AGENT_EVENTS} == union_members
