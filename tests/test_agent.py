import asyncio
from typing import Any

import pytest

from harness.agent.agent import Agent
from harness.agent.types import AgentEvent, ExecutionMode, ToolContext, ToolResult
from harness.llm.fake import FAKE_MODEL, FakeProvider, text_response, tool_call_response
from harness.llm.types import AssistantMessage, TextBlock, ToolCallBlock, UserMessage, text


def _agent(provider: FakeProvider, **kwargs: Any) -> Agent:
    return Agent(model=FAKE_MODEL, provider=provider, system_prompt="sys", **kwargs)


class _InstantTool:
    """A tool that returns immediately — just something for the model to call."""

    name = "work"
    description = "work"
    execution_mode: ExecutionMode = "parallel"

    def __init__(self) -> None:
        self.parameters: dict[str, object] = {"type": "object", "properties": {}}

    async def execute(self, args: dict[str, object], ctx: ToolContext) -> ToolResult:
        return ToolResult(content=text("ok"))


async def test_prompt_appends_user_and_assistant_messages() -> None:
    agent = _agent(FakeProvider([text_response("hello")]))

    await agent.prompt("hi")

    roles = [m.role for m in agent.state.messages]
    assert roles == ["user", "assistant"]
    last_block = agent.state.messages[-1].content[0]
    assert isinstance(last_block, TextBlock)
    assert last_block.text == "hello"


async def test_transcript_accumulates_across_prompts() -> None:
    agent = _agent(FakeProvider([text_response("one"), text_response("two")]))

    await agent.prompt("first")
    await agent.prompt("second")

    roles = [m.role for m in agent.state.messages]
    assert roles == ["user", "assistant", "user", "assistant"]


async def test_second_prompt_sees_the_first_exchange() -> None:
    provider = FakeProvider([text_response("one"), text_response("two")])
    agent = _agent(provider)

    await agent.prompt("first")
    await agent.prompt("second")

    assert len(provider.requests[1].messages) == 3


async def test_subscribers_receive_events() -> None:
    received: list[AgentEvent] = []

    async def listener(event: AgentEvent) -> None:
        received.append(event)

    agent = _agent(FakeProvider([text_response("hi")]))
    agent.subscribe(listener)

    await agent.prompt("go")

    assert next(e.type for e in received) == "agent_start"
    assert received[-1].type == "agent_end"


async def test_unsubscribe_stops_delivery() -> None:
    received: list[AgentEvent] = []

    async def listener(event: AgentEvent) -> None:
        received.append(event)

    agent = _agent(FakeProvider([text_response("a"), text_response("b")]))
    unsubscribe = agent.subscribe(listener)

    await agent.prompt("one")
    count_after_first = len(received)
    unsubscribe()
    await agent.prompt("two")

    assert len(received) == count_after_first


async def test_failing_subscriber_does_not_break_the_run() -> None:
    """A broken renderer must not take down the agent."""

    async def bad_listener(event: AgentEvent) -> None:
        raise RuntimeError("listener bug")

    agent = _agent(FakeProvider([text_response("survived")]))
    agent.subscribe(bad_listener)

    await agent.prompt("go")

    last_block = agent.state.messages[-1].content[0]
    assert isinstance(last_block, TextBlock)
    assert last_block.text == "survived"


async def test_rejects_concurrent_prompts() -> None:
    provider = FakeProvider([text_response("a"), text_response("b")])
    agent = _agent(provider)

    # The listener yields, so the first run is genuinely in flight across an
    # await when the second prompt arrives. Without it FakeProvider never
    # suspends, the first run completes synchronously, and there is no overlap
    # left for this test to exercise.
    async def slow_listener(event: AgentEvent) -> None:
        await asyncio.sleep(0.01)

    agent.subscribe(slow_listener)
    first = asyncio.create_task(agent.prompt("one"))
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="already running"):
        await agent.prompt("two")

    await first

    # The rejected prompt must leave no trace: no provider request, and its
    # user message must not have been appended before the guard fired.
    assert len(provider.requests) == 1
    roles = [m.role for m in agent.state.messages]
    assert roles == ["user", "assistant"]
    first_user = agent.state.messages[0]
    assert isinstance(first_user, UserMessage)
    first_block = first_user.content[0]
    assert isinstance(first_block, TextBlock)
    assert first_block.text == "one"


async def test_a_second_prompt_during_a_run_is_refused_not_queued() -> None:
    """One run at a time: a mid-run prompt is an error, not a deferred message.

    A frontend that wants to accept typing while the agent works buffers the
    text itself and prompts again when the run returns.
    """

    class SlowTool:
        name = "slow"
        description = "slow"
        execution_mode: ExecutionMode = "parallel"

        def __init__(self) -> None:
            self.parameters: dict[str, object] = {"type": "object", "properties": {}}

        async def execute(self, args: dict[str, object], ctx: ToolContext) -> ToolResult:
            await asyncio.sleep(0.02)
            return ToolResult(content=text("done"))

    call = ToolCallBlock(id="c1", name="slow", arguments={})
    provider = FakeProvider([tool_call_response([call]), text_response("finished")])
    agent = _agent(provider, tools=(SlowTool(),))

    task = asyncio.create_task(agent.prompt("start"))
    await asyncio.sleep(0.01)

    with pytest.raises(RuntimeError, match="already running"):
        await agent.prompt("also do this")

    await task

    # The refused prompt left nothing behind, and the run finished normally.
    roles = [m.role for m in agent.state.messages]
    assert roles == ["user", "assistant", "tool_result", "assistant"]
    user_texts = [
        block.text
        for m in agent.state.messages
        if isinstance(m, UserMessage)
        for block in m.content
        if isinstance(block, TextBlock)
    ]
    assert user_texts == ["start"]


async def test_abort_marks_the_run_aborted() -> None:
    class HangingTool:
        name = "hang"
        description = "hangs"
        execution_mode: ExecutionMode = "parallel"

        def __init__(self) -> None:
            self.parameters: dict[str, object] = {"type": "object", "properties": {}}

        async def execute(self, args: dict[str, object], ctx: ToolContext) -> ToolResult:
            await asyncio.sleep(0.05)
            return ToolResult(content=text("never"))

    call = ToolCallBlock(id="c1", name="hang", arguments={})
    provider = FakeProvider([tool_call_response([call]), text_response("unused")])
    agent = _agent(provider, tools=(HangingTool(),))

    task = asyncio.create_task(agent.prompt("go"))
    await asyncio.sleep(0.01)
    agent.abort()
    await task

    assert not agent.state.is_running
    assert any(
        isinstance(m, AssistantMessage) and m.stop_reason == "aborted" for m in agent.state.messages
    )


async def test_is_running_reflects_lifecycle() -> None:
    agent = _agent(FakeProvider([text_response("hi")]))
    assert not agent.state.is_running

    await agent.prompt("go")

    assert not agent.state.is_running


async def test_state_messages_match_what_the_final_request_carried() -> None:
    """state.messages is the record of what the model was actually sent.

    Divergence here is the bug class the message queue used to cause: a
    message present in state but missing from the provider's view.
    """
    call = ToolCallBlock(id="c1", name="work", arguments={})
    provider = FakeProvider([tool_call_response([call]), text_response("done")])
    agent = _agent(provider, tools=(_InstantTool(),))

    await agent.prompt("go")

    sent = provider.requests[-1].messages
    # The final request carries everything except the assistant reply to it.
    assert list(sent) == agent.state.messages[:-1]
