import asyncio
from typing import Any

from harness.agent.loop import run_agent_loop
from harness.agent.types import (
    AgentContext,
    AgentEnd,
    AgentEvent,
    AgentTool,
    BeforeToolCallResult,
    EventSink,
    ExecutionMode,
    LoopConfig,
    MessageUpdate,
    ToolContext,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolResult,
)
from harness.config import Config
from harness.llm.fake import (
    FAKE_MODEL,
    FakeProvider,
    error_response,
    text_response,
    tool_call_response,
    truncated_tool_call_response,
)
from harness.llm.types import (
    AssistantMessage,
    TextBlock,
    ToolCallBlock,
    ToolResultMessage,
    UserMessage,
    text,
)


class RecordingTool:
    """A tool that records its calls and returns a fixed result."""

    def __init__(
        self,
        name: str = "echo",
        *,
        result: str = "ok",
        is_error: bool = False,
        delay: float = 0.0,
        execution_mode: ExecutionMode = "parallel",
    ) -> None:
        self.name = name
        self.description = f"{name} tool"
        self.parameters: dict[str, object] = {"type": "object", "properties": {}}
        self.execution_mode: ExecutionMode = execution_mode
        self.calls: list[dict[str, object]] = []
        self._result = result
        self._is_error = is_error
        self._delay = delay

    async def execute(self, args: dict[str, object], ctx: ToolContext) -> ToolResult:
        self.calls.append(args)
        if self._delay:
            await asyncio.sleep(self._delay)
        return ToolResult(content=text(self._result), is_error=self._is_error)


class RaisingTool:
    name = "boom"
    description = "always raises"

    def __init__(self) -> None:
        self.parameters: dict[str, object] = {"type": "object", "properties": {}}
        self.execution_mode: ExecutionMode = "parallel"

    async def execute(self, args: dict[str, object], ctx: ToolContext) -> ToolResult:
        raise RuntimeError("tool exploded")


def _collector() -> tuple[list[AgentEvent], EventSink]:
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    return events, emit


def _cfg(provider: FakeProvider, **kwargs: Any) -> LoopConfig:
    return LoopConfig(model=FAKE_MODEL, provider=provider, config=Config(), **kwargs)


def _ctx(*tools: AgentTool) -> AgentContext:
    return AgentContext(
        system_prompt="sys",
        messages=(UserMessage(content=text("hi")),),
        tools=tuple(tools),
    )


async def test_simple_text_turn_emits_lifecycle_events() -> None:
    provider = FakeProvider([text_response("hello")])
    events, emit = _collector()

    messages = await run_agent_loop(_ctx(), _cfg(provider), emit)

    kinds = [e.type for e in events]
    assert kinds[0] == "agent_start"
    assert kinds[-1] == "agent_end"
    assert "turn_start" in kinds and "turn_end" in kinds
    assert len(messages) == 1
    assert isinstance(messages[0], AssistantMessage)
    first_block = messages[0].content[0]
    assert isinstance(first_block, TextBlock)
    assert first_block.text == "hello"


async def test_streaming_deltas_become_message_updates() -> None:
    provider = FakeProvider([text_response("abc", chunk_size=1)])
    events, emit = _collector()

    await run_agent_loop(_ctx(), _cfg(provider), emit)

    deltas = [e.delta for e in events if isinstance(e, MessageUpdate)]
    assert deltas == ["a", "b", "c"]


async def test_executes_tool_call_then_continues() -> None:
    tool = RecordingTool()
    call = ToolCallBlock(id="c1", name="echo", arguments={"v": 1})
    provider = FakeProvider([tool_call_response([call]), text_response("done")])
    events, emit = _collector()

    messages = await run_agent_loop(_ctx(tool), _cfg(provider), emit)

    assert tool.calls == [{"v": 1}]
    roles = [m.role for m in messages]
    assert roles == ["assistant", "tool_result", "assistant"]
    assert any(isinstance(e, ToolExecutionStart) for e in events)
    assert any(isinstance(e, ToolExecutionEnd) for e in events)


async def test_tool_result_is_sent_back_to_the_model() -> None:
    tool = RecordingTool(result="42")
    call = ToolCallBlock(id="c1", name="echo", arguments={})
    provider = FakeProvider([tool_call_response([call]), text_response("done")])
    _, emit = _collector()

    await run_agent_loop(_ctx(tool), _cfg(provider), emit)

    second_request = provider.requests[1]
    tool_results = [m for m in second_request.messages if isinstance(m, ToolResultMessage)]
    assert len(tool_results) == 1
    first_block = tool_results[0].content[0]
    assert isinstance(first_block, TextBlock)
    assert first_block.text == "42"


async def test_unknown_tool_yields_error_result_without_crashing() -> None:
    call = ToolCallBlock(id="c1", name="nonexistent", arguments={})
    provider = FakeProvider([tool_call_response([call]), text_response("recovered")])
    _, emit = _collector()

    messages = await run_agent_loop(_ctx(), _cfg(provider), emit)

    result = next(m for m in messages if isinstance(m, ToolResultMessage))
    assert result.is_error
    first_block = result.content[0]
    assert isinstance(first_block, TextBlock)
    assert "not found" in first_block.text.lower()


async def test_raising_tool_becomes_error_result() -> None:
    call = ToolCallBlock(id="c1", name="boom", arguments={})
    provider = FakeProvider([tool_call_response([call]), text_response("recovered")])
    _, emit = _collector()

    messages = await run_agent_loop(_ctx(RaisingTool()), _cfg(provider), emit)

    result = next(m for m in messages if isinstance(m, ToolResultMessage))
    assert result.is_error
    first_block = result.content[0]
    assert isinstance(first_block, TextBlock)
    assert "tool exploded" in first_block.text


async def test_provider_error_ends_the_run() -> None:
    provider = FakeProvider([error_response("rate limited")])
    events, emit = _collector()

    messages = await run_agent_loop(_ctx(), _cfg(provider), emit)

    last = messages[-1]
    assert isinstance(last, AssistantMessage)
    assert last.stop_reason == "error"
    assert last.error_message == "rate limited"
    assert isinstance(events[-1], AgentEnd)


async def test_truncated_response_fails_all_tool_calls_unexecuted() -> None:
    """stop_reason='length' means arguments may be silently incomplete."""
    tool = RecordingTool()
    calls = [
        ToolCallBlock(id="c1", name="echo", arguments={"v": 1}),
        ToolCallBlock(id="c2", name="echo", arguments={"v": 2}),
    ]
    provider = FakeProvider([truncated_tool_call_response(calls), text_response("retry")])
    _, emit = _collector()

    messages = await run_agent_loop(_ctx(tool), _cfg(provider), emit)

    assert tool.calls == []
    results = [m for m in messages if isinstance(m, ToolResultMessage)]
    assert len(results) == 2
    assert all(r.is_error for r in results)
    assert all(
        "token limit" in b.text for r in results for b in r.content if isinstance(b, TextBlock)
    )


async def test_before_tool_call_can_block() -> None:
    tool = RecordingTool()
    call = ToolCallBlock(id="c1", name="echo", arguments={})
    provider = FakeProvider([tool_call_response([call]), text_response("ok")])
    _, emit = _collector()

    async def deny(ctx: object) -> BeforeToolCallResult:
        return BeforeToolCallResult(block=True, reason="denied by policy")

    messages = await run_agent_loop(_ctx(tool), _cfg(provider, before_tool_call=deny), emit)

    assert tool.calls == []
    result = next(m for m in messages if isinstance(m, ToolResultMessage))
    assert result.is_error
    first_block = result.content[0]
    assert isinstance(first_block, TextBlock)
    assert first_block.text == "denied by policy"


async def test_parallel_tools_run_concurrently_but_results_keep_source_order() -> None:
    """tool_execution_end fires in completion order; result messages do not."""
    slow = RecordingTool(name="slow", result="slow-done", delay=0.05)
    fast = RecordingTool(name="fast", result="fast-done", delay=0.0)
    calls = [
        ToolCallBlock(id="c1", name="slow", arguments={}),
        ToolCallBlock(id="c2", name="fast", arguments={}),
    ]
    provider = FakeProvider([tool_call_response(calls), text_response("ok")])
    events, emit = _collector()

    messages = await run_agent_loop(_ctx(slow, fast), _cfg(provider), emit)

    completion = [e.tool_call_id for e in events if isinstance(e, ToolExecutionEnd)]
    assert completion == ["c2", "c1"], "fast tool should finish first"

    results = [m.tool_call_id for m in messages if isinstance(m, ToolResultMessage)]
    assert results == ["c1", "c2"], "result messages must follow assistant source order"


async def test_sequential_tool_forces_serial_execution() -> None:
    order: list[str] = []

    class OrderedTool:
        def __init__(self, name: str, delay: float) -> None:
            self.name = name
            self.description = name
            self.parameters: dict[str, object] = {"type": "object", "properties": {}}
            self.execution_mode: ExecutionMode = "sequential"
            self._delay = delay

        async def execute(self, args: dict[str, object], ctx: ToolContext) -> ToolResult:
            order.append(f"{self.name}-start")
            await asyncio.sleep(self._delay)
            order.append(f"{self.name}-end")
            return ToolResult(content=text("ok"))

    calls = [
        ToolCallBlock(id="c1", name="first", arguments={}),
        ToolCallBlock(id="c2", name="second", arguments={}),
    ]
    provider = FakeProvider([tool_call_response(calls), text_response("ok")])
    _, emit = _collector()

    await run_agent_loop(
        _ctx(OrderedTool("first", 0.03), OrderedTool("second", 0.0)),
        _cfg(provider),
        emit,
    )

    assert order == ["first-start", "first-end", "second-start", "second-end"]


async def test_prepare_request_can_replace_messages() -> None:
    """The seam compaction will use."""
    provider = FakeProvider([text_response("ok")])
    _, emit = _collector()
    replacement = (UserMessage(content=text("replaced")),)

    async def replace(ctx: object) -> object:
        from harness.agent.types import RequestUpdate

        return RequestUpdate(messages=replacement)

    await run_agent_loop(_ctx(), _cfg(provider, prepare_request=replace), emit)

    assert provider.requests[0].messages == replacement


async def test_max_turns_stops_a_runaway_loop() -> None:
    tool = RecordingTool()
    call = ToolCallBlock(id="c1", name="echo", arguments={})
    # Always asks for another tool call: without a cap this never terminates.
    provider = FakeProvider([tool_call_response([call]) for _ in range(5)])
    _, emit = _collector()

    from dataclasses import replace as dc_replace

    from harness.config import AgentConfig

    cfg = LoopConfig(
        model=FAKE_MODEL,
        provider=provider,
        config=dc_replace(Config(), agent=AgentConfig(max_turns=3)),
    )

    messages = await run_agent_loop(_ctx(tool), cfg, emit)

    assistant_count = sum(1 for m in messages if isinstance(m, AssistantMessage))
    assert assistant_count == 3


async def test_abort_signal_stops_the_run() -> None:
    signal = asyncio.Event()
    signal.set()
    provider = FakeProvider([text_response("never seen")])
    events, emit = _collector()

    messages = await run_agent_loop(_ctx(), _cfg(provider, signal=signal), emit)

    last = messages[-1]
    assert isinstance(last, AssistantMessage)
    assert last.stop_reason == "aborted"
    assert isinstance(events[-1], AgentEnd)
