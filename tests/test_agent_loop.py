import asyncio
import logging
from dataclasses import replace
from typing import Any

import pytest

from harness.agent.loop import run_agent_loop
from harness.agent.types import (
    AgentContext,
    AgentEnd,
    AgentEvent,
    AgentTool,
    BeforeToolCallResult,
    EventSink,
    ExecutionMode,
    FinishTurnContext,
    FinishTurnResult,
    LoopConfig,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    PrepareRequestContext,
    RequestUpdate,
    ToolContext,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolExecutionUpdate,
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


async def test_message_start_precedes_its_deltas() -> None:
    """docs/events.md: message_start -> message_update x N -> message_end.

    A renderer allocating its streaming buffer on message_start must not
    receive deltas for a message that has not begun.
    """
    provider = FakeProvider([text_response("abc", chunk_size=1)])
    events, emit = _collector()

    await run_agent_loop(_ctx(), _cfg(provider), emit)

    kinds = [e.type for e in events if e.type.startswith("message_")]
    assert kinds == [
        "message_start",
        "message_update",
        "message_update",
        "message_update",
        "message_end",
    ]

    # message_start carries a placeholder; message_end is the authoritative one.
    start = next(e for e in events if isinstance(e, MessageStart))
    end = next(e for e in events if isinstance(e, MessageEnd))
    assert isinstance(start.message, AssistantMessage)
    assert start.message.content == ()
    assert isinstance(end.message, AssistantMessage)
    first_block = end.message.content[0]
    assert isinstance(first_block, TextBlock)
    assert first_block.text == "abc"


async def test_lone_stream_error_still_emits_exactly_one_message_start() -> None:
    """A provider may terminate with `error` as its first event, with no
    StreamStart — the fallback must still open the message exactly once."""
    provider = FakeProvider([error_response("setup failed")])
    events, emit = _collector()

    await run_agent_loop(_ctx(), _cfg(provider), emit)

    kinds = [e.type for e in events if e.type.startswith("message_")]
    assert kinds == ["message_start", "message_end"]


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
    # Assert the block IS a TextBlock before reading .text: an `isinstance`
    # filter inside the comprehension would make this pass vacuously for
    # results carrying no text at all.
    for result in results:
        assert len(result.content) == 1
        block = result.content[0]
        assert isinstance(block, TextBlock)
        assert "token limit" in block.text


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


async def test_exhausting_max_turns_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    """Nothing in the returned messages says the budget ran out.

    Without a log record the run just stops mid-task, which reads as the model
    giving up rather than as a limit being hit.
    """
    from dataclasses import replace as dc_replace

    from harness.config import AgentConfig

    tool = RecordingTool()
    call = ToolCallBlock(id="c1", name="echo", arguments={})
    provider = FakeProvider([tool_call_response([call]) for _ in range(5)])
    _, emit = _collector()
    cfg = LoopConfig(
        model=FAKE_MODEL,
        provider=provider,
        config=dc_replace(Config(), agent=AgentConfig(max_turns=2)),
    )

    with caplog.at_level(logging.WARNING, logger="harness.agent.loop"):
        await run_agent_loop(_ctx(tool), cfg, emit)

    assert any("max_turns=2" in record.message for record in caplog.records)


async def test_a_run_that_finishes_normally_does_not_report_the_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The `for/else` must not fire on the ordinary path."""
    provider = FakeProvider([text_response("done")])
    _, emit = _collector()

    with caplog.at_level(logging.WARNING, logger="harness.agent.loop"):
        await run_agent_loop(_ctx(), _cfg(provider), emit)

    assert not [r for r in caplog.records if "max_turns" in r.message]


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

    # Finding 7: the aborted message must be announced, not just returned, or a
    # sink rendering from message events shows nothing for a message that exists.
    starts = [e for e in events if isinstance(e, MessageStart)]
    ends = [e for e in events if isinstance(e, MessageEnd)]
    assert [e.message for e in starts] == [last]
    assert [e.message for e in ends] == [last]


# --- hook and subscriber isolation (Finding 1) ----------------------------


async def test_raising_after_tool_call_hook_does_not_kill_the_run() -> None:
    """Hooks are the approval-gate seam; a raising one must degrade, not escape."""
    slow = RecordingTool(name="slow", result="slow-done", delay=0.05)
    fast = RecordingTool(name="fast", result="fast-done", delay=0.0)
    calls = [
        ToolCallBlock(id="c1", name="slow", arguments={}),
        ToolCallBlock(id="c2", name="fast", arguments={}),
    ]
    provider = FakeProvider([tool_call_response(calls), text_response("recovered")])
    _, emit = _collector()

    async def boom(ctx: object) -> None:
        raise RuntimeError("hook exploded")

    messages = await run_agent_loop(_ctx(slow, fast), _cfg(provider, after_tool_call=boom), emit)

    results = [m for m in messages if isinstance(m, ToolResultMessage)]
    assert [m.tool_call_id for m in results] == ["c1", "c2"]
    for result in results:
        assert result.is_error, "a failing hook must surface as an error result"
        block = result.content[0]
        assert isinstance(block, TextBlock)
        assert "hook exploded" in block.text
    # The run continued to the next turn rather than dying mid-batch.
    assert isinstance(messages[-1], AssistantMessage)
    assert messages[-1].stop_reason == "stop"


async def test_raising_subscriber_does_not_break_the_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """docs/events.md: listener exceptions are isolated."""
    tool = RecordingTool()
    call = ToolCallBlock(id="c1", name="echo", arguments={})
    provider = FakeProvider([tool_call_response([call]), text_response("done")])

    seen: list[str] = []

    async def hostile(event: AgentEvent) -> None:
        seen.append(event.type)
        raise RuntimeError(f"subscriber exploded on {event.type}")

    with caplog.at_level(logging.ERROR, logger="harness.agent.loop"):
        messages = await run_agent_loop(_ctx(tool), _cfg(provider), hostile)

    assert tool.calls == [{}]
    assert "agent_start" in seen and "agent_end" in seen
    roles = [m.role for m in messages]
    assert roles == ["assistant", "tool_result", "assistant"]
    assert any("subscriber exploded" in r.message or r.exc_info for r in caplog.records)


# --- abort during a batch (Finding 2) -------------------------------------


class AbortingTool:
    """Fires the abort signal, then keeps running — a tool that ignores it."""

    def __init__(self, signal: asyncio.Event, name: str = "aborter") -> None:
        self.name = name
        self.description = name
        self.parameters: dict[str, object] = {"type": "object", "properties": {}}
        self.execution_mode: ExecutionMode = "sequential"
        self._signal = signal

    async def execute(self, args: dict[str, object], ctx: ToolContext) -> ToolResult:
        self._signal.set()
        return ToolResult(content=text("aborting"))


async def test_abort_mid_batch_skips_remaining_serial_tools() -> None:
    signal = asyncio.Event()
    aborter = AbortingTool(signal)
    later = RecordingTool(name="later", execution_mode="sequential")
    calls = [
        ToolCallBlock(id="c1", name="aborter", arguments={}),
        ToolCallBlock(id="c2", name="later", arguments={}),
        ToolCallBlock(id="c3", name="later", arguments={}),
    ]
    provider = FakeProvider([tool_call_response(calls)])
    _, emit = _collector()

    messages = await run_agent_loop(_ctx(aborter, later), _cfg(provider, signal=signal), emit)

    assert later.calls == [], "tools after the abort must not execute"
    results = [m for m in messages if isinstance(m, ToolResultMessage)]
    assert [m.tool_call_id for m in results] == ["c1", "c2", "c3"]
    assert not results[0].is_error
    for result in results[1:]:
        assert result.is_error
        block = result.content[0]
        assert isinstance(block, TextBlock)
        assert "aborted" in block.text.lower()
    # Only one provider request: the loop must not start another turn.
    assert len(provider.requests) == 1


async def test_abort_during_parallel_batch_is_reported_on_the_last_turn() -> None:
    """The post-batch recheck is not redundant with the one at the top of the
    turn: on the final turn that one never runs, so an abort that fired during
    the batch would go unreported."""
    from dataclasses import replace as dc_replace

    from harness.config import AgentConfig

    signal = asyncio.Event()

    class Aborting:
        name = "aborter"
        description = "aborter"

        def __init__(self) -> None:
            self.parameters: dict[str, object] = {"type": "object", "properties": {}}
            self.execution_mode: ExecutionMode = "parallel"

        async def execute(self, args: dict[str, object], ctx: ToolContext) -> ToolResult:
            signal.set()
            return ToolResult(content=text("aborting"))

    calls = [ToolCallBlock(id="c1", name="aborter", arguments={})]
    provider = FakeProvider([tool_call_response(calls)])
    events, emit = _collector()
    cfg = LoopConfig(
        model=FAKE_MODEL,
        provider=provider,
        config=dc_replace(Config(), agent=AgentConfig(max_turns=1)),
        signal=signal,
    )

    messages = await run_agent_loop(_ctx(Aborting()), cfg, emit)

    assert len(provider.requests) == 1, "no provider request after the abort"
    last = messages[-1]
    assert isinstance(last, AssistantMessage)
    assert last.stop_reason == "aborted"
    assert any(isinstance(e, MessageEnd) and e.message is last for e in events)


# --- duplicate tool call ids (Finding 3) ----------------------------------


async def test_duplicate_tool_call_ids_do_not_cross_wire_results() -> None:
    """Call ids come from the provider — untrusted. Duplicates must not collapse."""
    alpha = RecordingTool(name="alpha", result="alpha-result")
    beta = RecordingTool(name="beta", result="beta-result")
    calls = [
        ToolCallBlock(id="dup", name="alpha", arguments={}),
        ToolCallBlock(id="dup", name="beta", arguments={}),
    ]
    provider = FakeProvider([tool_call_response(calls), text_response("done")])
    _, emit = _collector()

    messages = await run_agent_loop(_ctx(alpha, beta), _cfg(provider), emit)

    results = [m for m in messages if isinstance(m, ToolResultMessage)]
    assert len(results) == 2
    by_name = {}
    for result in results:
        block = result.content[0]
        assert isinstance(block, TextBlock)
        by_name[result.tool_name] = block.text
    assert by_name == {"alpha": "alpha-result", "beta": "beta-result"}


# --- tool update ordering (Finding 4) -------------------------------------


class UpdatingTool:
    """Reports progress through ctx.on_update before returning."""

    def __init__(self, name: str = "updating", *, updates: int = 1) -> None:
        self.name = name
        self.description = name
        self.parameters: dict[str, object] = {"type": "object", "properties": {}}
        self.execution_mode: ExecutionMode = "parallel"
        self._updates = updates

    async def execute(self, args: dict[str, object], ctx: ToolContext) -> ToolResult:
        assert ctx.on_update is not None
        for i in range(self._updates):
            ctx.on_update(ToolResult(content=text(f"partial-{i}")))
        return ToolResult(content=text("final"))


async def test_tool_update_is_emitted_before_that_tools_end_event() -> None:
    """docs/events.md orders update before end. A renderer that tears down its
    per-tool widget on end must not then receive an update for it."""
    call = ToolCallBlock(id="c1", name="updating", arguments={})
    provider = FakeProvider([tool_call_response([call]), text_response("done")])
    events, emit = _collector()

    await run_agent_loop(_ctx(UpdatingTool(updates=2)), _cfg(provider), emit)

    tool_events = [e.type for e in events if e.type.startswith("tool_execution_")]
    assert tool_events == [
        "tool_execution_start",
        "tool_execution_update",
        "tool_execution_update",
        "tool_execution_end",
    ]
    updates = [e for e in events if isinstance(e, ToolExecutionUpdate)]
    assert [e.tool_call_id for e in updates] == ["c1", "c1"]
    texts = []
    for update in updates:
        block = update.content[0]
        assert isinstance(block, TextBlock)
        texts.append(block.text)
    assert texts == ["partial-0", "partial-1"]


async def test_failing_update_emit_is_observable_not_an_orphaned_task(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A fire-and-forget update task swallows its exception into a stderr dump."""
    call = ToolCallBlock(id="c1", name="updating", arguments={})
    provider = FakeProvider([tool_call_response([call]), text_response("done")])
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)
        if isinstance(event, ToolExecutionUpdate):
            raise RuntimeError("update sink exploded")

    with caplog.at_level(logging.ERROR):
        messages = await run_agent_loop(_ctx(UpdatingTool()), _cfg(provider), emit)

    assert any(isinstance(e, ToolExecutionUpdate) for e in events)
    # Attributed to the loop's own logger. A fire-and-forget task instead
    # surfaces this as asyncio's "Task exception was never retrieved" dump,
    # which lands on the `asyncio` logger at an unpredictable time.
    ours = [r for r in caplog.records if r.name == "harness.agent.loop" and r.exc_info]
    assert ours, f"expected a loop-logged failure, got {[r.name for r in caplog.records]}"
    assert not any(r.name == "asyncio" for r in caplog.records)
    # The run survived.
    roles = [m.role for m in messages]
    assert roles == ["assistant", "tool_result", "assistant"]


class CancellingTool:
    """A tool whose own task is cancelled while it runs.

    Distinct from outer cancellation of the whole run: here CancelledError
    escapes execute() and reaches gather as a returned value.
    """

    def __init__(self) -> None:
        self.name = "cancelling"
        self.description = "cancels itself"
        self.parameters: dict[str, object] = {"type": "object", "properties": {}}
        self.execution_mode: ExecutionMode = "parallel"

    async def execute(self, args: dict[str, object], ctx: ToolContext) -> ToolResult:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        await asyncio.sleep(0)
        return ToolResult(content=text("never reached"))


async def test_cancelled_child_propagates_instead_of_becoming_an_error_result() -> None:
    """gather(return_exceptions=True) hands CancelledError back as a value.

    Swallowing it would convert cancellation into a blank error result and let
    the loop issue another provider request -- the same "kept going after you
    asked it to stop" failure the abort rechecks exist to prevent.
    """
    call = ToolCallBlock(id="c1", name="cancelling", arguments={})
    # A second script entry: if cancellation is swallowed the loop consumes it.
    provider = FakeProvider([tool_call_response([call]), text_response("SHOULD NOT RUN")])
    _, emit = _collector()

    with pytest.raises(asyncio.CancelledError):
        await run_agent_loop(_ctx(CancellingTool()), _cfg(provider), emit)

    assert len(provider.requests) == 1, "loop issued another request after cancellation"


async def test_message_start_reports_the_model_prepare_request_selected() -> None:
    """A swapped model must be named by MessageStart, not just MessageEnd."""
    swapped = replace(FAKE_MODEL, id="swapped-model")
    provider = FakeProvider([text_response("ok")])
    events, emit = _collector()

    async def swap(request: PrepareRequestContext) -> RequestUpdate:
        return RequestUpdate(model=swapped)

    await run_agent_loop(_ctx(), _cfg(provider, prepare_request=swap), emit)

    starts = [e for e in events if isinstance(e, MessageStart)]
    assert len(starts) == 1
    opened = starts[0].message
    assert isinstance(opened, AssistantMessage)
    assert opened.model == "swapped-model"


# --- finish_turn -----------------------------------------------------------


def _continue_with(value: str) -> Any:
    """A hook that asks for one more turn, then lets the run end."""
    calls: list[FinishTurnContext] = []

    async def hook(ctx: FinishTurnContext) -> FinishTurnResult | None:
        calls.append(ctx)
        if len(calls) == 1:
            return FinishTurnResult(continue_with=text(value))
        return None

    hook.calls = calls  # type: ignore[attr-defined]
    return hook


async def test_no_hook_means_the_run_ends_as_before() -> None:
    provider = FakeProvider([text_response("done")])
    _, emit = _collector()

    messages = await run_agent_loop(_ctx(), _cfg(provider), emit)

    assert len(messages) == 1


async def test_finish_turn_can_continue_the_run() -> None:
    provider = FakeProvider([text_response("first"), text_response("second")])
    _, emit = _collector()
    hook = _continue_with("keep going")

    messages = await run_agent_loop(_ctx(), _cfg(provider, finish_turn=hook), emit)

    texts = [
        b.text
        for m in messages
        if isinstance(m, AssistantMessage)
        for b in m.content
        if isinstance(b, TextBlock)
    ]
    assert texts == ["first", "second"]


async def test_the_injected_message_reaches_the_provider() -> None:
    """Not merely appended to local state — the next request must carry it."""
    provider = FakeProvider([text_response("first"), text_response("second")])
    _, emit = _collector()

    await run_agent_loop(_ctx(), _cfg(provider, finish_turn=_continue_with("keep going")), emit)

    second_request = provider.requests[1].messages
    assert any(
        isinstance(m, UserMessage) and m.content == text("keep going") for m in second_request
    ), "the injected message is missing from the second request"


async def test_the_injected_message_is_announced() -> None:
    """A subscriber and the session file must both see it.

    The message queue this hook replaces got exactly this wrong.
    """
    provider = FakeProvider([text_response("first"), text_response("second")])
    events, emit = _collector()

    await run_agent_loop(_ctx(), _cfg(provider, finish_turn=_continue_with("keep going")), emit)

    announced = [
        e.message
        for e in events
        if isinstance(e, MessageEnd) and isinstance(e.message, UserMessage)
    ]
    assert text("keep going") in [m.content for m in announced]


async def test_the_injected_message_is_returned_from_the_run() -> None:
    provider = FakeProvider([text_response("first"), text_response("second")])
    _, emit = _collector()

    messages = await run_agent_loop(
        _ctx(), _cfg(provider, finish_turn=_continue_with("keep going")), emit
    )

    assert any(isinstance(m, UserMessage) and m.content == text("keep going") for m in messages)


async def test_returning_none_ends_the_run() -> None:
    provider = FakeProvider([text_response("done")])
    _, emit = _collector()

    async def decline(ctx: FinishTurnContext) -> FinishTurnResult | None:
        return None

    messages = await run_agent_loop(_ctx(), _cfg(provider, finish_turn=decline), emit)

    assert len(messages) == 1


async def test_an_empty_result_ends_the_run() -> None:
    """`FinishTurnResult()` with no content is a decline, not a continue."""
    provider = FakeProvider([text_response("done")])
    _, emit = _collector()

    async def shrug(ctx: FinishTurnContext) -> FinishTurnResult | None:
        return FinishTurnResult()

    messages = await run_agent_loop(_ctx(), _cfg(provider, finish_turn=shrug), emit)

    assert len(messages) == 1


async def test_empty_content_ends_the_run_rather_than_sending_an_empty_message() -> None:
    """Providers reject a user message with no content."""
    provider = FakeProvider([text_response("done")])
    _, emit = _collector()

    async def blank(ctx: FinishTurnContext) -> FinishTurnResult | None:
        return FinishTurnResult(continue_with=())

    messages = await run_agent_loop(_ctx(), _cfg(provider, finish_turn=blank), emit)

    assert len(messages) == 1


async def test_a_raising_hook_ends_the_run() -> None:
    """Fail-safe, like the approval gate failing closed.

    Treating a crash as "continue" would burn the whole turn budget on code
    that is already known to be broken.
    """
    provider = FakeProvider([text_response("done")])
    _, emit = _collector()

    async def broken(ctx: FinishTurnContext) -> FinishTurnResult | None:
        raise RuntimeError("hook is broken")

    messages = await run_agent_loop(_ctx(), _cfg(provider, finish_turn=broken), emit)

    assert len(messages) == 1
    assert isinstance(messages[0], AssistantMessage)


async def test_auto_continue_is_bounded_by_max_turns() -> None:
    """A hook that always continues must not loop forever."""
    from dataclasses import replace as dc_replace

    from harness.config import AgentConfig

    provider = FakeProvider([text_response(f"turn {i}") for i in range(20)])
    _, emit = _collector()

    async def always(ctx: FinishTurnContext) -> FinishTurnResult | None:
        return FinishTurnResult(continue_with=text("again"))

    cfg = LoopConfig(
        model=FAKE_MODEL,
        provider=provider,
        config=dc_replace(Config(), agent=AgentConfig(max_turns=3)),
        finish_turn=always,
    )

    messages = await run_agent_loop(_ctx(), cfg, emit)

    assert sum(1 for m in messages if isinstance(m, AssistantMessage)) == 3


async def test_an_errored_turn_is_never_auto_continued() -> None:
    """ "Try again" is a decision for whoever handles the error."""
    provider = FakeProvider([error_response("upstream exploded")])
    _, emit = _collector()
    hook = _continue_with("keep going")

    await run_agent_loop(_ctx(), _cfg(provider, finish_turn=hook), emit)

    assert hook.calls == [], "the hook must not see an errored turn"


async def test_an_aborted_run_is_never_auto_continued() -> None:
    signal = asyncio.Event()
    signal.set()
    provider = FakeProvider([text_response("never seen")])
    _, emit = _collector()
    hook = _continue_with("keep going")

    await run_agent_loop(_ctx(), _cfg(provider, finish_turn=hook, signal=signal), emit)

    assert hook.calls == []


async def test_the_hook_is_not_called_while_tools_are_still_running() -> None:
    """It fires when a turn would END the run, not at every turn boundary.

    A turn that requested tools is going to continue regardless, so asking
    whether to continue would be meaningless — and a hook that counted those
    calls would see a different number of turns than actually ran.
    """
    tool = RecordingTool()
    call = ToolCallBlock(id="c1", name="echo", arguments={})
    provider = FakeProvider([tool_call_response([call]), text_response("done")])
    _, emit = _collector()
    seen: list[FinishTurnContext] = []

    async def decline(ctx: FinishTurnContext) -> FinishTurnResult | None:
        seen.append(ctx)
        return None

    await run_agent_loop(_ctx(tool), _cfg(provider, finish_turn=decline), emit)

    # Two turns ran; only the second — the one with no tool calls — could have
    # ended the run, so only that one consults the hook.
    assert len(seen) == 1
    assert seen[0].turn == 1


async def test_the_context_carries_the_turn_and_the_transcript() -> None:
    provider = FakeProvider([text_response("done")])
    _, emit = _collector()
    seen: list[FinishTurnContext] = []

    async def inspect(ctx: FinishTurnContext) -> FinishTurnResult | None:
        seen.append(ctx)
        return None

    await run_agent_loop(_ctx(), _cfg(provider, finish_turn=inspect), emit)

    assert seen[0].turn == 0
    assert isinstance(seen[0].message, AssistantMessage)
    assert any(isinstance(m, UserMessage) for m in seen[0].messages)
