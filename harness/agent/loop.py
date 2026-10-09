"""The agentic loop: stream a response, run its tool calls, repeat.

A pure async function. It owns no state — the caller passes a context and gets
back the messages the run produced. `Agent` wraps this with transcript state.

Error handling is concentrated at two boundaries, so the body stays linear:
providers terminate with an error event rather than raising, and tool failures
become error results rather than propagating.
"""

import asyncio
import time
from collections.abc import Sequence

from harness.agent.types import (
    AfterToolCallContext,
    AgentContext,
    AgentEnd,
    AgentStart,
    AgentTool,
    BeforeToolCallContext,
    EventSink,
    LoopConfig,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    PrepareRequestContext,
    ToolContext,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolExecutionUpdate,
    ToolResult,
    TurnEnd,
    TurnStart,
    declarations_for,
)
from harness.llm.events import (
    StreamDone,
    StreamError,
    TextDelta,
    ThinkingDelta,
)
from harness.llm.provider import StreamOptions
from harness.llm.types import (
    AssistantMessage,
    Context,
    Message,
    ToolCallBlock,
    ToolResultMessage,
    text,
)

_TRUNCATED_NOTICE = (
    "Tool call {name!r} was not executed: the response hit the output token "
    "limit, so its arguments may be truncated. Re-issue the call with complete "
    "arguments."
)


async def run_agent_loop(
    ctx: AgentContext,
    cfg: LoopConfig,
    emit: EventSink,
) -> list[Message]:
    """Run until the model stops requesting tools, or a limit is hit.

    Returns the messages produced by this run — not including the ones that
    were already in `ctx.messages`.
    """
    transcript: list[Message] = list(ctx.messages)
    produced: list[Message] = []
    tools_by_name = {t.name: t for t in ctx.tools}
    declarations = declarations_for(ctx.tools)

    await emit(AgentStart())

    for _turn in range(cfg.config.agent.max_turns):
        if _aborted(cfg):
            produced.append(_aborted_message(cfg))
            break

        await emit(TurnStart())

        messages = tuple(transcript)
        model = cfg.model
        if cfg.prepare_request is not None:
            update = await cfg.prepare_request(
                PrepareRequestContext(messages=messages, model=model)
            )
            if update is not None:
                messages = update.messages if update.messages is not None else messages
                model = update.model if update.model is not None else model

        request = Context(system_prompt=ctx.system_prompt, messages=messages, tools=declarations)
        assistant = await _stream_turn(request, model, cfg, emit)

        transcript.append(assistant)
        produced.append(assistant)

        if assistant.stop_reason in ("error", "aborted"):
            await emit(TurnEnd(message=assistant, tool_results=()))
            break

        calls = assistant.tool_calls
        if not calls:
            await emit(TurnEnd(message=assistant, tool_results=()))
            break

        results = await _execute_tool_calls(
            calls,
            tools_by_name,
            cfg,
            emit,
            transcript=tuple(transcript),
            truncated=assistant.stop_reason == "length",
        )
        for result in results:
            await emit(MessageStart(message=result))
            await emit(MessageEnd(message=result))
            transcript.append(result)
            produced.append(result)

        await emit(TurnEnd(message=assistant, tool_results=tuple(results)))

    await emit(AgentEnd(messages=tuple(produced)))
    return produced


async def _stream_turn(
    request: Context,
    model: object,
    cfg: LoopConfig,
    emit: EventSink,
) -> AssistantMessage:
    """Consume one provider stream into a final AssistantMessage.

    The provider contract says failures arrive as a StreamError event, so there
    is no try/except here.
    """
    opts = StreamOptions(signal=cfg.signal)
    final: AssistantMessage | None = None

    # A stream may terminate with StreamError as its FIRST event when request
    # setup fails before generation starts, so nothing here may assume a
    # preceding StreamStart.
    async for event in cfg.provider.stream(model, request, opts):  # type: ignore[arg-type]
        if isinstance(event, TextDelta):
            await emit(MessageUpdate(block_index=event.block_index, delta=event.delta, kind="text"))
        elif isinstance(event, ThinkingDelta):
            await emit(
                MessageUpdate(block_index=event.block_index, delta=event.delta, kind="thinking")
            )
        elif isinstance(event, (StreamDone, StreamError)):
            final = event.message

    if final is None:
        # A provider that ends without done/error violates the contract; surface
        # it as an error turn rather than hanging the loop.
        final = AssistantMessage(
            content=(),
            model=cfg.model.id,
            provider=cfg.provider.name,
            stop_reason="error",
            error_message="Provider stream ended without a terminal event",
        )

    await emit(MessageStart(message=final))
    await emit(MessageEnd(message=final))
    return final


async def _execute_tool_calls(
    calls: Sequence[ToolCallBlock],
    tools_by_name: dict[str, AgentTool],
    cfg: LoopConfig,
    emit: EventSink,
    *,
    transcript: tuple[Message, ...],
    truncated: bool,
) -> list[ToolResultMessage]:
    """Run a batch of tool calls.

    Two orderings, deliberately different:
      - tool_execution_end fires in completion order, for live feedback
      - the returned result messages follow assistant source order, because
        providers require results to match the calls that requested them
    """
    if truncated:
        truncated_results: list[ToolResultMessage] = []
        for call in calls:
            await emit(
                ToolExecutionStart(
                    tool_call_id=call.id, tool_name=call.name, arguments=call.arguments
                )
            )
            final = await _finish(
                call,
                ToolResult(content=text(_TRUNCATED_NOTICE.format(name=call.name)), is_error=True),
                0,
                cfg,
                emit,
            )
            truncated_results.append(
                ToolResultMessage(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    content=final.content,
                    is_error=final.is_error,
                    duration_ms=0,
                )
            )
        return truncated_results

    prepared: list[tuple[ToolCallBlock, AgentTool | None, ToolResult | None]] = []
    for call in calls:
        await emit(
            ToolExecutionStart(tool_call_id=call.id, tool_name=call.name, arguments=call.arguments)
        )
        tool = tools_by_name.get(call.name)
        if tool is None:
            error_result = ToolResult(content=text(f"Tool {call.name!r} not found"), is_error=True)
            prepared.append((call, None, error_result))
            continue
        if cfg.before_tool_call is not None:
            decision = await cfg.before_tool_call(
                BeforeToolCallContext(tool_call=call, messages=transcript)
            )
            if decision is not None and decision.block:
                reason = decision.reason or "Tool execution was blocked"
                prepared.append((call, None, ToolResult(content=text(reason), is_error=True)))
                continue
        prepared.append((call, tool, None))

    run_serially = not cfg.config.agent.parallel_tools or any(
        tool is not None and tool.execution_mode == "sequential" for _, tool, _ in prepared
    )

    async def run_one(call: ToolCallBlock, tool: AgentTool) -> tuple[ToolResult, int]:
        """Execute one tool and emit its end event.

        The emit happens HERE, inside the coroutine, so end events fire in
        completion order. `asyncio.gather` still returns in submission order,
        which is what gives the result messages their source ordering. Moving
        this emit outside would collapse both orderings into one.
        """
        start = time.monotonic()

        def on_update(partial: ToolResult) -> None:
            # Fire-and-forget: tool updates are advisory, and awaiting here would
            # serialize parallel tools.
            asyncio.ensure_future(  # noqa: RUF006
                emit(
                    ToolExecutionUpdate(
                        tool_call_id=call.id, tool_name=call.name, content=partial.content
                    )
                )
            )

        tool_ctx = ToolContext(cwd=cfg.cwd, call_id=call.id, signal=cfg.signal, on_update=on_update)
        try:
            result = await tool.execute(call.arguments, tool_ctx)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a tool must never kill the loop
            result = ToolResult(content=text(str(exc)), is_error=True)

        duration_ms = int((time.monotonic() - start) * 1000)
        final = await _finish(call, result, duration_ms, cfg, emit)
        return final, duration_ms

    # call id -> (final result, duration). Populated in completion order by the
    # coroutines; read back below in assistant source order.
    outcomes: dict[str, tuple[ToolResult, int]] = {}

    if run_serially:
        for call, tool, _immediate in prepared:
            if tool is None:
                continue
            outcomes[call.id] = await run_one(call, tool)
    else:
        runnable = [(call, tool) for call, tool, _i in prepared if tool is not None]
        if runnable:
            gathered = await asyncio.gather(*(run_one(c, t) for c, t in runnable))
            outcomes = {
                call.id: outcome for (call, _), outcome in zip(runnable, gathered, strict=True)
            }

    results: list[ToolResultMessage] = []
    for call, _tool, immediate in prepared:
        if immediate is not None:
            final, duration = await _finish(call, immediate, 0, cfg, emit), 0
        else:
            final, duration = outcomes[call.id]
        results.append(
            ToolResultMessage(
                tool_call_id=call.id,
                tool_name=call.name,
                content=final.content,
                is_error=final.is_error,
                duration_ms=duration,
            )
        )
    return results


async def _finish(
    call: ToolCallBlock,
    result: ToolResult,
    duration_ms: int,
    cfg: LoopConfig,
    emit: EventSink,
) -> ToolResult:
    """Apply after_tool_call, then emit the end event.

    Returns the final ToolResult; the caller builds the result message so that
    message ordering stays independent of completion ordering.
    """
    if cfg.after_tool_call is not None:
        override = await cfg.after_tool_call(AfterToolCallContext(tool_call=call, result=result))
        if override is not None:
            result = ToolResult(
                content=override.content if override.content is not None else result.content,
                is_error=(override.is_error if override.is_error is not None else result.is_error),
                details=result.details,
            )

    await emit(
        ToolExecutionEnd(
            tool_call_id=call.id,
            tool_name=call.name,
            content=result.content,
            is_error=result.is_error,
            duration_ms=duration_ms,
        )
    )
    return result


def _aborted(cfg: LoopConfig) -> bool:
    return cfg.signal is not None and cfg.signal.is_set()


def _aborted_message(cfg: LoopConfig) -> AssistantMessage:
    return AssistantMessage(
        content=(),
        model=cfg.model.id,
        provider=cfg.provider.name,
        stop_reason="aborted",
        error_message="Operation aborted",
    )
