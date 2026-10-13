"""The agentic loop: stream a response, run its tool calls, repeat.

A pure async function. It owns no state — the caller passes a context and gets
back the messages the run produced. `Agent` wraps this with transcript state.

One turn, in order:

    prepare_request?  ->  stream from provider  ->  assistant message
                                                          |
        +-------------------+-----------------------------+
        |                   |                             |
    errored or           no calls                     tool calls
    aborted                 |                             |
        |              finish_turn?          before_tool_call? -> execute
        |               |       |                           -> after_tool_call?
        |         continue     END                           |
       END            |                             results appended, loop again

Four exits, and every one of them `break`s: abort, an errored turn, a clean
turn the `finish_turn` hook declines to continue, and — via `for/else` — the
`max_turns` budget running out.

Error handling is concentrated at three boundaries, so the body stays linear:
providers terminate with an error event rather than raising, tool failures
become error results rather than propagating, and third-party code reached
through hooks or the event sink is isolated from the run.
"""

import asyncio
import logging
import time
from collections.abc import Sequence

from harness.agent.types import (
    AfterToolCallContext,
    AgentContext,
    AgentEnd,
    AgentEvent,
    AgentStart,
    AgentTool,
    BeforeToolCallContext,
    EventSink,
    FinishTurnContext,
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
    Model,
    ToolCallBlock,
    ToolResultMessage,
    UserMessage,
    text,
)

_log = logging.getLogger(__name__)

_TRUNCATED_NOTICE = (
    "Tool call {name!r} was not executed: the response hit the output token "
    "limit, so its arguments may be truncated. Re-issue the call with complete "
    "arguments."
)

_ABORTED_NOTICE = "Operation aborted"

_HOOK_FAILED_NOTICE = "{hook} hook failed: {error}"


async def run_agent_loop(
    ctx: AgentContext,
    cfg: LoopConfig,
    emit: EventSink,
) -> list[Message]:
    """Run until the model stops requesting tools, or a limit is hit.

    Returns the messages produced by this run — not including the ones that
    were already in `ctx.messages`.
    """
    # Two lists, deliberately. `transcript` is everything the provider sees,
    # seeded with the history it was handed; `produced` is only what this run
    # added, which is what the caller gets back. Appending to one without the
    # other is the bug to watch for: a message missing from `transcript` never
    # reaches the model, and one missing from `produced` is never persisted.
    transcript: list[Message] = list(ctx.messages)
    produced: list[Message] = []
    tools_by_name = {t.name: t for t in ctx.tools}
    declarations = declarations_for(ctx.tools)

    await _safe_emit(emit, AgentStart())

    # Messages the caller already holds but no subscriber has seen — normally
    # the user turn that triggered this run. Announced, not appended: they are
    # already in `transcript`, and `produced` must not re-report them.
    for message in ctx.unannounced:
        await _safe_emit(emit, MessageStart(message=message))
        await _safe_emit(emit, MessageEnd(message=message))

    for turn in range(cfg.config.agent.max_turns):
        if _aborted(cfg):
            produced.append(await _emit_aborted(cfg, emit))
            break

        _log.debug("Turn %d of at most %d", turn + 1, cfg.config.agent.max_turns)
        await _safe_emit(emit, TurnStart())

        # `prepare_request` transforms the *view* sent to the provider, never
        # the transcript. That is what makes compaction safe to add here: a
        # summarized request cannot silently rewrite the history on disk.
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
            await _safe_emit(emit, TurnEnd(message=assistant, tool_results=()))
            break

        calls = assistant.tool_calls
        if not calls:
            # The model is done talking. Only here does `finish_turn` get a
            # say, because only here would the run otherwise end.
            await _safe_emit(emit, TurnEnd(message=assistant, tool_results=()))
            injected = await _finish_turn(assistant, cfg, tuple(transcript), turn)
            if injected is None:
                break
            # Appended to the transcript *and* announced.
            await _safe_emit(emit, MessageStart(message=injected))
            await _safe_emit(emit, MessageEnd(message=injected))
            transcript.append(injected)
            produced.append(injected)
            continue

        results = await _execute_tool_calls(
            calls,
            tools_by_name,
            cfg,
            emit,
            transcript=tuple(transcript),
            # A length-stopped turn may have had its tool arguments cut off
            # mid-JSON, so none of its calls are executed. See the notice.
            truncated=assistant.stop_reason == "length",
        )
        # Results go back in assistant-call order, which every provider
        # requires: a result must be matched to the call that asked for it.
        for result in results:
            await _safe_emit(emit, MessageStart(message=result))
            await _safe_emit(emit, MessageEnd(message=result))
            transcript.append(result)
            produced.append(result)

        await _safe_emit(emit, TurnEnd(message=assistant, tool_results=tuple(results)))

        # A batch can run for a long time, so the signal may have fired while it
        # was in flight. Check here rather than leaning on the next iteration's
        # check: that one never runs when the batch landed on the final turn.
        if _aborted(cfg):
            produced.append(await _emit_aborted(cfg, emit))
            break
    else:
        # `for/else` runs only when the range was exhausted without a break —
        # the model was still calling tools when the budget ran out. Nothing in
        # the returned messages says so, so without this the run just stops.
        _log.warning(
            "Stopped after agent.max_turns=%d with the model still working",
            cfg.config.agent.max_turns,
        )

    await _safe_emit(emit, AgentEnd(messages=tuple(produced)))
    return produced


async def _stream_turn(
    request: Context,
    model: Model,
    cfg: LoopConfig,
    emit: EventSink,
) -> AssistantMessage:
    """Consume one provider stream into a final AssistantMessage.

    The provider contract says failures arrive as a StreamError event, so there
    is no try/except here.
    """
    opts = StreamOptions(signal=cfg.signal)
    final: AssistantMessage | None = None
    started = False

    async def start_message() -> None:
        """Open the message before any delta can reference it.

        The assembled message does not exist until the terminal event, so
        MessageStart carries an empty placeholder and MessageEnd remains the
        authoritative one. Consumers that need the in-flight text accumulate
        the deltas; this event exists to tell them when to start.
        """
        nonlocal started
        if started:
            return
        started = True
        await _safe_emit(
            emit,
            MessageStart(
                # model, not cfg.model: prepare_request may have swapped it for
                # this turn, and MessageStart must name the model actually used.
                message=AssistantMessage(content=(), model=model.id, provider=cfg.provider.name)
            ),
        )

    # A stream may terminate with StreamError as its FIRST event when request
    # setup fails before generation starts, so nothing here may assume a
    # preceding StreamStart — every branch opens the message if it is not open.
    async for event in cfg.provider.stream(model, request, opts):
        await start_message()
        if isinstance(event, TextDelta):
            await _safe_emit(
                emit, MessageUpdate(block_index=event.block_index, delta=event.delta, kind="text")
            )
        elif isinstance(event, ThinkingDelta):
            await _safe_emit(
                emit,
                MessageUpdate(block_index=event.block_index, delta=event.delta, kind="thinking"),
            )
        elif isinstance(event, (StreamDone, StreamError)):
            final = event.message

    if final is None:
        # A provider that ends without done/error violates the contract; surface
        # it as an error turn rather than hanging the loop.
        final = AssistantMessage(
            content=(),
            model=model.id,
            provider=cfg.provider.name,
            stop_reason="error",
            error_message="Provider stream ended without a terminal event",
        )

    # A stream that yielded nothing at all never opened the message.
    await start_message()
    await _safe_emit(emit, MessageEnd(message=final))
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

    The model can ask for several tools in one message. This runs them, in
    three phases:

      1. **Prepare**, sequentially. Resolve each name and run the approval
         hook. Sequential on purpose: approval prompts must appear in the
         order the model asked, not in whatever order coroutines wake up.
      2. **Execute**, in parallel unless anything opted out.
      3. **Collect**, back into assistant-call order.

    Two orderings, deliberately different:
      - `tool_execution_end` fires in *completion* order, for live feedback
      - the returned result messages follow *assistant source* order, because
        providers require each result to match the call that requested it

    A call that was refused or whose tool does not exist never reaches phase 2;
    it carries its error result through from phase 1, so the model still gets
    one result per call. A provider rejects a turn where any call is unanswered.
    """
    if truncated:
        # The response hit the output-token limit, so the JSON of these calls
        # may be cut off mid-object. None of them run: arguments that parse but
        # are missing a field would have the tool do the wrong thing silently,
        # which is worse than asking the model to re-issue the call.
        truncated_results: list[ToolResultMessage] = []
        for call in calls:
            await _safe_emit(
                emit,
                ToolExecutionStart(
                    tool_call_id=call.id, tool_name=call.name, arguments=call.arguments
                ),
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

    # Phase 1. Each entry is (call, tool-to-run or None, result-already-known
    # or None) — exactly one of the last two is set.
    prepared: list[tuple[ToolCallBlock, AgentTool | None, ToolResult | None]] = []
    for call in calls:
        await _safe_emit(
            emit,
            ToolExecutionStart(tool_call_id=call.id, tool_name=call.name, arguments=call.arguments),
        )
        tool = tools_by_name.get(call.name)
        if tool is None:
            # Phrased for the model, which may be able to pick another tool.
            error_result = ToolResult(content=text(f"Tool {call.name!r} not found"), is_error=True)
            prepared.append((call, None, error_result))
            continue
        refusal = await _before_call(call, cfg, transcript)
        if refusal is not None:
            prepared.append((call, None, refusal))
            continue
        prepared.append((call, tool, None))

    # One sequential tool serializes the whole batch. Coarse on purpose: `write`
    # and `edit` opt out because concurrent mutation of one file is not worth
    # leaving to chance, and tracking which calls touch which paths would mean
    # interpreting arguments the loop deliberately does not understand.
    run_serially = not cfg.config.agent.parallel_tools or any(
        tool is not None and tool.execution_mode == "sequential" for _, tool, _ in prepared
    )

    # Strong references to in-flight update emits, so nothing is garbage
    # collected mid-flight. Each call awaits its own before its end event.
    update_tasks: set[asyncio.Task[None]] = set()

    async def run_one(call: ToolCallBlock, tool: AgentTool) -> tuple[ToolResult, int]:
        """Execute one tool and emit its end event.

        The emit happens HERE, inside the coroutine, so end events fire in
        completion order. `asyncio.gather` still returns in submission order,
        which is what gives the result messages their source ordering. Moving
        this emit outside would collapse both orderings into one.
        """
        start = time.monotonic()
        pending: list[asyncio.Task[None]] = []

        def on_update(partial: ToolResult) -> None:
            # Scheduled rather than awaited, because tools report progress from
            # a synchronous callback and awaiting here would serialize parallel
            # tools. The tasks are drained before the end event below, which is
            # what keeps `update` ahead of `end` and keeps failures observable.
            task = asyncio.create_task(
                _safe_emit(
                    emit,
                    ToolExecutionUpdate(
                        tool_call_id=call.id, tool_name=call.name, content=partial.content
                    ),
                )
            )
            pending.append(task)
            update_tasks.add(task)
            task.add_done_callback(update_tasks.discard)

        # Everything the tool is allowed to know, passed in. Tools never read
        # the process working directory or a global config, which is what makes
        # them testable against tmp_path and safe when two agents differ.
        tool_ctx = ToolContext(
            cwd=cfg.cwd,
            call_id=call.id,
            config=cfg.config,
            signal=cfg.signal,
            on_update=on_update,
        )
        try:
            result = await tool.execute(call.arguments, tool_ctx)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A tool must never kill the loop: the model can often recover from
            # a failed call, and it cannot recover from a dead run.
            result = ToolResult(content=text(str(exc)), is_error=True)

        if pending:
            # _safe_emit never raises, so this cannot fail the tool.
            await asyncio.gather(*pending)

        duration_ms = int((time.monotonic() - start) * 1000)
        _log.debug(
            "Tool %r finished in %dms%s",
            call.name,
            duration_ms,
            " (error)" if result.is_error else "",
        )
        final = await _finish(call, result, duration_ms, cfg, emit)
        return final, duration_ms

    # POSITION in `prepared` -> (final result, duration). Deliberately not keyed
    # on call.id: ids come from the provider, and two calls sharing one would
    # collapse to a single entry and hand the same outcome to both messages.
    outcomes: dict[int, tuple[ToolResult, int]] = {}

    # Phase 2. `finally`, not a trailing await: on cancellation the drain below
    # would be skipped and an in-flight update task would outlive the loop,
    # which is the abandonment `update_tasks` exists to prevent.
    try:
        if run_serially:
            for index, (call, tool, _immediate) in enumerate(prepared):
                if tool is None:
                    continue
                if _aborted(cfg):
                    # Backstop for tools that ignore the signal they were handed:
                    # without it a long sequential batch runs to completion after
                    # the user has already asked for it to stop.
                    outcomes[index] = (await _finish(call, _aborted_result(), 0, cfg, emit), 0)
                    continue
                outcomes[index] = await run_one(call, tool)
        else:
            runnable = [
                (index, call, tool)
                for index, (call, tool, _immediate) in enumerate(prepared)
                if tool is not None
            ]
            if runnable:
                # return_exceptions=True so one child blowing up cannot leave its
                # siblings running unsupervised after the gather has given up.
                gathered = await asyncio.gather(
                    *(run_one(call, tool) for _index, call, tool in runnable),
                    return_exceptions=True,
                )
                for (index, call, _tool), outcome in zip(runnable, gathered, strict=True):
                    if isinstance(outcome, asyncio.CancelledError):
                        # return_exceptions=True hands cancellation back as a value
                        # rather than propagating it. Re-raise so cancelling the run
                        # actually stops it: swallowing this would turn a cancelled
                        # child into a blank error result and let the loop issue
                        # another provider request.
                        raise outcome
                    if isinstance(outcome, BaseException):
                        # run_one already converts tool failures, so reaching here
                        # means the loop's own machinery failed. Report it as an
                        # error result rather than losing the call entirely.
                        _log.error(
                            "Tool %r failed outside its own error handling",
                            call.name,
                            exc_info=outcome,
                        )
                        failed = ToolResult(content=text(str(outcome)), is_error=True)
                        outcomes[index] = (await _finish(call, failed, 0, cfg, emit), 0)
                    else:
                        outcomes[index] = outcome

        # Phase 3. Back into assistant-call order, one result per call.
        results: list[ToolResultMessage] = []
        for index, (call, _tool, immediate) in enumerate(prepared):
            if immediate is not None:
                final = await _finish(call, immediate, 0, cfg, emit)
                duration = 0
            else:
                final, duration = outcomes[index]
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
    finally:
        if update_tasks:
            # Anything a cancelled or abandoned child left behind.
            await asyncio.gather(*update_tasks, return_exceptions=True)


async def _before_call(
    call: ToolCallBlock,
    cfg: LoopConfig,
    transcript: tuple[Message, ...],
) -> ToolResult | None:
    """Run before_tool_call. A non-None return replaces executing the tool."""
    if cfg.before_tool_call is None:
        return None
    try:
        decision = await cfg.before_tool_call(
            BeforeToolCallContext(tool_call=call, messages=transcript)
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # This hook is the approval gate. A gate that errors must deny rather
        # than fall open, and it must not take the run down with it.
        _log.exception("before_tool_call hook failed for %r", call.name)
        return ToolResult(
            content=text(_HOOK_FAILED_NOTICE.format(hook="before_tool_call", error=exc)),
            is_error=True,
        )
    if decision is not None and decision.block:
        return ToolResult(
            content=text(decision.reason or "Tool execution was blocked"), is_error=True
        )
    return None


async def _finish_turn(
    assistant: AssistantMessage,
    cfg: LoopConfig,
    transcript: tuple[Message, ...],
    turn: int,
) -> UserMessage | None:
    """Ask the finish_turn hook whether to keep going. None means stop.

    The run is bounded by `max_turns` whatever this returns, because the call
    sits inside that loop — an auto-continue hook can waste a budget but it
    cannot run forever.
    """
    if cfg.finish_turn is None:
        return None
    try:
        decision = await cfg.finish_turn(
            FinishTurnContext(message=assistant, messages=transcript, turn=turn)
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        # Ending is the safe failure here, the mirror of the approval gate
        # failing closed: a hook that errors on every turn and is treated as
        # "continue" would burn the whole turn budget on broken code.
        _log.exception("finish_turn hook failed; ending the run")
        return None
    if decision is None or decision.continue_with is None:
        return None
    if not decision.continue_with:
        # Empty content would be an empty user message, which providers reject.
        _log.warning("finish_turn returned empty content; ending the run")
        return None
    return UserMessage(content=decision.continue_with)


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
        try:
            override = await cfg.after_tool_call(
                AfterToolCallContext(tool_call=call, result=result)
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Report the hook failure rather than passing the tool's result
            # through: this seam is where result redaction and approval live,
            # so silently forwarding an unreviewed result is the worse default.
            # That goes for the details as much as the content — they reach
            # every renderer too.
            _log.exception("after_tool_call hook failed for %r", call.name)
            result = ToolResult(
                content=text(_HOOK_FAILED_NOTICE.format(hook="after_tool_call", error=exc)),
                is_error=True,
            )
        else:
            if override is not None:
                result = ToolResult(
                    content=override.content if override.content is not None else result.content,
                    is_error=(
                        override.is_error if override.is_error is not None else result.is_error
                    ),
                    details=override.details if override.details is not None else result.details,
                )

    await _safe_emit(
        emit,
        ToolExecutionEnd(
            tool_call_id=call.id,
            tool_name=call.name,
            content=result.content,
            is_error=result.is_error,
            duration_ms=duration_ms,
            details=result.details,
        ),
    )
    return result


async def _safe_emit(emit: EventSink, event: AgentEvent) -> None:
    """Deliver an event, isolating subscriber failures.

    docs/events.md promises that a failing subscriber must not break a run. The
    sink is third-party code — a terminal renderer, a JSONL writer — and the
    loop has no way to repair its state, so the only sane response is to log
    and keep going.
    """
    try:
        await emit(event)
    except asyncio.CancelledError:
        raise
    except Exception:
        _log.exception("Event subscriber failed on %s", event.type)


async def _emit_aborted(cfg: LoopConfig, emit: EventSink) -> AssistantMessage:
    """Announce the aborted message like any other the loop produces.

    Without the start/end pair a sink rendering from message events shows
    nothing for a message the run still returns.
    """
    message = _aborted_message(cfg)
    await _safe_emit(emit, MessageStart(message=message))
    await _safe_emit(emit, MessageEnd(message=message))
    return message


def _aborted(cfg: LoopConfig) -> bool:
    return cfg.signal is not None and cfg.signal.is_set()


def _aborted_result() -> ToolResult:
    return ToolResult(content=text(_ABORTED_NOTICE), is_error=True)


def _aborted_message(cfg: LoopConfig) -> AssistantMessage:
    return AssistantMessage(
        content=(),
        model=cfg.model.id,
        provider=cfg.provider.name,
        stop_reason="aborted",
        error_message=_ABORTED_NOTICE,
    )
