"""Stateful wrapper around the agent loop.

Owns the transcript, fans events out to subscribers, and exposes abort and
message queueing. This is the object a frontend talks to; `run_agent_loop`
stays pure.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from harness.agent.loop import run_agent_loop
from harness.agent.types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEvent,
    AgentTool,
    BeforeToolCallContext,
    BeforeToolCallResult,
    LoopConfig,
    MessageEnd,
    MessageStart,
    PrepareRequestContext,
    RequestUpdate,
)
from harness.config import Config, load_config
from harness.llm.provider import Provider
from harness.llm.types import AssistantMessage, Message, Model, UserMessage, text

logger = logging.getLogger(__name__)

Listener = Callable[[AgentEvent], Awaitable[None]]


@dataclass
class AgentState:
    system_prompt: str
    model: Model
    messages: list[Message] = field(default_factory=list)
    is_running: bool = False
    #: Text accumulated for the in-flight assistant message, for renderers that
    #: want the whole thing rather than deltas.
    streaming_message: str = ""


class Agent:
    def __init__(
        self,
        *,
        model: Model,
        provider: Provider,
        system_prompt: str = "",
        tools: tuple[AgentTool, ...] = (),
        config: Config | None = None,
        cwd: str = ".",
        before_tool_call: (
            Callable[[BeforeToolCallContext], Awaitable[BeforeToolCallResult | None]] | None
        ) = None,
        after_tool_call: (
            Callable[[AfterToolCallContext], Awaitable[AfterToolCallResult | None]] | None
        ) = None,
        prepare_request: (
            Callable[[PrepareRequestContext], Awaitable[RequestUpdate | None]] | None
        ) = None,
    ) -> None:
        self.state = AgentState(system_prompt=system_prompt, model=model)
        self._provider = provider
        self._tools = tools
        self._config = config or load_config()
        self._cwd = cwd
        self._before_tool_call = before_tool_call
        self._after_tool_call = after_tool_call
        self._prepare_request = prepare_request
        self._listeners: list[Listener] = []
        self._queue: list[Message] = []
        self._signal: asyncio.Event | None = None
        self._idle = asyncio.Event()
        self._idle.set()

    # --- subscriptions ----------------------------------------------------

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Register an event listener. Returns an unsubscribe callable."""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    async def _emit(self, event: AgentEvent) -> None:
        self._reduce(event)
        for listener in list(self._listeners):
            try:
                await listener(event)
            except Exception:
                # A broken renderer must not take down the run.
                logger.exception("Event listener raised; continuing")

    def _reduce(self, event: AgentEvent) -> None:
        """Fold an event into agent state before listeners see it."""
        if isinstance(event, MessageEnd):
            self.state.messages.append(event.message)
            if isinstance(event.message, AssistantMessage):
                self.state.streaming_message = ""
        elif isinstance(event, MessageStart):
            if isinstance(event.message, AssistantMessage):
                self.state.streaming_message = ""
        elif event.type == "message_update":
            self.state.streaming_message += event.delta

    # --- running ----------------------------------------------------------

    async def prompt(self, message: str | Message) -> list[Message]:
        """Send a message and run until the agent stops."""
        # Claim the run in the same synchronous step as the check, so there is
        # no await between testing the flag and setting it.
        if self.state.is_running:
            raise RuntimeError(
                "Agent is already running. Use queue() to inject a message, or "
                "await wait_for_idle()."
            )
        self.state.is_running = True

        user = UserMessage(content=text(message)) if isinstance(message, str) else message
        self.state.messages.append(user)
        return await self._run()

    async def _run(self) -> list[Message]:
        # Idempotent with the claim in prompt(); set here too so _run() is
        # correct on its own. _run's finally always clears it.
        self.state.is_running = True
        self._idle.clear()
        self._signal = asyncio.Event()

        try:
            ctx = AgentContext(
                system_prompt=self.state.system_prompt,
                messages=tuple(self.state.messages),
                tools=self._tools,
            )
            cfg = LoopConfig(
                model=self.state.model,
                provider=self._provider,
                config=self._config,
                cwd=self._cwd,
                before_tool_call=self._before_tool_call,
                after_tool_call=self._after_tool_call,
                prepare_request=self._prepare_request_with_queue,
                signal=self._signal,
            )
            # The loop appends to its own transcript copy; _reduce keeps
            # self.state.messages in sync via MessageEnd events. Drop the
            # pre-existing messages the loop echoes back.
            produced = await run_agent_loop(ctx, cfg, self._emit)
            return produced
        finally:
            self.state.is_running = False
            self._signal = None
            self._idle.set()

    async def _prepare_request_with_queue(
        self, request: PrepareRequestContext
    ) -> RequestUpdate | None:
        """Inject queued messages at the turn boundary, then delegate.

        This is where 'type while it's working' lands: the queue is drained
        once per request, and the messages are emitted so subscribers render
        them in order alongside everything else.
        """
        messages = request.messages
        changed = False

        if self._queue:
            drained = tuple(self._queue)
            self._queue.clear()
            messages = messages + drained
            changed = True
            for message in drained:
                await self._emit(MessageStart(message=message))
                await self._emit(MessageEnd(message=message))

        if self._prepare_request is not None:
            update = await self._prepare_request(
                PrepareRequestContext(messages=messages, model=request.model)
            )
            if update is not None:
                return RequestUpdate(
                    messages=update.messages if update.messages is not None else messages,
                    model=update.model,
                )

        return RequestUpdate(messages=messages) if changed else None

    # --- control ----------------------------------------------------------

    def queue(self, message: str | Message) -> None:
        """Inject a message at the next turn boundary.

        This is how 'type while it's working' is handled — one queue, drained
        when the loop next prepares a request.
        """
        self._queue.append(
            UserMessage(content=text(message)) if isinstance(message, str) else message
        )

    def abort(self) -> None:
        """Cancel the in-flight run. The transcript stays valid."""
        if self._signal is not None:
            self._signal.set()

    async def wait_for_idle(self) -> None:
        await self._idle.wait()
