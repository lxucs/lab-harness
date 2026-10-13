"""Stateful wrapper around the agent loop.

Owns the transcript, fans events out to subscribers, and exposes abort. This is
the object a frontend talks to; `run_agent_loop` stays pure.

One run at a time, by design. `prompt()` raises if a run is already in flight
rather than coordinating with it. That removes the machinery a concurrent model
would need — a message queue, injection at turn boundaries, an idle barrier —
and with it the state-divergence bugs that machinery invites. A terminal that
wants to accept typing mid-run buffers the text itself and calls `prompt()`
when the run finishes.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
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
    FinishTurnContext,
    FinishTurnResult,
    LoopConfig,
    MessageEnd,
    MessageStart,
    PrepareRequestContext,
    RequestUpdate,
)
from harness.config import Config, load_config
from harness.llm.provider import Provider
from harness.llm.types import AssistantMessage, Message, Model, UserMessage, text

_log = logging.getLogger(__name__)

Listener = Callable[[AgentEvent], Awaitable[None]]


@dataclass
class AgentState:
    system_prompt: str
    model: Model
    #: Declared to the model on every request, in this order.
    tools: tuple[AgentTool, ...] = ()
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
        finish_turn: (
            Callable[[FinishTurnContext], Awaitable[FinishTurnResult | None]] | None
        ) = None,
    ) -> None:
        self.state = AgentState(system_prompt=system_prompt, model=model, tools=tools)
        self._provider = provider
        self._config = config if config is not None else load_config()
        self._cwd = cwd
        self._before_tool_call = before_tool_call
        self._after_tool_call = after_tool_call
        self._prepare_request = prepare_request
        self._finish_turn = finish_turn
        self._listeners: list[Listener] = []
        self._signal: asyncio.Event | None = None

    def restore(self, messages: Sequence[Message]) -> None:
        """Seed the transcript from a saved session.

        Deliberately silent: these messages are already on disk, so announcing
        them would make a resumed session re-record its own history on the
        first write. Call before any `prompt()`.
        """
        if self.state.is_running:
            raise RuntimeError("Cannot restore while a run is in flight.")
        self.state.messages = list(messages)

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
        # Snapshot: a listener may subscribe or unsubscribe while being called.
        for listener in list(self._listeners):
            try:
                await listener(event)
            except Exception:
                # A broken renderer must not take down the run.
                _log.exception("Event listener raised; continuing")

    def _reduce(self, event: AgentEvent) -> None:
        """Fold an event into agent state before listeners see it.

        This owns every append to `state.messages` — including the user
        message, which `prompt()` hands to the loop rather than appending, so
        that this stays the only writer. Every message is announced exactly
        once, so appending anywhere else would duplicate it.
        """
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
        """Send a message and run until the agent stops.

        Raises if a run is already in flight.
        """
        # Claim the run in the same synchronous step as the check, so there is
        # no await between testing the flag and setting it.
        if self.state.is_running:
            raise RuntimeError("Agent is already running; wait for prompt() to return.")
        self.state.is_running = True
        self._signal = asyncio.Event()

        try:
            user = UserMessage(content=text(message)) if isinstance(message, str) else message
            # Deliberately not appended here. The loop announces it, and
            # `_reduce` appends it on the way through — keeping `_reduce` the
            # single writer to state.messages with no special case. It still
            # has to reach the provider on this turn, hence `messages` below.
            ctx = AgentContext(
                system_prompt=self.state.system_prompt,
                messages=(*self.state.messages, user),
                tools=self.state.tools,
                unannounced=(user,),
            )
            cfg = LoopConfig(
                model=self.state.model,
                provider=self._provider,
                config=self._config,
                cwd=self._cwd,
                before_tool_call=self._before_tool_call,
                after_tool_call=self._after_tool_call,
                prepare_request=self._prepare_request,
                finish_turn=self._finish_turn,
                signal=self._signal,
            )
            # The loop returns only what it produced; `_reduce` has already
            # appended each of those to state.messages via MessageEnd.
            return await run_agent_loop(ctx, cfg, self._emit)
        finally:
            self.state.is_running = False
            self._signal = None

    # --- control ----------------------------------------------------------

    def abort(self) -> None:
        """Stop the in-flight run. The transcript stays valid to continue from.

        The signal is cleared when the run ends, so the next `prompt()` starts
        clean rather than being born aborted.
        """
        if self._signal is not None:
            self._signal.set()
