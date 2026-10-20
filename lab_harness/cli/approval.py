"""Tool approval, as a `before_tool_call` hook.

Not loop logic — the loop knows only that a hook may refuse a call. That
separation is what lets `-p` mode run without a gate at all and the terminal run
with one, without either path branching inside the loop.

**Fail closed.** Anything other than an explicit yes denies. The loop already
converts a hook that raises into a blocked call, so a crash here denies too.

The `always` answer is remembered for the session and never persisted: a
decision made in one run should not silently authorize the next one. A skill's
`allowed-tools` is remembered the same way — see `skill_allowances` — but
comes from a file the user installed rather than from an answer at a prompt.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable

from rich.console import Console
from rich.text import Text

from lab_harness.agent.types import (
    AgentEvent,
    BeforeToolCallContext,
    BeforeToolCallResult,
    EventSink,
)
from lab_harness.config import ApprovalConfig
from lab_harness.tools.skills import skill

#: What a prompt returns. "always" implies "yes" plus remembering the tool.
Answer = str
Asker = Callable[[str, dict[str, object]], Awaitable[Answer]]

DENIED_NOTICE = "The user denied permission to run this tool."
#: What a tool result carries its granted tools under.
ALLOWED_TOOLS = "allowed_tools"

_log = logging.getLogger(__name__)


class ApprovalGate:
    """Decides whether a tool call may proceed, asking the user when required."""

    def __init__(self, config: ApprovalConfig, ask: Asker) -> None:
        self._config = config
        self._ask = ask
        self._remembered: set[str] = set()
        # Serializes prompts: a parallel batch would otherwise interleave two
        # questions on one terminal and read one answer for both.
        self._lock = asyncio.Lock()

    def requires_approval(self, tool_name: str) -> bool:
        return tool_name in self._config.required and tool_name not in self._remembered

    def allow(self, tool_names: Iterable[str]) -> tuple[str, ...]:
        """Grant these tools for the rest of the session, as "always" would.

        Returns what was granted, which is nothing when
        `approval.allow_remember` is false: an operator who turned remembering
        off wants to be asked, and a file should not quietly undo that.
        """
        names = tuple(tool_names)
        if not self._config.allow_remember:
            _log.warning(
                "Not allowing %s: approval.allow_remember is false", ", ".join(names) or "nothing"
            )
            return ()
        self._remembered.update(names)
        _log.debug("Allowed for this session: %s", ", ".join(names))
        return names

    async def __call__(self, ctx: BeforeToolCallContext) -> BeforeToolCallResult | None:
        name = ctx.tool_call.name
        if not self.requires_approval(name):
            return None

        async with self._lock:
            # Re-check inside the lock: a sibling call may have been answered
            # with "always" while this one waited.
            if not self.requires_approval(name):
                return None
            answer = await self._ask(name, ctx.tool_call.arguments)
        _log.debug("Tool call %s: approval answered %r", ctx.tool_call.id, answer)

        if answer == "always":
            if self._config.allow_remember:
                self._remembered.add(name)
            return None
        if answer == "yes":
            return None
        return BeforeToolCallResult(block=True, reason=DENIED_NOTICE)


def skill_allowances(gate: ApprovalGate, console: Console, style: str) -> EventSink:
    """A subscriber applying a loaded skill's `allowed-tools` to `gate`.

    The names reach it in the `skill` result's `details` — the channel a result
    uses for what a renderer needs and the model does not. A tool cannot reach
    the gate, and the gate is not the loop's to know about, so this is wiring
    beside the recorder rather than a branch anywhere in the runtime.

    The grant lands when the call ends, so a tool the model called alongside
    the skill in the same block is still asked about.
    """

    async def listener(event: AgentEvent) -> None:
        if event.type != "tool_execution_end" or event.tool_name != skill.name:
            return
        named = event.details.get(ALLOWED_TOOLS)
        if not isinstance(named, (list, tuple)):
            return
        if granted := gate.allow(str(name) for name in named):
            console.print(
                Text(f"{skill.name}: {', '.join(granted)} allowed for this session", style=style)
            )

    return listener
