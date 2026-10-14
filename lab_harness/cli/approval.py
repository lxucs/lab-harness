"""Tool approval, as a `before_tool_call` hook.

Not loop logic — the loop knows only that a hook may refuse a call. That
separation is what lets `-p` mode run without a gate at all and the terminal run
with one, without either path branching inside the loop.

**Fail closed.** Anything other than an explicit yes denies. The loop already
converts a hook that raises into a blocked call, so a crash here denies too.

The `always` answer is remembered for the session and never persisted: a
decision made in one run should not silently authorize the next one.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from lab_harness.agent.types import BeforeToolCallContext, BeforeToolCallResult
from lab_harness.config import ApprovalConfig

#: What a prompt returns. "always" implies "yes" plus remembering the tool.
Answer = str
Asker = Callable[[str, dict[str, object]], Awaitable[Answer]]

DENIED_NOTICE = "The user denied permission to run this tool."

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
