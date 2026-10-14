"""The bash tool.

Three things make this harder than "run a subprocess":

**Streaming.** Output is forwarded through `ctx.on_update` as it arrives, so a
long build is visible while it runs rather than after.

**Cancellation.** `ctx.signal` must actually stop the process. We terminate the
whole process *group*, not just the shell — otherwise `sh -c "sleep 100"` leaves
the sleep running after the shell dies.

**Bounded output.** A command that prints a gigabyte must not end up in the
context window. Output is capped with the head and tail kept and the middle
elided, because both ends usually matter and the middle rarely does.
"""

import asyncio
import contextlib
import os
import signal as signal_module

from pydantic import BaseModel, Field

from lab_harness.llm.types import text
from lab_harness.tools.base import ToolContext, ToolResult, fail, ok, tool

_ELISION = "\n\n[... {skipped} bytes elided ...]\n\n"
_TIMEOUT_NOTICE = "Command timed out after {seconds}s and was terminated."
_ABORTED_NOTICE = "Command was aborted."
_EXIT_NOTICE = "\n[exit status {code}]"
#: Grace period between SIGTERM and SIGKILL, in seconds.
_KILL_GRACE_SECONDS = 2.0


class BashParams(BaseModel):
    command: str = Field(description="The shell command to run.")
    timeout_seconds: int | None = Field(
        default=None,
        ge=1,
        description="Wall-clock limit. Defaults to the configured timeout.",
    )


@tool(
    name="bash",
    description=(
        "Run a shell command and return its combined stdout and stderr. "
        "Runs in the working directory. Long output is truncated in the middle."
    ),
    # Sequential: concurrent shell commands mutate shared state — the
    # filesystem, the package index, a running server — in ways nobody can
    # reason about after the fact.
    execution_mode="sequential",
)
async def bash(p: BashParams, ctx: ToolContext) -> ToolResult:
    limits = ctx.config.tools.bash
    timeout = min(p.timeout_seconds or limits.timeout_seconds, limits.max_timeout_seconds)

    try:
        process = await asyncio.create_subprocess_shell(
            p.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=ctx.cwd,
            # Own process group, so cancellation reaches children too.
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        return fail(f"Could not start command: {exc}")

    collector = _Collector(limit=limits.max_output_bytes)
    outcome = await _pump(process, collector, ctx, timeout, limits.update_interval_seconds)

    body = collector.render()
    if outcome == "timeout":
        notice = _TIMEOUT_NOTICE.format(seconds=timeout)
        return ToolResult(
            content=text(f"{body}\n{notice}" if body else notice),
            is_error=True,
            details={"timed_out": True, "command": p.command},
        )
    if outcome == "aborted":
        return ToolResult(
            content=text(f"{body}\n{_ABORTED_NOTICE}" if body else _ABORTED_NOTICE),
            is_error=True,
            details={"aborted": True, "command": p.command},
        )

    code = process.returncode or 0
    if code != 0:
        body = (
            f"{body}{_EXIT_NOTICE.format(code=code)}"
            if body
            else (_EXIT_NOTICE.format(code=code).strip())
        )
    return ToolResult(
        content=text(body or "[no output]"),
        is_error=code != 0,
        details={"exit_code": code, "command": p.command, "truncated": collector.truncated},
    )


async def _pump(
    process: asyncio.subprocess.Process,
    collector: "_Collector",
    ctx: ToolContext,
    timeout: float,
    update_interval: float,
) -> str:
    """Read output until the process ends, the timeout fires, or abort is signalled.

    Returns "done", "timeout", or "aborted".
    """
    assert process.stdout is not None
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    last_update = loop.time()

    while True:
        if ctx.signal is not None and ctx.signal.is_set():
            await _terminate(process)
            return "aborted"

        remaining = deadline - loop.time()
        if remaining <= 0:
            await _terminate(process)
            return "timeout"

        try:
            # Short waits rather than one long read, so the abort check above
            # runs regularly instead of only when output happens to arrive.
            chunk = await asyncio.wait_for(
                process.stdout.read(4096), timeout=min(remaining, update_interval)
            )
        except TimeoutError:
            continue
        except asyncio.CancelledError:
            # The run is being torn down; do not leave the process behind.
            await _terminate(process)
            raise

        if not chunk:
            break

        collector.feed(chunk)
        now = loop.time()
        if ctx.on_update is not None and now - last_update >= update_interval:
            last_update = now
            ctx.on_update(ok(collector.render()))

    try:
        await asyncio.wait_for(process.wait(), timeout=_KILL_GRACE_SECONDS)
    except TimeoutError:
        await _terminate(process)
    return "done"


async def _terminate(process: asyncio.subprocess.Process) -> None:
    """Stop the process group, escalating to SIGKILL if it ignores SIGTERM.

    The group, not the process: `create_subprocess_shell` spawns a shell, and
    killing only the shell orphans whatever it started.
    """
    if process.returncode is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal_module.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        # Already gone, or we cannot signal the group — fall back to the
        # process itself and let the kill below handle a stubborn child.
        try:
            process.terminate()
        except ProcessLookupError:
            return

    try:
        await asyncio.wait_for(process.wait(), timeout=_KILL_GRACE_SECONDS)
        return
    except TimeoutError:
        pass

    try:
        os.killpg(os.getpgid(process.pid), signal_module.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.kill()
        except ProcessLookupError:
            return
    # Reap it, so the transport does not warn about an abandoned child.
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(process.wait(), timeout=_KILL_GRACE_SECONDS)


class _Collector:
    """Accumulates output, keeping the head and tail within a byte budget.

    Streaming means we cannot know the total in advance, so the tail is a ring
    buffer: everything past the head budget is appended and trimmed from the
    front, which keeps memory bounded no matter how much a command prints.
    """

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._head_budget = limit // 2
        self._head = bytearray()
        self._tail = bytearray()
        self._skipped = 0

    def feed(self, chunk: bytes) -> None:
        if len(self._head) < self._head_budget:
            room = self._head_budget - len(self._head)
            self._head += chunk[:room]
            chunk = chunk[room:]
            if not chunk:
                return
        self._tail += chunk
        overflow = len(self._tail) - (self._limit - self._head_budget)
        if overflow > 0:
            del self._tail[:overflow]
            self._skipped += overflow

    @property
    def truncated(self) -> bool:
        return self._skipped > 0

    def render(self) -> str:
        head = self._head.decode("utf-8", errors="replace")
        tail = self._tail.decode("utf-8", errors="replace")
        if not self._skipped:
            return (head + tail).rstrip()
        return (head + _ELISION.format(skipped=self._skipped) + tail).rstrip()


SHELL_TOOLS = (bash,)
