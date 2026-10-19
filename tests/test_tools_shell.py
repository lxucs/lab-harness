"""The bash tool: output capture, limits, timeout, and cancellation.

These run real subprocesses — `sh`, `echo`, `sleep` — which is deliberate. The
parts most likely to break (process-group termination, timeout escalation,
streaming) cannot be exercised against a mock. No network is involved.
"""

import asyncio
import os
import time
from pathlib import Path

from lab_harness.config import BashConfig, Config, ToolsConfig
from lab_harness.llm.types import TextBlock
from lab_harness.tools.base import ToolContext, ToolResult
from lab_harness.tools.shell import _Collector, bash


def _ctx(tmp_path: Path, config: Config | None = None, **kwargs: object) -> ToolContext:
    return ToolContext(
        cwd=str(tmp_path),
        call_id="c1",
        config=config or Config(),
        **kwargs,  # type: ignore[arg-type]
    )


def _text(result: ToolResult) -> str:
    return "".join(b.text for b in result.content if isinstance(b, TextBlock))


def _with_bash(**kwargs: object) -> Config:
    return Config(tools=ToolsConfig(bash=BashConfig(**kwargs)))  # type: ignore[arg-type]


# --- basics ---------------------------------------------------------------


async def test_captures_stdout(tmp_path: Path) -> None:
    result = await bash.execute({"command": "echo hello"}, _ctx(tmp_path))

    assert not result.is_error
    assert _text(result) == "hello"
    assert result.details["exit_code"] == 0


async def test_captures_stderr_too(tmp_path: Path) -> None:
    """Combined streams: the model needs the error text, not just a code."""
    result = await bash.execute({"command": "echo oops >&2"}, _ctx(tmp_path))

    assert _text(result) == "oops"


async def test_a_nonzero_exit_is_an_error_result_with_the_code(tmp_path: Path) -> None:
    result = await bash.execute({"command": "echo out; exit 3"}, _ctx(tmp_path))

    assert result.is_error
    assert result.details["exit_code"] == 3
    assert "out" in _text(result)
    assert "exit status 3" in _text(result)


async def test_runs_in_ctx_cwd_not_the_process_cwd(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("")

    result = await bash.execute({"command": "ls"}, _ctx(tmp_path))

    assert "marker.txt" in _text(result)


async def test_reports_no_output_rather_than_an_empty_result(tmp_path: Path) -> None:
    result = await bash.execute({"command": "true"}, _ctx(tmp_path))

    assert _text(result) == "[no output]"
    assert not result.is_error


# --- limits ---------------------------------------------------------------


async def test_long_output_keeps_the_head_and_tail(tmp_path: Path) -> None:
    """Both ends usually matter; the middle rarely does."""
    config = _with_bash(max_output_bytes=400)
    command = "for i in $(seq 1 500); do echo line$i; done"

    result = await bash.execute({"command": command}, _ctx(tmp_path, config))

    body = _text(result)
    assert "line1\n" in body
    assert "line500" in body
    assert "elided" in body
    assert result.details["truncated"] is True


async def test_output_within_the_limit_is_not_marked_truncated(tmp_path: Path) -> None:
    result = await bash.execute({"command": "echo small"}, _ctx(tmp_path))

    assert result.details["truncated"] is False
    assert "elided" not in _text(result)


def test_collector_bounds_memory_regardless_of_volume() -> None:
    """The tail is a ring buffer, so a gigabyte of output costs the limit."""
    collector = _Collector(limit=100)
    for _ in range(1000):
        collector.feed(b"x" * 1000)

    rendered = collector.render()
    assert collector.truncated
    assert len(rendered) < 200


async def test_a_requested_timeout_is_capped_by_config(tmp_path: Path) -> None:
    """A model asking for an hour does not get an hour."""
    config = _with_bash(max_timeout_seconds=1)

    start = time.monotonic()
    result = await bash.execute(
        {"command": "sleep 30", "timeout_seconds": 3600}, _ctx(tmp_path, config)
    )
    elapsed = time.monotonic() - start

    assert result.is_error
    assert result.details["timed_out"] is True
    assert elapsed < 10


# --- timeout and cancellation ---------------------------------------------


async def test_a_slow_command_times_out_and_is_terminated(tmp_path: Path) -> None:
    config = _with_bash(timeout_seconds=1)

    result = await bash.execute({"command": "sleep 30"}, _ctx(tmp_path, config))

    assert result.is_error
    assert result.details["timed_out"] is True
    assert "timed out" in _text(result)


async def test_output_before_a_timeout_is_still_returned(tmp_path: Path) -> None:
    config = _with_bash(timeout_seconds=1)

    result = await bash.execute({"command": "echo partial; sleep 30"}, _ctx(tmp_path, config))

    assert "partial" in _text(result)
    assert "timed out" in _text(result)


async def test_an_abort_signal_stops_the_command(tmp_path: Path) -> None:
    signal_event = asyncio.Event()
    config = _with_bash(update_interval_seconds=0.05)
    ctx = _ctx(tmp_path, config, signal=signal_event)

    async def abort_soon() -> None:
        await asyncio.sleep(0.2)
        signal_event.set()

    start = time.monotonic()
    result, _ = await asyncio.gather(
        bash.execute({"command": "sleep 30"}, ctx),
        abort_soon(),
    )
    elapsed = time.monotonic() - start

    assert result.is_error
    assert result.details["aborted"] is True
    assert elapsed < 10


async def test_termination_kills_the_whole_process_group(tmp_path: Path) -> None:
    """Killing only the shell would orphan whatever it started.

    The command writes its grandchild's pid to a file, then the tool is
    cancelled; afterwards that pid must be gone.
    """
    pid_file = tmp_path / "child.pid"
    command = f"sh -c 'echo $$ > {pid_file}; sleep 30' & wait"
    signal_event = asyncio.Event()
    config = _with_bash(update_interval_seconds=0.05)
    ctx = _ctx(tmp_path, config, signal=signal_event)

    async def abort_once_started() -> None:
        for _ in range(100):
            if pid_file.exists() and pid_file.read_text().strip():
                break
            await asyncio.sleep(0.05)
        signal_event.set()

    result, _ = await asyncio.gather(
        bash.execute({"command": command}, ctx),
        abort_once_started(),
    )

    assert result.details["aborted"] is True
    child_pid = int(pid_file.read_text().strip())

    # Give the signal a moment to land, then confirm the grandchild is gone.
    for _ in range(40):
        if not _alive(child_pid):
            break
        await asyncio.sleep(0.05)
    assert not _alive(child_pid), f"pid {child_pid} survived termination"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def test_cancelling_the_tool_task_does_not_leave_the_process_running(
    tmp_path: Path,
) -> None:
    """CancelledError must propagate *and* clean up."""
    pid_file = tmp_path / "child.pid"
    ctx = _ctx(tmp_path, _with_bash(update_interval_seconds=0.05))
    command = f"echo $$ > {pid_file}; sleep 30"

    task = asyncio.create_task(bash.execute({"command": command}, ctx))
    for _ in range(100):
        if pid_file.exists() and pid_file.read_text().strip():
            break
        await asyncio.sleep(0.05)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("cancellation must propagate")

    shell_pid = int(pid_file.read_text().strip())
    for _ in range(40):
        if not _alive(shell_pid):
            break
        await asyncio.sleep(0.05)
    assert not _alive(shell_pid)


# --- streaming ------------------------------------------------------------


async def test_output_streams_through_on_update(tmp_path: Path) -> None:
    """A long build should be visible while it runs, not only after."""
    updates: list[str] = []
    config = _with_bash(update_interval_seconds=0.05)
    ctx = _ctx(
        tmp_path,
        config,
        on_update=lambda partial: updates.append(
            "".join(b.text for b in partial.content if isinstance(b, TextBlock))
        ),
    )

    result = await bash.execute({"command": "echo first; sleep 0.3; echo second"}, ctx)

    assert "first" in _text(result)
    assert "second" in _text(result)
    assert updates, "expected at least one streaming update"
    assert "first" in updates[0]


# --- declaration ----------------------------------------------------------


def test_bash_is_sequential() -> None:
    """Concurrent shell commands mutate shared state unpredictably."""
    assert bash.execution_mode == "sequential"
