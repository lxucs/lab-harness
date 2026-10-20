"""The approval gate: fail closed, remember only within a session."""

import asyncio
import logging
from io import StringIO
from pathlib import Path

import pytest
from pydantic import BaseModel
from rich.console import Console

from lab_harness.agent import Agent
from lab_harness.agent.types import BeforeToolCallContext, EventSink, ToolExecutionEnd
from lab_harness.cli.approval import DENIED_NOTICE, ApprovalGate, skill_allowances
from lab_harness.config import ApprovalConfig, Config
from lab_harness.llm.fake import FAKE_MODEL, FakeProvider, text_response, tool_call_response
from lab_harness.llm.types import TextBlock, ToolCallBlock, UserMessage, text
from lab_harness.tools.base import ToolContext, ToolResult, ok, tool
from lab_harness.tools.skills import skill


class _Params(BaseModel):
    pass


@tool(name="dangerous", description="mutates things", execution_mode="sequential")
async def dangerous(p: _Params, ctx: ToolContext) -> ToolResult:
    return ok("executed")


@tool(name="harmless", description="reads things")
async def harmless(p: _Params, ctx: ToolContext) -> ToolResult:
    return ok("read")


@tool(name="bash", description="runs commands", execution_mode="sequential")
async def bash(p: _Params, ctx: ToolContext) -> ToolResult:
    return ok("ran")


def _ctx(tool_name: str) -> BeforeToolCallContext:
    return BeforeToolCallContext(
        tool_call=ToolCallBlock(id="c1", name=tool_name, arguments={}),
        messages=(UserMessage(content=text("go")),),
    )


def _gate(answers: list[str], required: tuple[str, ...] = ("dangerous",), **kwargs: object):
    asked: list[str] = []

    async def ask(name: str, arguments: dict[str, object]) -> str:
        asked.append(name)
        return answers.pop(0) if answers else "no"

    config = ApprovalConfig(required=required, **kwargs)  # type: ignore[arg-type]
    return ApprovalGate(config, ask), asked


# --- the decision ---------------------------------------------------------


async def test_an_unlisted_tool_is_never_prompted() -> None:
    gate, asked = _gate([])

    assert await gate(_ctx("harmless")) is None
    assert asked == []


async def test_yes_allows_the_call() -> None:
    gate, asked = _gate(["yes"])

    assert await gate(_ctx("dangerous")) is None
    assert asked == ["dangerous"]


async def test_no_blocks_the_call_with_a_reason() -> None:
    gate, _ = _gate(["no"])

    decision = await gate(_ctx("dangerous"))

    assert decision is not None
    assert decision.block
    assert decision.reason == DENIED_NOTICE


@pytest.mark.parametrize("answer", ["", "maybe", "Y E S", "yesno", "\n", "quit"])
async def test_anything_but_an_explicit_yes_denies(answer: str) -> None:
    """Fail closed. An unrecognized answer is not consent."""
    gate, _ = _gate([answer])

    decision = await gate(_ctx("dangerous"))

    assert decision is not None and decision.block


async def test_always_allows_and_is_remembered_for_the_session() -> None:
    gate, asked = _gate(["always"])

    assert await gate(_ctx("dangerous")) is None
    assert await gate(_ctx("dangerous")) is None
    assert await gate(_ctx("dangerous")) is None

    assert asked == ["dangerous"], "should have been asked exactly once"


async def test_remembering_can_be_disabled() -> None:
    gate, asked = _gate(["always", "always"], allow_remember=False)

    await gate(_ctx("dangerous"))
    await gate(_ctx("dangerous"))

    assert asked == ["dangerous", "dangerous"]


async def test_remembering_one_tool_does_not_authorize_another() -> None:
    gate, asked = _gate(["always", "no"], required=("dangerous", "other"))

    await gate(_ctx("dangerous"))
    decision = await gate(_ctx("other"))

    assert decision is not None and decision.block
    assert asked == ["dangerous", "other"]


async def test_a_new_gate_starts_with_no_memory() -> None:
    """`always` is per-session, never persisted."""
    first, _ = _gate(["always"])
    await first(_ctx("dangerous"))

    second, asked = _gate(["no"])
    decision = await second(_ctx("dangerous"))

    assert decision is not None and decision.block
    assert asked == ["dangerous"]


# --- concurrency ----------------------------------------------------------


async def test_concurrent_prompts_are_serialized() -> None:
    """Two questions on one terminal would interleave and read one answer."""
    in_flight = 0
    peak = 0

    async def ask(name: str, arguments: dict[str, object]) -> str:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return "yes"

    gate = ApprovalGate(ApprovalConfig(required=("dangerous",)), ask)

    await asyncio.gather(*(gate(_ctx("dangerous")) for _ in range(5)))

    assert peak == 1


async def test_an_always_answer_short_circuits_waiting_siblings() -> None:
    """A sibling that queued behind 'always' must not be asked again."""
    asked: list[str] = []

    async def ask(name: str, arguments: dict[str, object]) -> str:
        asked.append(name)
        await asyncio.sleep(0.01)
        return "always"

    gate = ApprovalGate(ApprovalConfig(required=("dangerous",)), ask)

    results = await asyncio.gather(*(gate(_ctx("dangerous")) for _ in range(4)))

    assert asked == ["dangerous"]
    assert all(r is None for r in results)


# --- through the loop -----------------------------------------------------


async def test_a_denied_tool_does_not_execute_and_the_model_is_told() -> None:
    call = ToolCallBlock(id="c1", name="dangerous", arguments={})
    provider = FakeProvider([tool_call_response([call]), text_response("understood")])
    gate, _ = _gate(["no"])
    agent = Agent(
        model=FAKE_MODEL,
        provider=provider,
        tools=(dangerous,),
        config=Config(),
        before_tool_call=gate,
    )

    await agent.prompt("do it")

    results = [m for m in agent.state.messages if m.role == "tool_result"]
    assert len(results) == 1
    assert results[0].is_error
    body = "".join(b.text for b in results[0].content if isinstance(b, TextBlock))
    assert body == DENIED_NOTICE
    assert "executed" not in body


async def test_a_crashing_asker_denies_rather_than_falling_open() -> None:
    """The loop converts a raising hook into a blocked call. Verify end to end."""

    async def ask(name: str, arguments: dict[str, object]) -> str:
        raise RuntimeError("terminal exploded")

    gate = ApprovalGate(ApprovalConfig(required=("dangerous",)), ask)
    call = ToolCallBlock(id="c1", name="dangerous", arguments={})
    provider = FakeProvider([tool_call_response([call]), text_response("ok")])
    agent = Agent(
        model=FAKE_MODEL,
        provider=provider,
        tools=(dangerous,),
        config=Config(),
        before_tool_call=gate,
    )

    await agent.prompt("do it")

    results = [m for m in agent.state.messages if m.role == "tool_result"]
    body = "".join(b.text for b in results[0].content if isinstance(b, TextBlock))
    assert results[0].is_error
    assert "executed" not in body


async def test_the_default_required_set_covers_the_mutating_tools() -> None:
    defaults = ApprovalConfig()
    assert set(defaults.required) == {"bash", "write", "edit"}


# --- what is logged -------------------------------------------------------


async def test_each_answer_is_logged_against_its_call(caplog: pytest.LogCaptureFixture) -> None:
    gate, _ = _gate(["always"])

    with caplog.at_level(logging.DEBUG, logger="lab_harness.cli.approval"):
        await gate(_ctx("dangerous"))

    assert "Tool call c1: approval answered 'always'" in caplog.text


# --- a skill's allowed-tools ---------------------------------------------


def _end(details: dict[str, object], tool_name: str = "skill") -> ToolExecutionEnd:
    return ToolExecutionEnd(
        tool_call_id="c1",
        tool_name=tool_name,
        content=(),
        is_error=False,
        duration_ms=1,
        details=details,
    )


def _listener(gate: ApprovalGate, buffer: StringIO) -> EventSink:
    return skill_allowances(gate, Console(file=buffer, width=200), "dim")


async def test_a_skill_grants_the_tools_it_names() -> None:
    gate, asked = _gate([])

    await _listener(gate, StringIO())(_end({"allowed_tools": ["dangerous"]}))

    assert await gate(_ctx("dangerous")) is None
    assert asked == []


async def test_only_the_skill_tool_grants() -> None:
    """Another tool's result, or one that names none, changes nothing."""
    gate, asked = _gate(["yes"])
    listener = _listener(gate, StringIO())

    await listener(_end({"allowed_tools": ["dangerous"]}, tool_name="read"))
    await listener(_end({}))

    await gate(_ctx("dangerous"))

    assert asked == ["dangerous"]


async def test_the_grant_is_said() -> None:
    """A permission granted silently is the kind of thing that surprises later."""
    buffer = StringIO()
    gate, _ = _gate([])

    await _listener(gate, buffer)(_end({"allowed_tools": ["dangerous"]}))

    assert "dangerous allowed for this session" in buffer.getvalue()


async def test_remembering_off_suppresses_the_grant() -> None:
    """An operator who turned remembering off wants to be asked."""
    gate, asked = _gate(["yes"], allow_remember=False)

    await _listener(gate, StringIO())(_end({"allowed_tools": ["dangerous"]}))

    assert await gate(_ctx("dangerous")) is None
    assert asked == ["dangerous"]


async def test_a_loaded_skill_lets_the_loop_run_its_tools(tmp_path: Path) -> None:
    """End to end: the grant reaches the gate through the event stream."""
    skills = tmp_path / ".lab-harness" / "skills" / "commit"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text(
        "---\nname: commit\ndescription: Write one\nallowed-tools: Bash\n---\nDo it.\n",
        encoding="utf-8",
    )
    provider = FakeProvider(
        [
            tool_call_response(
                [ToolCallBlock(id="c1", name="skill", arguments={"name": "commit"})]
            ),
            tool_call_response([ToolCallBlock(id="c2", name="bash", arguments={})]),
            text_response("done"),
        ]
    )
    gate, asked = _gate([], required=("bash",))
    agent = Agent(
        model=FAKE_MODEL,
        provider=provider,
        tools=(skill, bash),
        config=Config(),
        before_tool_call=gate,
        cwd=str(tmp_path),
    )
    agent.subscribe(_listener(gate, StringIO()))

    await agent.prompt("commit this")

    assert asked == [], "the skill granted bash, so it should not have been asked about"
