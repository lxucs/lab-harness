"""The JSONL sink, and the end-to-end run it makes assertable."""

import io
import json

from harness.agent import Agent
from harness.agent.types import AgentEvent
from harness.cli.json_sink import json_sink
from harness.llm.fake import FAKE_MODEL, FakeProvider, text_response, tool_call_response
from harness.llm.types import ToolCallBlock
from harness.tools import ToolRegistry
from harness.tools.fs import FILESYSTEM_TOOLS


def _lines(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


async def test_each_event_is_one_json_line() -> None:
    stream = io.StringIO()
    agent = Agent(model=FAKE_MODEL, provider=FakeProvider([text_response("hi")]))
    agent.subscribe(json_sink(stream))

    await agent.prompt("go")

    kinds = [r["type"] for r in _lines(stream)]
    assert kinds[0] == "agent_start"
    assert kinds[-1] == "agent_end"


async def test_the_stream_is_valid_json_throughout_a_tool_run(tmp_path) -> None:
    """The point of the sink: a structured surface to assert on."""
    (tmp_path / "a.txt").write_text("hello\n")
    call = ToolCallBlock(id="c1", name="read", arguments={"path": "a.txt"})
    provider = FakeProvider([tool_call_response([call]), text_response("done")])

    stream = io.StringIO()
    agent = Agent(
        model=FAKE_MODEL,
        provider=provider,
        tools=tuple(ToolRegistry(FILESYSTEM_TOOLS)),
        cwd=str(tmp_path),
    )
    agent.subscribe(json_sink(stream))

    await agent.prompt("read it")

    records = _lines(stream)
    kinds = [r["type"] for r in records]
    assert kinds.count("turn_start") == 2
    assert "tool_execution_end" in kinds

    end = next(r for r in records if r["type"] == "tool_execution_end")
    assert end["is_error"] is False
    assert "hello" in json.dumps(end["content"])


async def test_sink_output_survives_a_round_trip_back_into_events() -> None:
    """Serialized events must be re-readable — that is what a second consumer does."""
    from harness.llm.serde import message_from_dict

    stream = io.StringIO()
    agent = Agent(model=FAKE_MODEL, provider=FakeProvider([text_response("hello")]))
    agent.subscribe(json_sink(stream))

    await agent.prompt("go")

    ends = [r for r in _lines(stream) if r["type"] == "message_end"]
    payload = ends[-1]["message"]
    assert isinstance(payload, dict)
    assert message_from_dict(payload) == agent.state.messages[-1]


async def test_a_failing_sink_does_not_break_the_run() -> None:
    class Exploding(io.StringIO):
        def write(self, s: str) -> int:
            raise OSError("pipe closed")

    agent = Agent(model=FAKE_MODEL, provider=FakeProvider([text_response("survived")]))
    agent.subscribe(json_sink(Exploding()))

    messages = await agent.prompt("go")

    assert messages[-1].role == "assistant"


async def test_events_carry_no_live_objects() -> None:
    """json.dumps is the enforcement; this asserts the sink relies on it."""
    stream = io.StringIO()
    agent = Agent(model=FAKE_MODEL, provider=FakeProvider([text_response("hi")]))
    sink = json_sink(stream)

    class NotAnEvent:
        type = "nope"

    try:
        await sink(NotAnEvent())  # type: ignore[arg-type]
    except TypeError as exc:
        assert "protocol dataclass" in str(exc)
    else:
        raise AssertionError("a non-dataclass must not serialize")

    del agent


async def test_default_stream_is_stdout(capsys) -> None:
    agent = Agent(model=FAKE_MODEL, provider=FakeProvider([text_response("hi")]))
    agent.subscribe(json_sink())

    await agent.prompt("go")

    captured = capsys.readouterr().out
    assert json.loads(captured.splitlines()[0])["type"] == "agent_start"


def _unused(event: AgentEvent) -> None:  # pragma: no cover - import sanity only
    pass
