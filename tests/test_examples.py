"""The examples run, each against a scripted model.

An example is documentation that executes, and one that falls behind the API
misleads exactly the reader it was written for. Each is loaded from its file
and run with `route` swapped for a `FakeProvider`, so nothing here needs a key
or a network.
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from lab_harness.agent import Agent
from lab_harness.agent.types import declarations_for
from lab_harness.config import load_config
from lab_harness.llm.fake import (
    FAKE_MODEL,
    FakeProvider,
    error_response,
    text_response,
    tool_call_response,
)
from lab_harness.llm.serde import to_dict
from lab_harness.llm.types import Message, TextBlock, ToolCallBlock, ToolResultMessage, UserMessage
from lab_harness.session import start_recorder
from lab_harness.tools import ToolContext, default_registry

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(autouse=True)
def _in_tmp_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Examples work on the current directory, as they do when run from a shell."""
    monkeypatch.chdir(tmp_path)


def load(name: str) -> ModuleType:
    """Import an example from its file; `examples/` is not a package."""
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scripted(monkeypatch: pytest.MonkeyPatch, module: ModuleType, *responses) -> FakeProvider:
    """Make the example's `route` hand back a provider replaying `responses`."""
    provider = FakeProvider(responses)
    monkeypatch.setattr(module, "route", lambda model_id, config: (FAKE_MODEL, provider))
    return provider


def call(name: str, **arguments: object) -> ToolCallBlock:
    return ToolCallBlock(id=f"call-{name}", name=name, arguments=arguments)


def results(provider: FakeProvider, request: int) -> list[ToolResultMessage]:
    """The tool results the given request (0-based) carried."""
    messages = provider.requests[request].messages
    return [m for m in messages if isinstance(m, ToolResultMessage)]


def joined(message: Message) -> str:
    assert not isinstance(message.content, str)
    return "".join(b.text for b in message.content if isinstance(b, TextBlock))


async def test_scripted_runs_a_real_tool_with_no_key(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    (tmp_path / "notes.txt").write_text("hi\n")
    await load("scripted").main()
    out = capsys.readouterr().out
    assert "ls returned:\nnotes.txt\n" in out
    assert "request 2: user, assistant, tool_result" in out


async def test_quickstart_streams_the_answer_and_shows_tool_calls(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load("quickstart")
    provider = scripted(
        monkeypatch,
        module,
        tool_call_response([call("ls")]),
        text_response("A small project.", chunk_size=4),
    )
    await module.main("What is this?")
    out = capsys.readouterr().out
    assert "→ ls({})" in out
    assert "A small project." in out
    assert "2 request(s)" in out
    assert [t.name for t in provider.requests[0].tools] == module.READ_ONLY_TOOLS


async def test_custom_tool_is_declared_and_counts_words(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "a.txt").write_text("The cat and the hat")
    module = load("custom_tool")
    provider = scripted(
        monkeypatch,
        module,
        tool_call_response([call("word_count", path="a.txt", top=1)]),
        text_response("Mostly 'the'."),
    )
    await module.main("Which word?")
    assert [t.name for t in provider.requests[0].tools] == ["ls", "glob", "word_count"]
    [result] = results(provider, 1)
    assert not result.is_error
    assert joined(result) == "5 words in a.txt\n2\tthe"


async def test_custom_tool_tells_the_model_what_to_do_when_it_cannot_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load("custom_tool")
    provider = scripted(
        monkeypatch,
        module,
        tool_call_response([call("word_count", path="missing.txt")]),
        text_response("No such file."),
    )
    await module.main("Count missing.txt")
    [result] = results(provider, 1)
    assert result.is_error
    assert "Use ls to find the file" in joined(result)


async def test_custom_tool_resolves_paths_against_its_context(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "a.txt").write_text("one two")
    word_count = load("custom_tool").word_count
    result = await word_count.execute({"path": "a.txt"}, ToolContext(cwd=str(project), call_id="c"))
    assert not result.is_error
    assert result.details == {"words": 2}


async def test_hooks_hold_the_budget_redact_log_and_self_check(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    (tmp_path / "team.txt").write_text("Ask ada@example.com\n")
    module = load("hooks")
    monkeypatch.setattr(module, "TOOL_BUDGET", 1)
    provider = scripted(
        monkeypatch,
        module,
        tool_call_response([call("read", path="team.txt"), call("ls")]),
        text_response("Ask Ada."),
        text_response("Checked: ask Ada."),
    )
    await module.main("Who do I ask?")

    read, ls = results(provider, 1)
    assert "ada@example.com" not in joined(read)
    assert "[email]" in joined(read)
    assert ls.is_error
    assert "budget of 1 tool calls is spent" in joined(ls)

    # The self-check is a real user message, sent once, and then the run ends.
    last = provider.requests[2].messages[-1]
    assert isinstance(last, UserMessage)
    assert joined(last) == module.SELF_CHECK
    assert len(provider.requests) == 3

    out = capsys.readouterr().out
    assert "[request 3: 6 message(s) to fake-model]" in out
    assert "assistant: Checked: ask Ada." in out


async def test_trajectories_exports_each_finished_request_as_sent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = load_config()
    provider = FakeProvider(
        [tool_call_response([call("ls")]), text_response("Done."), error_response("Overloaded")]
    )
    tools = default_registry().select(["ls"])
    agent = Agent(
        model=FAKE_MODEL, provider=provider, system_prompt="Be brief.", tools=tools, config=config
    )
    recorder = start_recorder(
        cwd=".",
        model=FAKE_MODEL,
        system_prompt="Be brief.",
        tools=declarations_for(tools),
        config=config,
        restored=0,
    )
    agent.subscribe(recorder)
    await agent.prompt("List the files.")
    await agent.prompt("And again.")  # errors, so it is left out
    recorder.close()

    assert load("trajectories").main(None) == 0
    exported = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    sent = [json.loads(json.dumps(to_dict(request))) for request in provider.requests]
    assert [line["request"] for line in exported] == sent[:2]


def test_trajectories_says_so_when_there_is_no_session(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert load("trajectories").main(None) == 1
    assert "No saved session" in capsys.readouterr().err
