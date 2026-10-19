"""CLI wiring: argument parsing, mode selection, and what gets attached.

The agent is built with a real `AnthropicProvider`, which never connects here
because nothing calls `stream()` — construction alone makes no request.
"""

import argparse
import asyncio
import logging
import sys
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from lab_harness.agent import Agent
from lab_harness.cli.approval import ApprovalGate
from lab_harness.cli.main import (
    _attach_session,
    _brief,
    _build_agent,
    build_parser,
    main,
    run_interactive,
)
from lab_harness.cli.terminal import LogStream
from lab_harness.config import (
    CONFIG_FILE,
    PROJECT_DIRECTORY,
    AgentConfig,
    ApprovalConfig,
    Config,
    ConfigError,
    ProjectConfig,
    PromptsConfig,
    ToolsConfig,
    load_config,
    snapshot,
)
from lab_harness.llm.fake import FAKE_MODEL, FakeProvider, text_response
from lab_harness.llm.providers.chat_completions import ChatCompletionsProvider
from lab_harness.llm.registry import resolve_model
from lab_harness.logs import ROOT_LOGGER
from lab_harness.prompts import build_system_prompt, detect_environment
from lab_harness.session import list_sessions, read_transcript, recorded_requests
from lab_harness.tools import default_registry


def _args(**kwargs: object) -> argparse.Namespace:
    defaults = {
        "prompt": None,
        "json": False,
        "model": None,
        "provider": None,
        "reasoning_effort": None,
        "cwd": ".",
        "tools": None,
        "system_prompt": None,
        "system_prompt_file": None,
        "yes": False,
        "resume": None,
        # Tests must never write to the real ~/.lab-harness.
        "no_save": True,
    }
    return argparse.Namespace(**{**defaults, **kwargs})


class _StubTerminal:
    """Stands in for Terminal: records the attached agent, runs nothing.

    Mirrors the real surface run_interactive touches — attach, run, and
    ask_approval, which build_approval_gate binds into the gate.
    """

    def __init__(self, captured: dict[str, Agent]) -> None:
        self._captured = captured

    def attach(self, agent: Agent) -> None:
        self._captured["agent"] = agent

    async def run(self) -> int:
        return 0

    async def ask_approval(self, tool_name: str, arguments: dict[str, object]) -> str:
        return "no"


# --- parsing --------------------------------------------------------------


def test_flags_parse() -> None:
    args = build_parser().parse_args(
        ["-p", "hi", "--json", "--model", "m", "--cwd", "/tmp", "--provider", "deepseek"]
    )

    assert args.prompt == "hi"
    assert args.json is True
    assert args.model == "m"
    assert args.cwd == "/tmp"
    assert args.provider == "deepseek"


def test_the_overriding_flags_default_to_unset() -> None:
    """None, not a value — anything else would outrank config permanently."""
    args = build_parser().parse_args([])

    assert args.model is None
    assert args.provider is None
    assert args.reasoning_effort is None
    assert args.log_level is None
    assert args.tools is None
    assert args.system_prompt is None
    assert args.system_prompt_file is None


def test_yes_flag_defaults_off() -> None:
    """Approval must be opt-out, never opt-in."""
    assert build_parser().parse_args([]).yes is False
    assert build_parser().parse_args(["--yes"]).yes is True


# --- agent construction ---------------------------------------------------


@pytest.mark.parametrize("prompt", ["hi", None], ids=["one-shot", "interactive"])
def test_the_configured_tool_set_narrows_either_mode(prompt: str | None) -> None:
    """One-shot runs have no approval gate, so leaving a tool out is what stops it."""
    config = Config(tools=ToolsConfig(enabled=("read", "ls")))

    agent = _build_agent(_args(prompt=prompt), config)

    assert [t.name for t in agent.state.tools] == ["read", "ls"]


def test_the_flag_replaces_the_configured_tool_set() -> None:
    config = Config(tools=ToolsConfig(enabled=("read", "ls")))

    agent = _build_agent(_args(tools=("grep", "read")), config)

    assert [t.name for t in agent.state.tools] == ["grep", "read"]


def test_an_empty_tool_set_gives_the_model_none() -> None:
    agent = _build_agent(_args(tools=()), Config())

    assert agent.state.tools == ()


def test_the_flag_takes_a_comma_separated_list() -> None:
    parse = build_parser().parse_args

    assert parse(["--tools", "read, ls,,grep"]).tools == ("read", "ls", "grep")
    assert parse(["--tools", ""]).tools == ()
    # Absent is not empty: absent defers to tools.enabled.
    assert parse([]).tools is None


@pytest.mark.parametrize(
    ("args", "config", "source"),
    [
        (_args(), Config(tools=ToolsConfig(enabled=("read", "typo"))), "tools.enabled"),
        (_args(tools=("read", "typo")), Config(), "--tools"),
    ],
    ids=["config", "flag"],
)
def test_an_unknown_tool_is_a_config_error_naming_its_source(
    args: argparse.Namespace, config: Config, source: str
) -> None:
    """A typo should fail at wiring, not silently disable a tool."""
    with pytest.raises(ConfigError) as caught:
        _build_agent(args, config)

    message = str(caught.value)
    assert message.startswith(f"{source}: ")
    assert "typo" in message
    assert "bash" in message, "the message must list the way out"


def test_an_unknown_tool_is_reported_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["-p", "hi", "--tools", "typo", "--cwd", str(tmp_path), "--no-save"]) == 1
    assert "error: --tools: Unknown tool(s): typo" in capsys.readouterr().err


def test_the_tools_and_their_source_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="lab_harness.cli.main"):
        _build_agent(_args(tools=("read", "ls")), Config())
        _build_agent(_args(tools=()), Config())

    assert "Tools from --tools: read, ls" in caplog.messages
    assert "Tools from --tools: none" in caplog.messages


def test_one_shot_has_no_approval_gate() -> None:
    """There is nobody at the keyboard to answer a prompt."""
    agent = _build_agent(_args(prompt="hi"), Config())

    assert agent._before_tool_call is None


def test_cwd_reaches_the_agent() -> None:
    agent = _build_agent(_args(cwd="/tmp/somewhere"), Config())

    assert agent._cwd == "/tmp/somewhere"


def test_the_configured_model_reaches_the_agent() -> None:
    """Without a flag, `agent.model` decides — not a constant in the code."""
    config = Config(agent=AgentConfig(model="claude-haiku-4-5-20251001"))

    agent = _build_agent(_args(), config)

    assert agent.state.model.id == "claude-haiku-4-5-20251001"


def test_the_model_flag_beats_the_configured_model() -> None:
    config = Config(agent=AgentConfig(model="claude-haiku-4-5-20251001"))

    agent = _build_agent(_args(model="claude-opus-5"), config)

    assert agent.state.model.id == "claude-opus-5"


@pytest.mark.parametrize(("flag", "sent"), [("low", "low"), ("", "")], ids=["level", "empty"])
def test_the_effort_flag_beats_the_configured_one(flag: str, sent: str) -> None:
    """An empty flag sends no level, whatever config says."""
    config = Config(agent=AgentConfig(reasoning_effort="high"))

    assert _build_agent(_args(), config).state.model.reasoning_effort == "high"
    assert _build_agent(_args(reasoning_effort=flag), config).state.model.reasoning_effort == sent


# --- project instructions -------------------------------------------------


def test_no_project_file_means_the_configured_sections(tmp_path: Path) -> None:
    agent = _build_agent(_args(), Config())

    environment = detect_environment(tmp_path, agent.state.model.id)
    assert agent.state.system_prompt == build_system_prompt(Config().prompts, environment)


def test_the_environment_is_filled_from_cwd_and_the_model(tmp_path: Path) -> None:
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()

    agent = _build_agent(_args(cwd=str(elsewhere), model="claude-opus-5"), Config())

    assert f"Primary working directory: {elsewhere.resolve()}\n" in agent.state.system_prompt
    assert "You are powered by the model claude-opus-5." in agent.state.system_prompt


def test_project_instructions_are_read_from_cwd_not_the_process_directory(
    tmp_path: Path,
) -> None:
    """`lab-harness --cwd other/` must pick up *that* project's conventions."""
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()
    (elsewhere / "HARNESS.md").write_text("Conventions of the other project.")

    agent = _build_agent(_args(cwd=str(elsewhere)), Config())

    assert "Conventions of the other project." in agent.state.system_prompt


def test_the_configured_filename_is_honored(tmp_path: Path) -> None:
    (tmp_path / "CONVENTIONS.md").write_text("house style")
    config = Config(project=ProjectConfig(context_files=("CONVENTIONS.md",)))

    agent = _build_agent(_args(), config)

    assert "house style" in agent.state.system_prompt


# --- the system prompt flags ------------------------------------------------


def test_a_configured_prompt_is_the_one_sent() -> None:
    config = Config(prompts=PromptsConfig(system="Answer in French."))

    assert _build_agent(_args(), config).state.system_prompt == "Answer in French."


def test_the_flag_replaces_the_configured_prompt(tmp_path: Path) -> None:
    """Only `system`: the other sections, and the project's instructions, stay."""
    (tmp_path / "AGENTS.md").write_text("Indent with tabs.")

    agent = _build_agent(_args(system_prompt="Be terse."), Config())

    assert agent.state.system_prompt.startswith(f"Be terse.\n\n{Config().prompts.harness}\n\n")
    assert agent.state.system_prompt.endswith("Indent with tabs.")
    assert Config().prompts.system not in agent.state.system_prompt


def test_the_file_flag_reads_the_prompt_from_a_file(tmp_path: Path) -> None:
    prompt_file = tmp_path / "terse.md"
    prompt_file.write_text("Be terse.\nAnswer in one line.\n")

    agent = _build_agent(_args(system_prompt_file=str(prompt_file)), Config())

    # Stripped, as project instructions are: the editor's final newline goes.
    assert agent.state.system_prompt.startswith("Be terse.\nAnswer in one line.\n\n# ")


def test_an_empty_flag_drops_only_the_system_section() -> None:
    agent = _build_agent(_args(system_prompt=""), Config())

    assert agent.state.system_prompt.startswith(Config().prompts.harness)


def test_the_two_prompt_flags_exclude_each_other(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--system-prompt", "a", "--system-prompt-file", "b"])
    assert "not allowed with" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("content", "expected"),
    [(None, "cannot read"), (b"\xff\xfe", "not UTF-8 text")],
    ids=["missing", "binary"],
)
def test_an_unusable_prompt_file_is_reported_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: bytes | None, expected: str
) -> None:
    prompt_file = tmp_path / "prompt.md"
    if content is not None:
        prompt_file.write_bytes(content)
    argv = ["-p", "hi", "--system-prompt-file", str(prompt_file), "--cwd", str(tmp_path)]

    assert main([*argv, "--no-save"]) == 1
    err = capsys.readouterr().err
    assert "error: --system-prompt-file: " in err
    assert expected in err


@pytest.mark.parametrize(
    ("args", "source"),
    [
        (_args(), "prompts.system"),
        (_args(system_prompt="x"), "--system-prompt"),
    ],
    ids=["config", "flag"],
)
def test_the_prompt_source_is_logged(
    caplog: pytest.LogCaptureFixture, args: argparse.Namespace, source: str
) -> None:
    with caplog.at_level(logging.INFO, logger="lab_harness.cli.main"):
        _build_agent(args, Config())

    assert any(m.startswith(f"System prompt from {source} (") for m in caplog.messages)


# --- interactive wiring ---------------------------------------------------


async def test_interactive_attaches_an_approval_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Agent] = {}

    class _FakeTerminal(_StubTerminal):
        def __init__(self, console: object, config: Config) -> None:
            super().__init__(captured)
            self.config = config

    monkeypatch.setattr("lab_harness.cli.main.Terminal", _FakeTerminal)

    code = await run_interactive(_args(), Config())

    assert code == 0
    agent = captured["agent"]
    assert agent._before_tool_call is not None, "interactive runs must gate tools"


async def test_yes_disables_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Agent] = {}

    class _FakeTerminal(_StubTerminal):
        def __init__(self, console: object, config: Config) -> None:
            super().__init__(captured)

    monkeypatch.setattr("lab_harness.cli.main.Terminal", _FakeTerminal)

    await run_interactive(_args(yes=True), Config())

    assert captured["agent"]._before_tool_call is None


async def test_the_gate_guards_the_configured_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Agent] = {}

    class _FakeTerminal(_StubTerminal):
        def __init__(self, console: object, config: Config) -> None:
            super().__init__(captured)

    monkeypatch.setattr("lab_harness.cli.main.Terminal", _FakeTerminal)
    config = Config(tools=ToolsConfig(approval=ApprovalConfig(required=("bash",))))

    await run_interactive(_args(), config)

    gate = captured["agent"]._before_tool_call
    assert isinstance(gate, ApprovalGate)
    assert gate.requires_approval("bash")
    assert not gate.requires_approval("read")


def test_interactive_resume_shows_the_restored_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--resume` replays the saved conversation, so the session continues
    where the eye left off rather than from a one-line note."""
    provider = FakeProvider([text_response("first answer")])
    monkeypatch.setattr("lab_harness.cli.main.route", lambda *_: (FAKE_MODEL, provider))
    assert main(["-p", "first", "--cwd", str(tmp_path)]) == 0
    [path] = list_sessions(tmp_path, Config().session)

    captured: dict[str, Agent] = {}
    replayed: list[tuple] = []
    terminal: dict[str, object] = {}

    class _RecordingRenderer:
        def replay(self, messages: object) -> None:
            replayed.append(tuple(messages))  # type: ignore[arg-type]

    class _FakeTerminal(_StubTerminal):
        renderer = _RecordingRenderer()

        def __init__(self, console: object, config: Config) -> None:
            super().__init__(captured)
            terminal["constructed"] = True

    monkeypatch.setattr("lab_harness.cli.main.Terminal", _FakeTerminal)

    assert asyncio.run(run_interactive(_args(resume=path.stem, cwd=str(tmp_path)), Config())) == 0

    assert terminal == {"constructed": True}
    assert len(replayed) == 1, "the restored history is shown once"
    assert [m.role for m in replayed[0]] == ["user", "assistant"]
    assert "first answer" in "".join(
        b.text for m in replayed[0] for b in getattr(m, "content", ()) if hasattr(b, "text")
    )


# --- text sink helpers ----------------------------------------------------


def test_brief_renders_arguments_on_one_line() -> None:
    assert _brief({"path": "a.py"}) == "path=a.py"
    assert _brief({}) == ""


def test_brief_collapses_newlines_and_truncates() -> None:
    result = _brief({"command": "line one\nline two\n" + "x" * 100})

    assert "\n" not in result
    assert len(result) < 60


# --- logging ---------------------------------------------------------------


def test_an_unusable_log_level_is_rejected_at_parse_time() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--log-level", "verbose"])


def test_debug_covers_the_loading_of_config_itself(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of configuring logging before `load_config` runs.

    A level set in a config file cannot report that same file being read, so
    the flag and the env var have to take effect first.
    """
    project = tmp_path / ".lab-harness"
    project.mkdir()
    (project / "config.toml").write_text("[agent]\nmax_turns = 9\n")
    monkeypatch.setenv("LAB_HARNESS_LOG_LEVEL", "debug")

    main(["--resume", "--cwd", str(tmp_path)])

    assert str(project / "config.toml") in capsys.readouterr().err


def test_a_config_file_can_set_the_level(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """What the second configure_logging call buys."""
    project = tmp_path / ".lab-harness"
    project.mkdir()
    (project / "config.toml").write_text('[logging]\nlevel = "debug"\n')

    main(["--resume", "--cwd", str(tmp_path)])
    resolve_model("claude-sonnet-5")

    assert "Using model" in capsys.readouterr().err


def test_the_flag_beats_a_config_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    project = tmp_path / ".lab-harness"
    project.mkdir()
    (project / "config.toml").write_text('[logging]\nlevel = "debug"\n')

    main(["--resume", "--cwd", str(tmp_path), "--log-level", "error"])
    resolve_model("claude-sonnet-5")

    assert "Using model" not in capsys.readouterr().err


def test_logs_go_to_stderr_not_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """stdout carries the answer and the JSONL stream; a log line would corrupt both."""
    monkeypatch.setenv("LAB_HARNESS_LOG_LEVEL", "debug")

    main(["--resume", "--cwd", str(tmp_path)])

    captured = capsys.readouterr()
    assert "DEBUG" in captured.err
    assert "DEBUG" not in captured.out


def test_an_unusable_level_in_a_config_file_is_a_startup_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Same contract as any other bad config value: fail, do not guess."""
    project = tmp_path / ".lab-harness"
    project.mkdir()
    (project / "config.toml").write_text('[logging]\nlevel = "loud"\n')

    assert main(["--resume", "--cwd", str(tmp_path)]) == 1
    assert "loud" in capsys.readouterr().err


# --- provider routing ------------------------------------------------------


def test_a_vendor_added_in_a_config_file_reaches_the_agent(tmp_path: Path) -> None:
    """A vendor and its model, added purely in config, end to end through the wiring."""
    directory = tmp_path / PROJECT_DIRECTORY
    directory.mkdir()
    (directory / CONFIG_FILE).write_text(
        '[providers.openai.groq]\nbase_url = "https://api.groq.com/openai/v1"\n\n'
        '[models."llama-3.3-70b"]\nprovider = "groq"\n'
    )

    agent = _build_agent(_args(model="llama-3.3-70b"), load_config(tmp_path))

    assert isinstance(agent._provider, ChatCompletionsProvider)
    assert agent._provider.name == "groq"
    # The resolved model must claim the provider that actually serves it.
    assert agent.state.model.provider == "groq"


def test_an_unknown_provider_is_a_startup_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["-p", "hi", "--provider", "nope", "--cwd", str(tmp_path), "--no-save"]) == 1
    assert "nope" in capsys.readouterr().err


# --- sessions ---------------------------------------------------------------


def test_a_saved_session_rebuilds_every_request_the_cli_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wiring half of the trajectory guarantee: what `main` records is
    enough to rebuild each request — project instructions, tools and all —
    across a `--resume`."""
    provider = FakeProvider([text_response("one"), text_response("two")])
    monkeypatch.setattr("lab_harness.cli.main.route", lambda *_: (FAKE_MODEL, provider))
    (tmp_path / "AGENTS.md").write_text("Indent with tabs.")

    assert main(["-p", "first", "--cwd", str(tmp_path)]) == 0
    [path] = list_sessions(tmp_path, Config().session)
    assert main(["-p", "second", "--resume", path.stem, "--cwd", str(tmp_path)]) == 0

    rebuilt = recorded_requests(read_transcript(path))
    assert [r.request for r in rebuilt] == provider.requests
    assert "Indent with tabs." in rebuilt[0].request.system_prompt
    assert {t.name for t in rebuilt[0].request.tools} == {t.name for t in default_registry()}
    assert rebuilt[0].config == snapshot(Config())


def test_resume_warns_when_the_project_instructions_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editing AGENTS.md changes the system prompt, which resume should say."""
    provider = FakeProvider([text_response("one")])
    monkeypatch.setattr("lab_harness.cli.main.route", lambda *_: (FAKE_MODEL, provider))
    (tmp_path / "AGENTS.md").write_text("Indent with tabs.")
    assert main(["-p", "first", "--cwd", str(tmp_path)]) == 0
    (tmp_path / "AGENTS.md").write_text("Indent with spaces.")

    [path] = list_sessions(tmp_path, Config().session)
    args = _args(cwd=str(tmp_path), resume=path.stem)
    output = StringIO()
    _attach_session(_build_agent(args, Config()), args, Config(), Console(file=output))

    assert "system prompt has changed" in output.getvalue()


def test_one_shot_resume_warnings_go_to_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`-p` has no terminal to print them in, and must not mix them into the answer."""
    provider = FakeProvider([text_response("one"), text_response("two")])
    monkeypatch.setattr("lab_harness.cli.main.route", lambda *_: (FAKE_MODEL, provider))
    (tmp_path / "AGENTS.md").write_text("Indent with tabs.")
    assert main(["-p", "first", "--cwd", str(tmp_path)]) == 0
    (tmp_path / "AGENTS.md").write_text("Indent with spaces.")
    capsys.readouterr()

    [path] = list_sessions(tmp_path, Config().session)
    assert main(["-p", "second", "--resume", path.stem, "--cwd", str(tmp_path)]) == 0

    captured = capsys.readouterr()
    assert "system prompt has changed" in captured.err
    assert captured.out == "two\n"


def test_resume_without_an_id_lists_the_sessions_and_runs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Continuing a conversation takes its id, so the bare flag shows which ids exist."""
    provider = FakeProvider([text_response("one"), text_response("two")])
    monkeypatch.setattr("lab_harness.cli.main.route", lambda *_: (FAKE_MODEL, provider))
    assert main(["-p", "first", "--cwd", str(tmp_path)]) == 0
    assert main(["-p", "second", "--cwd", str(tmp_path)]) == 0
    capsys.readouterr()

    assert main(["-p", "third", "--resume", "--cwd", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    newest, oldest = list_sessions(tmp_path, Config().session)
    assert out.index(newest.stem) < out.index(oldest.stem)
    assert "--resume SESSION_ID" in out
    assert len(provider.requests) == 2, "the prompt was not sent"
    assert len(list_sessions(tmp_path, Config().session)) == 2, "and nothing was recorded"


def test_resume_without_an_id_and_no_sessions_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--resume", "--cwd", str(tmp_path)]) == 0
    assert "No saved sessions" in capsys.readouterr().out


# --- where log records go -------------------------------------------------


class _Tty(StringIO):
    def isatty(self) -> bool:
        return True


def _streams() -> list[object]:
    handlers = logging.getLogger(ROOT_LOGGER).handlers
    return [h.stream for h in handlers if isinstance(h, logging.StreamHandler)]


async def _interactive(monkeypatch: pytest.MonkeyPatch, *, stderr: StringIO) -> None:
    class _FakeTerminal(_StubTerminal):
        def __init__(self, console: object, config: Config) -> None:
            super().__init__({})

    monkeypatch.setattr("lab_harness.cli.main.Terminal", _FakeTerminal)
    monkeypatch.setattr("lab_harness.cli.main.Console", lambda: Console(file=_Tty()))
    monkeypatch.setattr(sys, "stderr", stderr)
    await run_interactive(_args(log_level=None), Config())


async def test_records_go_through_the_console_when_stderr_is_its_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Written to the terminal directly, they would break the live region."""
    await _interactive(monkeypatch, stderr=_Tty())

    assert [type(s) for s in _streams()] == [LogStream]


async def test_a_redirected_stderr_keeps_the_records(monkeypatch: pytest.MonkeyPatch) -> None:
    """`lab-harness 2> debug.log` is how to keep a trace out of the conversation."""
    await _interactive(monkeypatch, stderr=StringIO())

    assert not any(isinstance(s, LogStream) for s in _streams())


# --- export -----------------------------------------------------------------


def test_export_writes_the_latest_session_as_a_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = FakeProvider([text_response("the answer")])
    monkeypatch.setattr("lab_harness.cli.main.route", lambda *_: (FAKE_MODEL, provider))
    assert main(["-p", "the question", "--cwd", str(tmp_path)]) == 0
    [path] = list_sessions(tmp_path, Config().session)
    capsys.readouterr()

    assert main(["--export", "--cwd", str(tmp_path)]) == 0

    page = path.with_suffix(".html")
    assert capsys.readouterr().out.strip() == str(page)
    assert "the question" in page.read_text(encoding="utf-8")
    assert "the answer" in page.read_text(encoding="utf-8")


def test_export_of_an_unknown_session_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--export", "nope", "--cwd", str(tmp_path)]) == 1
    assert "nope" in capsys.readouterr().err
