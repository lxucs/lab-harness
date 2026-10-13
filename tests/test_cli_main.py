"""CLI wiring: argument parsing, mode selection, and what gets attached.

The agent is built with a real `AnthropicProvider`, which never connects here
because nothing calls `stream()` — construction alone makes no request.
"""

import argparse
from pathlib import Path

import pytest

from harness.agent import Agent
from harness.cli.approval import ApprovalGate
from harness.cli.main import (
    SYSTEM_PROMPT,
    _brief,
    _build_agent,
    build_parser,
    build_system_prompt,
    main,
    run_interactive,
)
from harness.config import (
    AgentConfig,
    ApprovalConfig,
    Config,
    OpenAIConfig,
    ProjectConfig,
    ProvidersConfig,
    TerminalConfig,
    ToolsConfig,
)
from harness.llm.providers.anthropic import AnthropicProvider
from harness.llm.providers.chat_completions import ChatCompletionsProvider
from harness.llm.registry import resolve_model
from harness.project import ProjectContext


@pytest.fixture(autouse=True)
def _neutral_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Run every CLI test in an empty directory.

    `_args()` leaves `cwd` at ".", and `_build_agent` reads project
    instructions from there. Without this the suite would pick up the
    repository's own AGENTS.md and quietly couple these assertions to it.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _args(**kwargs: object) -> argparse.Namespace:
    defaults = {
        "prompt": None,
        "json": False,
        "model": None,
        "provider": None,
        "cwd": ".",
        "yes": False,
        "resume": None,
        "sessions": False,
        # Tests must never write to the real ~/.harness.
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


def test_prompt_is_optional_now_that_interactive_exists() -> None:
    """Bare `harness` must parse — it starts the terminal."""
    args = build_parser().parse_args([])
    assert args.prompt is None


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
    assert args.log_level is None


def test_yes_flag_defaults_off() -> None:
    """Approval must be opt-out, never opt-in."""
    assert build_parser().parse_args([]).yes is False
    assert build_parser().parse_args(["--yes"]).yes is True


# --- agent construction ---------------------------------------------------


def test_one_shot_gets_every_builtin_tool() -> None:
    agent = _build_agent(_args(prompt="hi"), Config())

    names = {t.name for t in agent._tools}
    assert {"read", "write", "edit", "ls", "bash", "grep", "glob"} <= names


def test_interactive_narrows_to_the_configured_tool_set() -> None:
    config = Config(terminal=TerminalConfig(enabled_tools=("read", "ls")))

    agent = _build_agent(_args(), config, tool_names=config.terminal.enabled_tools)

    assert [t.name for t in agent._tools] == ["read", "ls"]


def test_an_unknown_configured_tool_fails_loudly() -> None:
    """A typo should fail at wiring, not silently disable a tool."""
    config = Config(terminal=TerminalConfig(enabled_tools=("read", "typo")))

    with pytest.raises(ValueError, match="Unknown tool"):
        _build_agent(_args(), config, tool_names=config.terminal.enabled_tools)


def test_one_shot_has_no_approval_gate() -> None:
    """There is nobody at the keyboard to answer a prompt."""
    agent = _build_agent(_args(prompt="hi"), Config())

    assert agent._before_tool_call is None


def test_cwd_reaches_the_agent() -> None:
    agent = _build_agent(_args(cwd="/tmp/somewhere"), Config())

    assert agent._cwd == "/tmp/somewhere"


def test_the_model_flag_reaches_the_agent() -> None:
    agent = _build_agent(_args(model="claude-opus-5"), Config())

    assert agent.state.model.id == "claude-opus-5"


def test_the_configured_model_reaches_the_agent() -> None:
    """Without a flag, `agent.model` decides — not a constant in the code."""
    config = Config(agent=AgentConfig(model="claude-haiku-4-5-20251001"))

    agent = _build_agent(_args(), config)

    assert agent.state.model.id == "claude-haiku-4-5-20251001"


def test_the_model_flag_beats_the_configured_model() -> None:
    config = Config(agent=AgentConfig(model="claude-haiku-4-5-20251001"))

    agent = _build_agent(_args(model="claude-opus-5"), config)

    assert agent.state.model.id == "claude-opus-5"


# --- project instructions -------------------------------------------------


def test_no_project_file_means_the_base_prompt() -> None:
    agent = _build_agent(_args(), Config())

    assert agent.state.system_prompt == SYSTEM_PROMPT


def test_project_instructions_reach_the_system_prompt(_neutral_cwd: Path) -> None:
    (_neutral_cwd / "AGENTS.md").write_text("Always run `uv run pytest`.")

    agent = _build_agent(_args(), Config())

    assert "Always run `uv run pytest`." in agent.state.system_prompt
    assert agent.state.system_prompt.startswith(SYSTEM_PROMPT)


def test_project_instructions_are_read_from_cwd_not_the_process_directory(
    tmp_path: Path,
) -> None:
    """`harness --cwd other/` must pick up *that* project's conventions."""
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()
    (elsewhere / "HARNESS.md").write_text("Conventions of the other project.")

    agent = _build_agent(_args(cwd=str(elsewhere)), Config())

    assert "Conventions of the other project." in agent.state.system_prompt


def test_the_configured_filename_is_honored(_neutral_cwd: Path) -> None:
    (_neutral_cwd / "CONVENTIONS.md").write_text("house style")
    config = Config(project=ProjectConfig(context_files=("CONVENTIONS.md",)))

    agent = _build_agent(_args(), config)

    assert "house style" in agent.state.system_prompt


def test_the_prompt_names_the_file_it_came_from() -> None:
    """So the model can edit the source of a rule rather than just obey it."""
    context = ProjectContext(path=Path("/repo/AGENTS.md"), text="Use tabs.")

    prompt = build_system_prompt(context)

    assert "AGENTS.md" in prompt
    assert "Use tabs." in prompt


def test_the_prompt_does_not_leak_the_absolute_path() -> None:
    """A transcript may be shared; the machine's layout should not ride along."""
    context = ProjectContext(path=Path("/home/someone/secret-project/AGENTS.md"), text="x")

    prompt = build_system_prompt(context)

    assert "/home/someone" not in prompt


def test_no_context_leaves_the_prompt_untouched() -> None:
    assert build_system_prompt(None) == SYSTEM_PROMPT


# --- interactive wiring ---------------------------------------------------


async def test_interactive_attaches_an_approval_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Agent] = {}

    class _FakeTerminal(_StubTerminal):
        def __init__(self, console: object, config: Config) -> None:
            super().__init__(captured)
            self.config = config

    monkeypatch.setattr("harness.cli.main.Terminal", _FakeTerminal)

    code = await run_interactive(_args(), Config())

    assert code == 0
    agent = captured["agent"]
    assert agent._before_tool_call is not None, "interactive runs must gate tools"


async def test_yes_disables_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Agent] = {}

    class _FakeTerminal(_StubTerminal):
        def __init__(self, console: object, config: Config) -> None:
            super().__init__(captured)

    monkeypatch.setattr("harness.cli.main.Terminal", _FakeTerminal)

    await run_interactive(_args(yes=True), Config())

    assert captured["agent"]._before_tool_call is None


async def test_the_gate_guards_the_configured_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Agent] = {}

    class _FakeTerminal(_StubTerminal):
        def __init__(self, console: object, config: Config) -> None:
            super().__init__(captured)

    monkeypatch.setattr("harness.cli.main.Terminal", _FakeTerminal)
    config = Config(tools=ToolsConfig(approval=ApprovalConfig(required=("bash",))))

    await run_interactive(_args(), config)

    gate = captured["agent"]._before_tool_call
    assert isinstance(gate, ApprovalGate)
    assert gate.requires_approval("bash")
    assert not gate.requires_approval("read")


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
    project = tmp_path / ".harness"
    project.mkdir()
    (project / "config.toml").write_text("[agent]\nmax_turns = 9\n")
    monkeypatch.setenv("HARNESS_LOG_LEVEL", "debug")

    main(["--sessions", "--cwd", str(tmp_path)])

    assert str(project / "config.toml") in capsys.readouterr().err


def test_a_config_file_can_set_the_level(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """What the second configure_logging call buys."""
    project = tmp_path / ".harness"
    project.mkdir()
    (project / "config.toml").write_text('[logging]\nlevel = "debug"\n')

    main(["--sessions", "--cwd", str(tmp_path)])
    resolve_model("claude-sonnet-5")

    assert "Using model" in capsys.readouterr().err


def test_the_flag_beats_a_config_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    project = tmp_path / ".harness"
    project.mkdir()
    (project / "config.toml").write_text('[logging]\nlevel = "debug"\n')

    main(["--sessions", "--cwd", str(tmp_path), "--log-level", "error"])
    resolve_model("claude-sonnet-5")

    assert "Using model" not in capsys.readouterr().err


def test_logs_go_to_stderr_not_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """stdout carries the answer and the JSONL stream; a log line would corrupt both."""
    monkeypatch.setenv("HARNESS_LOG_LEVEL", "debug")

    main(["--sessions", "--cwd", str(tmp_path)])

    captured = capsys.readouterr()
    assert "DEBUG" in captured.err
    assert "DEBUG" not in captured.out


def test_an_unusable_level_in_a_config_file_is_a_startup_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Same contract as any other bad config value: fail, do not guess."""
    project = tmp_path / ".harness"
    project.mkdir()
    (project / "config.toml").write_text('[logging]\nlevel = "loud"\n')

    assert main(["--sessions", "--cwd", str(tmp_path)]) == 1
    assert "loud" in capsys.readouterr().err


# --- provider routing ------------------------------------------------------


def test_the_default_build_uses_anthropic() -> None:
    agent = _build_agent(_args(), Config())

    assert isinstance(agent._provider, AnthropicProvider)


def test_a_deepseek_model_id_routes_without_a_flag() -> None:
    """The ergonomics the profiles exist for."""
    agent = _build_agent(_args(model="deepseek-flash"), Config())

    assert isinstance(agent._provider, ChatCompletionsProvider)
    assert agent._provider.name == "deepseek"
    assert agent.state.model.id == "deepseek-flash"


def test_the_provider_flag_overrides_the_model_id() -> None:
    agent = _build_agent(_args(model="deepseek-flash", provider="anthropic"), Config())

    assert isinstance(agent._provider, AnthropicProvider)


def test_a_config_only_profile_reaches_the_agent() -> None:
    """A vendor added purely in config, end to end through the wiring."""
    config = Config(
        providers=ProvidersConfig(
            openai={
                "groq": OpenAIConfig(
                    base_url="https://api.groq.com/openai/v1", models=("llama-3.3-70b",)
                )
            }
        )
    )

    agent = _build_agent(_args(model="llama-3.3-70b"), config)

    assert isinstance(agent._provider, ChatCompletionsProvider)
    assert agent._provider.name == "groq"
    # The resolved model must claim the provider that actually serves it.
    assert agent.state.model.provider == "groq"


def test_an_unknown_provider_is_a_startup_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["-p", "hi", "--provider", "nope", "--cwd", str(tmp_path), "--no-save"]) == 1
    assert "nope" in capsys.readouterr().err
