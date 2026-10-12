"""CLI wiring: argument parsing, mode selection, and what gets attached.

The agent is built with a real `AnthropicProvider`, which never connects here
because nothing calls `stream()` — construction alone makes no request.
"""

import argparse

import pytest

from harness.agent import Agent
from harness.cli.approval import ApprovalGate
from harness.cli.main import _brief, _build_agent, build_parser, run_interactive
from harness.config import ApprovalConfig, Config, TerminalConfig, ToolsConfig


def _args(**kwargs: object) -> argparse.Namespace:
    defaults = {
        "prompt": None,
        "json": False,
        "model": None,
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
    args = build_parser().parse_args(["-p", "hi", "--json", "--model", "m", "--cwd", "/tmp"])

    assert args.prompt == "hi"
    assert args.json is True
    assert args.model == "m"
    assert args.cwd == "/tmp"


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
