"""The prompts: shipped as data, overridden by config, assembled into the system prompt.

The text itself lives in lab_harness/prompts/prompts.toml. What is pinned here is the
contract around it: it loads as strictly as config, a config layer replaces it
key by key, it names no tools, the sections go out in a fixed order with the
environment filled in, and a template a user writes fails at startup rather
than in the one directory where it is first filled.
"""

import platform
import re
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path

import pytest

import lab_harness.config as config_module
import lab_harness.prompts.environment as environment_module
from lab_harness.config import (
    CONFIG_FILE,
    PROJECT_DIRECTORY,
    Config,
    ConfigError,
    PromptsConfig,
    _shipped_prompts,
    load_config,
)
from lab_harness.prompts import (
    Environment,
    ProjectContext,
    build_system_prompt,
    detect_environment,
)
from lab_harness.tools import BUILTIN_TOOLS

INSTRUCTIONS = ProjectContext(path=Path("/repo/AGENTS.md"), text="Use tabs.")
ENVIRONMENT = Environment(
    cwd="/path/to/project",
    git_enabled=True,
    system_platform="linux",
    shell="/bin/sh",
    os_version="Linux 6.1.0",
    agent_model="claude-sonnet-5",
)
SECTIONS = [f.name for f in fields(PromptsConfig)]


# --- the shipped file -----------------------------------------------------


def test_the_shipped_prompts_load() -> None:
    prompts = Config().prompts

    assert all(getattr(prompts, name) for name in SECTIONS)
    assert "{text}" in prompts.project_instructions


@pytest.mark.parametrize("section", SECTIONS)
def test_the_shipped_prompts_name_no_tool(section: str) -> None:
    """What the model can call comes from the declarations alone, so the prompt
    stays true whatever `--tools` selects."""
    text = getattr(Config().prompts, section)

    named = [t.name for t in BUILTIN_TOOLS if re.search(rf"\b{re.escape(t.name)}\b", text)]
    assert not named, f"prompts.{section} names tools: {named}"


def test_the_shipped_environment_says_where_the_agent_runs() -> None:
    prompt = build_system_prompt(Config().prompts, ENVIRONMENT)

    assert (
        "# Environment\n"
        "You have been invoked in the following environment:\n"
        " - Primary working directory: /path/to/project\n"
        " - Is a git repository: true\n"
        " - Platform: linux\n"
        " - Shell: /bin/sh\n"
        " - OS Version: Linux 6.1.0\n"
        " - You are powered by the model claude-sonnet-5."
    ) in prompt


def test_the_shipped_wrapper() -> None:
    assert build_system_prompt(
        PromptsConfig(system="S", project_instructions=Config().prompts.project_instructions),
        ENVIRONMENT,
        INSTRUCTIONS,
    ) == (
        "S\n\n# Project instructions (AGENTS.md)\n\n"
        "These come from the working directory and describe this project's own "
        "conventions. Follow them.\n\nUse tabs."
    )


def test_the_prompts_file_may_only_hold_prompts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """It is read like a config file, so a stray key fails rather than vanishing."""
    stray = tmp_path / "prompts.toml"
    stray.write_text('[prompts]\nsystem = "x"\nsystem_prompt = "typo"\n')
    monkeypatch.setattr(config_module, "PROMPTS_FILE", stray)

    with pytest.raises(
        ConfigError, match=r"prompts\.toml: unknown setting 'prompts\.system_prompt'"
    ):
        _shipped_prompts.__wrapped__()


def test_a_missing_prompts_file_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config_module, "PROMPTS_FILE", tmp_path / "prompts.toml")

    with pytest.raises(ConfigError, match="package is incomplete"):
        _shipped_prompts.__wrapped__()


# --- overriding from config -----------------------------------------------


def write_project_config(tmp_path: Path, body: str) -> None:
    path = tmp_path / PROJECT_DIRECTORY / CONFIG_FILE
    path.parent.mkdir(parents=True)
    path.write_text(body)


def test_a_config_file_replaces_one_prompt_and_keeps_the_others(tmp_path: Path) -> None:
    write_project_config(tmp_path, '[prompts]\nsystem = """\nAnswer in French."""\n')

    prompts = load_config(tmp_path).prompts

    assert prompts.system == "Answer in French."
    assert replace(prompts, system=Config().prompts.system) == Config().prompts


def test_a_section_set_empty_in_config_is_not_sent(tmp_path: Path) -> None:
    write_project_config(tmp_path, '[prompts]\nharness = ""\n')

    prompt = build_system_prompt(load_config(tmp_path).prompts, ENVIRONMENT)

    assert "# Harness" not in prompt
    assert f"{Config().prompts.system}\n\n# Environment\n" in prompt


def test_a_typo_in_the_prompts_section_is_an_error(tmp_path: Path) -> None:
    write_project_config(tmp_path, '[prompts]\nsytem = "x"\n')

    with pytest.raises(ConfigError, match=r"unknown setting 'prompts\.sytem'"):
        load_config(tmp_path)


# --- assembly -------------------------------------------------------------


def test_the_sections_are_sent_in_order() -> None:
    prompts = PromptsConfig(
        system="S",
        harness="H",
        environment="E",
        context_management="C",
        project_instructions="P {text}",
    )

    assert build_system_prompt(prompts, ENVIRONMENT, INSTRUCTIONS) == (
        "S\n\nH\n\nE\n\nC\n\nP Use tabs."
    )


def test_an_empty_section_is_left_out() -> None:
    prompts = PromptsConfig(system="S", harness="", environment="", context_management="C")

    assert build_system_prompt(prompts, ENVIRONMENT) == "S\n\nC"


def test_no_instructions_leaves_the_prompt_alone() -> None:
    assert build_system_prompt(PromptsConfig(system="S"), ENVIRONMENT) == "S"


def test_every_environment_placeholder_is_filled() -> None:
    template = "{cwd} {git_enabled} {system_platform} {shell} {os_version} {agent_model}"

    prompt = build_system_prompt(PromptsConfig(environment=template), ENVIRONMENT)

    assert prompt == "/path/to/project true linux /bin/sh Linux 6.1.0 claude-sonnet-5"


def test_outside_a_repository_git_reads_false() -> None:
    outside = replace(ENVIRONMENT, git_enabled=False)

    prompt = build_system_prompt(PromptsConfig(environment="{git_enabled}"), outside)

    assert prompt == "false"


def test_only_the_templates_are_filled() -> None:
    """Braces elsewhere are text, so a prompt passed with `--system-prompt` that
    holds JSON, or happens to say {cwd}, is sent as written."""
    prompts = PromptsConfig(system='Reply as {"a": 1}.', harness="Mind {cwd}.")

    assert build_system_prompt(prompts, ENVIRONMENT) == 'Reply as {"a": 1}.\n\nMind {cwd}.'


def test_the_prompt_names_the_file_it_came_from() -> None:
    """So the model can edit the source of a rule rather than just obey it."""
    prompt = build_system_prompt(Config().prompts, ENVIRONMENT, INSTRUCTIONS)

    assert prompt.startswith(Config().prompts.system + "\n\n")
    assert "AGENTS.md" in prompt
    assert prompt.endswith("Use tabs.")


def test_the_instructions_do_not_carry_their_absolute_path() -> None:
    """Where the project is, is `environment`'s to say: a config that sets it
    to "" keeps this machine's layout out of every request."""
    context = ProjectContext(path=Path("/home/someone/secret-project/AGENTS.md"), text="x")
    prompts = replace(Config().prompts, environment="")

    assert "/home/someone" not in build_system_prompt(prompts, ENVIRONMENT, context)


def test_with_no_system_prompt_the_instructions_stand_alone() -> None:
    prompts = PromptsConfig(system="", project_instructions="From {name}: {text}")

    assert build_system_prompt(prompts, ENVIRONMENT, INSTRUCTIONS) == "From AGENTS.md: Use tabs."


def test_braces_in_the_instructions_are_left_alone() -> None:
    """Only the template is parsed; a file full of JSON must not be."""
    context = ProjectContext(path=Path("AGENTS.md"), text='Config is {"a": 1}.')

    assert build_system_prompt(PromptsConfig(), ENVIRONMENT, context) == 'Config is {"a": 1}.'


@pytest.mark.parametrize(
    ("template", "message"),
    [
        ("{text} from {file}", r"unknown placeholder\(s\) \{file\}"),
        ("{text} {}", r"unknown placeholder\(s\) \{\}"),
        ("{text.upper}", r"unknown placeholder\(s\) \{text\.upper\}"),
        ("{text} in {cwd}", r"unknown placeholder\(s\) \{cwd\}"),
        ("Instructions from {name}", r"needs \{text\}"),
        ("{text", "Write {{ and }} for literal braces"),
        ("{text:d}", "Write {{ and }} for literal braces"),
    ],
    ids=["unknown", "positional", "attribute", "environment's", "no-text", "unclosed", "bad-spec"],
)
def test_a_broken_template_fails_even_with_nothing_to_fill(template: str, message: str) -> None:
    """It fails in every directory, not only in the first one with an AGENTS.md."""
    with pytest.raises(ConfigError, match=rf"^prompts\.project_instructions: .*{message}"):
        build_system_prompt(PromptsConfig(project_instructions=template), ENVIRONMENT)


@pytest.mark.parametrize(
    ("template", "message"),
    [
        ("Running in {directory}", r"unknown placeholder\(s\) \{directory\}; .*\{cwd\}"),
        ("From {name}", r"unknown placeholder\(s\) \{name\}"),
        ("{cwd", "Write {{ and }} for literal braces"),
        ("{git_enabled:d}", "Write {{ and }} for literal braces"),
    ],
    ids=["unknown", "instructions'", "unclosed", "bad-spec"],
)
def test_a_broken_environment_template_fails(template: str, message: str) -> None:
    with pytest.raises(ConfigError, match=rf"^prompts\.environment: .*{message}"):
        build_system_prompt(PromptsConfig(environment=template), ENVIRONMENT)


def test_literal_braces_are_written_doubled() -> None:
    prompts = PromptsConfig(
        system="", environment="{{cwd}} is {cwd}", project_instructions="{{rules}}\n{text}"
    )

    assert build_system_prompt(prompts, ENVIRONMENT, INSTRUCTIONS) == (
        "{cwd} is /path/to/project\n\n{rules}\nUse tabs."
    )


# --- the environment ------------------------------------------------------


def test_the_working_directory_is_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert detect_environment(".", "m").cwd == str(tmp_path.resolve())


def test_the_model_is_the_one_given(tmp_path: Path) -> None:
    assert detect_environment(tmp_path, "claude-opus-5").agent_model == "claude-opus-5"


def test_a_repository_is_found_from_a_subdirectory(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    inside = tmp_path / "src" / "pkg"
    inside.mkdir(parents=True)

    assert detect_environment(inside, "m").git_enabled


def test_a_worktree_counts_as_a_repository(tmp_path: Path) -> None:
    """A linked worktree's `.git` is a file pointing at the main repository."""
    (tmp_path / ".git").write_text("gitdir: /path/to/repo/.git/worktrees/x\n")

    assert detect_environment(tmp_path, "m").git_enabled


def test_a_directory_outside_any_repository(tmp_path: Path) -> None:
    assert not detect_environment(tmp_path, "m").git_enabled


def test_the_platform_is_python_s_own_description(tmp_path: Path) -> None:
    environment = detect_environment(tmp_path, "m")

    assert environment.system_platform == sys.platform
    assert environment.os_version == f"{platform.system()} {platform.release()}"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shells name themselves in $0")
def test_the_shell_is_the_one_commands_run_in(tmp_path: Path) -> None:
    """`shell=True` is what `create_subprocess_shell` uses, and $0 names the shell."""
    ran = subprocess.run("echo $0", shell=True, capture_output=True, text=True, check=True)

    assert detect_environment(tmp_path, "m").shell == ran.stdout.strip()


def test_on_windows_the_shell_is_comspec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(environment_module.sys, "platform", "win32")
    monkeypatch.setenv("COMSPEC", r"C:\Windows\system32\cmd.exe")

    assert environment_module._shell() == r"C:\Windows\system32\cmd.exe"

    monkeypatch.delenv("COMSPEC")
    assert environment_module._shell() == "cmd.exe"
