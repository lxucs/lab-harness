"""The prompts: shipped as data, overridden by config, assembled into the system prompt.

The text itself lives in lab_harness/prompts/prompts.toml. What is pinned here is the
contract around it: it loads as strictly as config, a config layer replaces it
key by key, it names no tools, the sections go out in a fixed order with the
variables filled in, and a section naming one that does not exist fails at
startup rather than in the one directory where it is first filled.
"""

import datetime
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
    date="2026-10-21",
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


def test_the_skills_section_holds_the_list_it_is_given() -> None:
    prompt = build_system_prompt(Config().prompts, ENVIRONMENT, skills="- commit — Write one")

    assert "# Skills" in prompt
    assert "- commit — Write one" in prompt


def test_the_skills_section_is_left_out_when_there_is_no_list() -> None:
    """No skills found, or none loadable: nothing says so in the prompt."""
    prompt = build_system_prompt(Config().prompts, ENVIRONMENT, skills="")

    assert "Skills" not in prompt


def test_the_skills_template_must_hold_the_list() -> None:
    """A template without {skills} would drop the list without saying so."""
    prompts = replace(Config().prompts, skills="# Skills\nNothing to see.")

    with pytest.raises(ConfigError, match=r"prompts\.skills: needs \{skills\}"):
        build_system_prompt(prompts, ENVIRONMENT, skills="- commit — Write one")


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


def test_every_variable_is_filled() -> None:
    template = "{cwd} {git_enabled} {system_platform} {shell} {os_version} {agent_model} {date}"

    prompt = build_system_prompt(PromptsConfig(environment=template), ENVIRONMENT)

    assert prompt == "/path/to/project true linux /bin/sh Linux 6.1.0 claude-sonnet-5 2026-10-21"


def test_a_variable_is_filled_in_any_section() -> None:
    """Every section is a template, so there is no list of the ones that are."""
    prompts = PromptsConfig(system="You are {agent_model}.", harness="In {cwd} on {date}.")

    assert build_system_prompt(prompts, ENVIRONMENT) == (
        "You are claude-sonnet-5.\n\nIn /path/to/project on 2026-10-21."
    )


def test_the_project_variables_are_filled_anywhere() -> None:
    """`{skills}`, `{name}` and `{text}` belong to the sections that wrap them,
    but a variable is a variable wherever it is written."""
    # `project_instructions` still has to hold {text}: what it wraps would
    # otherwise never reach the model.
    prompts = PromptsConfig(
        system="Skills: {skills}.", harness="{name}: {text}", project_instructions="({name}) {text}"
    )

    prompt = build_system_prompt(prompts, ENVIRONMENT, INSTRUCTIONS, skills="- commit — Write one")

    assert prompt == (
        "Skills: - commit — Write one.\n\nAGENTS.md: Use tabs.\n\n(AGENTS.md) Use tabs."
    )


def test_a_brace_that_names_no_variable_is_text() -> None:
    """`{"a": 1}`, `{}` and `{ cwd }` are text, so a section quoting JSON needs
    no escaping — which is why only a brace holding a name is a variable."""
    prompts = PromptsConfig(system='Reply as {"a": 1}.', harness="Use {} or { cwd }.")

    assert build_system_prompt(prompts, ENVIRONMENT) == ('Reply as {"a": 1}.\n\nUse {} or { cwd }.')


def test_outside_a_repository_git_reads_false() -> None:
    outside = replace(ENVIRONMENT, git_enabled=False)

    prompt = build_system_prompt(PromptsConfig(environment="{git_enabled}"), outside)

    assert prompt == "false"


def test_the_instructions_do_not_carry_their_absolute_path() -> None:
    """Where the project is, is `environment`'s to say: a config that sets it
    to "" keeps the machine's layout out of every request."""
    context = ProjectContext(path=Path("/home/someone/secret-project/AGENTS.md"), text="x")
    prompts = replace(Config().prompts, environment="")

    assert "/home/someone" not in build_system_prompt(prompts, ENVIRONMENT, context)


def test_braces_in_the_instructions_are_left_alone() -> None:
    """Only the template is parsed, and in one pass: a file full of JSON, or
    quoting a template of its own, must not be read as one."""
    context = ProjectContext(
        path=Path("AGENTS.md"), text='Config is {"a": 1}, and {{ this }} is doubled.'
    )

    assert build_system_prompt(PromptsConfig(), ENVIRONMENT, context) == (
        'Config is {"a": 1}, and {{ this }} is doubled.'
    )


@pytest.mark.parametrize(
    ("template", "message"),
    [
        ("{text} from {file}", r"unknown placeholder\(s\) \{file\}"),
        ("{text.upper}", r"\{text\.upper\} is not a placeholder"),
        ("{text} in {directory}", r"unknown placeholder\(s\) \{directory\}"),
        ("Instructions from {name}", r"needs \{text\}"),
        ("{text", r"\{text is not a placeholder"),
        ("{text:d}", r"\{text:d\} is not a placeholder"),
    ],
    ids=["unknown", "attribute", "unknown-elsewhere", "no-text", "unclosed", "bad-spec"],
)
def test_a_broken_template_fails_even_with_nothing_to_fill(template: str, message: str) -> None:
    """It fails in every directory, not only in the first one with an AGENTS.md."""
    with pytest.raises(ConfigError, match=rf"^prompts\.project_instructions: .*{message}"):
        build_system_prompt(PromptsConfig(project_instructions=template), ENVIRONMENT)


@pytest.mark.parametrize(
    ("template", "message"),
    [
        ("Running in {directory}", r"unknown placeholder\(s\) \{directory\}; .*\{cwd\}"),
        ("{cwd", r"is not a placeholder"),
    ],
    ids=["unknown", "unclosed"],
)
def test_a_broken_environment_template_fails(template: str, message: str) -> None:
    with pytest.raises(ConfigError, match=rf"^prompts\.environment: .*{message}"):
        build_system_prompt(PromptsConfig(environment=template), ENVIRONMENT)


def test_an_unknown_variable_is_an_error_in_any_section() -> None:
    """A section that names one nobody fills would say nothing where it means
    to say something; a startup error says what it should have said."""
    with pytest.raises(ConfigError, match=r"^prompts\.system: unknown placeholder\(s\) \{model\}"):
        build_system_prompt(PromptsConfig(system="You are {model}."), ENVIRONMENT)


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


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shells name themselves in $0")
def test_the_shell_is_the_one_commands_run_in(tmp_path: Path) -> None:
    """`shell=True` is what `create_subprocess_shell` uses, and $0 names the shell."""
    ran = subprocess.run("echo $0", shell=True, capture_output=True, text=True, check=True)

    assert detect_environment(tmp_path, "m").shell == ran.stdout.strip()


def test_the_date_is_the_day_the_prompt_is_built(tmp_path: Path) -> None:
    assert detect_environment(tmp_path, "m").date == datetime.date.today().isoformat()


def test_on_windows_the_shell_is_comspec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(environment_module.sys, "platform", "win32")
    monkeypatch.setenv("COMSPEC", r"C:\Windows\system32\cmd.exe")

    assert environment_module._shell() == r"C:\Windows\system32\cmd.exe"

    monkeypatch.delenv("COMSPEC")
    assert environment_module._shell() == "cmd.exe"
