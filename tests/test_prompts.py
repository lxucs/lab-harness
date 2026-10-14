"""The prompts: shipped as data, overridden by config, assembled with a project's instructions.

The text itself lives in lab_harness/prompts.toml. What is pinned here is the
contract around it: it loads as strictly as config, a config layer replaces it
key by key, it names no tools, and a template a user writes fails at startup
rather than in the one directory where it is first filled.
"""

import re
from pathlib import Path

import pytest

import lab_harness.config as config_module
from lab_harness.config import (
    CONFIG_FILE,
    PROJECT_DIRECTORY,
    Config,
    ConfigError,
    PromptsConfig,
    _shipped_prompts,
    load_config,
)
from lab_harness.project import ProjectContext, build_system_prompt
from lab_harness.tools import BUILTIN_TOOLS

INSTRUCTIONS = ProjectContext(path=Path("/repo/AGENTS.md"), text="Use tabs.")


# --- the shipped file -----------------------------------------------------


def test_the_shipped_prompts_load() -> None:
    prompts = Config().prompts

    assert prompts.system
    assert "{text}" in prompts.project_instructions


def test_the_shipped_system_prompt_names_no_tool() -> None:
    """What the model can call comes from the declarations alone, so the prompt
    stays true whatever `--tools` selects."""
    system = Config().prompts.system

    named = [t.name for t in BUILTIN_TOOLS if re.search(rf"\b{re.escape(t.name)}\b", system)]
    assert not named, f"The system prompt names tools: {named}"


def test_the_shipped_wrapper_is_unchanged() -> None:
    """Moving it out of Python must not have changed a byte of what is sent."""
    assert build_system_prompt(
        PromptsConfig(system="S", project_instructions=Config().prompts.project_instructions),
        INSTRUCTIONS,
    ) == (
        "S\n\n## Project instructions (AGENTS.md)\n\n"
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


def test_a_config_file_replaces_one_prompt_and_keeps_the_other(tmp_path: Path) -> None:
    write_project_config(tmp_path, '[prompts]\nsystem = """\nAnswer in French."""\n')

    prompts = load_config(tmp_path).prompts

    assert prompts.system == "Answer in French."
    assert prompts.project_instructions == Config().prompts.project_instructions


def test_a_typo_in_the_prompts_section_is_an_error(tmp_path: Path) -> None:
    write_project_config(tmp_path, '[prompts]\nsytem = "x"\n')

    with pytest.raises(ConfigError, match=r"unknown setting 'prompts\.sytem'"):
        load_config(tmp_path)


# --- assembly -------------------------------------------------------------


def test_no_instructions_leaves_the_prompt_alone() -> None:
    assert build_system_prompt(PromptsConfig(system="S"), None) == "S"


def test_the_prompt_names_the_file_it_came_from() -> None:
    """So the model can edit the source of a rule rather than just obey it."""
    prompt = build_system_prompt(Config().prompts, INSTRUCTIONS)

    assert prompt.startswith(Config().prompts.system + "\n\n")
    assert "AGENTS.md" in prompt
    assert prompt.endswith("Use tabs.")


def test_the_prompt_does_not_leak_the_absolute_path() -> None:
    """A transcript may be shared; the machine's layout should not ride along."""
    context = ProjectContext(path=Path("/home/someone/secret-project/AGENTS.md"), text="x")

    assert "/home/someone" not in build_system_prompt(Config().prompts, context)


def test_with_no_system_prompt_the_instructions_stand_alone() -> None:
    prompts = PromptsConfig(system="", project_instructions="From {name}: {text}")

    assert build_system_prompt(prompts, INSTRUCTIONS) == "From AGENTS.md: Use tabs."


def test_braces_in_the_instructions_are_left_alone() -> None:
    """Only the template is parsed; a file full of JSON must not be."""
    context = ProjectContext(path=Path("AGENTS.md"), text='Config is {"a": 1}.')

    assert build_system_prompt(PromptsConfig(), context) == 'Config is {"a": 1}.'


@pytest.mark.parametrize(
    ("template", "message"),
    [
        ("{text} from {file}", r"unknown placeholder\(s\) \{file\}"),
        ("{text} {}", r"unknown placeholder\(s\) \{\}"),
        ("{text.upper}", r"unknown placeholder\(s\) \{text\.upper\}"),
        ("Instructions from {name}", r"needs \{text\}"),
        ("{text", "Write {{ and }} for literal braces"),
        ("{text:d}", "Write {{ and }} for literal braces"),
    ],
    ids=["unknown", "positional", "attribute", "no-text", "unclosed", "bad-spec"],
)
def test_a_broken_template_fails_even_with_nothing_to_fill(template: str, message: str) -> None:
    """It fails in every directory, not only in the first one with an AGENTS.md."""
    with pytest.raises(ConfigError, match=rf"^prompts\.project_instructions: .*{message}"):
        build_system_prompt(PromptsConfig(project_instructions=template), None)


def test_literal_braces_are_written_doubled() -> None:
    prompts = PromptsConfig(system="", project_instructions="{{rules}}\n{text}")

    assert build_system_prompt(prompts, INSTRUCTIONS) == "{rules}\nUse tabs."
