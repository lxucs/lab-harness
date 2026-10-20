"""Skills: discovery, frontmatter, and the tool that loads one.

Nothing here reads a real home: the autouse fixture points `LAB_HARNESS_HOME`
at a directory under `tmp_path`, which is also the working directory.
"""

import logging
from pathlib import Path

import pytest

from lab_harness.config import Config, ConfigError, SkillsConfig
from lab_harness.llm.types import TextBlock
from lab_harness.tools.base import ToolContext, ToolResult
from lab_harness.tools.skills import directories, index, load_skills, skill


def _write(root: Path, name: str, text: str) -> Path:
    """A skill `name` under `root`, as its SKILL.md."""
    path = root / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _skill(name: str = "commit", description: str = "Write a commit message", body: str = "Do it."):
    return f"---\nname: {name}\ndescription: {description}\n---\n{body}\n"


def _project(tmp_path: Path) -> Path:
    """The working directory's own skills directory."""
    return tmp_path / ".lab-harness" / "skills"


def _home(tmp_path: Path) -> Path:
    """Home's, which the fixture puts under tmp_path."""
    return tmp_path / "lab-harness-home" / "skills"


def _ctx(tmp_path: Path, config: Config | None = None) -> ToolContext:
    return ToolContext(cwd=str(tmp_path), call_id="c1", config=config or Config())


def _text(result: ToolResult) -> str:
    return "".join(b.text for b in result.content if isinstance(b, TextBlock))


# --- discovery ------------------------------------------------------------


def test_a_skill_is_a_directory_holding_a_skill_md(tmp_path: Path) -> None:
    _write(_project(tmp_path), "commit", _skill())

    (found,) = load_skills(tmp_path)

    assert found.name == "commit"
    assert found.description == "Write a commit message"
    assert found.body == "Do it."


def test_home_is_searched_too(tmp_path: Path) -> None:
    _write(_home(tmp_path), "release", _skill("release"))

    assert [s.name for s in load_skills(tmp_path)] == ["release"]


def test_the_working_directory_wins(tmp_path: Path) -> None:
    """A project overrides a skill of its own name rather than shadowing it
    by accident of ordering."""
    _write(_home(tmp_path), "commit", _skill(body="from home"))
    _write(_project(tmp_path), "commit", _skill(body="from the project"))

    (found,) = load_skills(tmp_path)

    assert found.body == "from the project"


def test_skills_are_sorted_by_name(tmp_path: Path) -> None:
    _write(_project(tmp_path), "zeta", _skill("zeta"))
    _write(_home(tmp_path), "alpha", _skill("alpha"))

    assert [s.name for s in load_skills(tmp_path)] == ["alpha", "zeta"]


def test_disabled_means_none_are_found(tmp_path: Path) -> None:
    _write(_project(tmp_path), "commit", _skill())
    config = Config(skills=SkillsConfig(enabled=False))

    assert load_skills(tmp_path, config) == ()


def test_the_directories_searched_are_the_projects_then_home(tmp_path: Path) -> None:
    assert directories(tmp_path) == (_project(tmp_path), _home(tmp_path))


# --- frontmatter ----------------------------------------------------------


def test_the_frontmatter_name_beats_the_directorys(tmp_path: Path) -> None:
    _write(_project(tmp_path), "commit-messages", _skill("commit"))

    (found,) = load_skills(tmp_path)

    assert found.name == "commit"
    assert found.path.parent.name == "commit-messages"


def test_a_skill_without_a_description_is_an_error(tmp_path: Path) -> None:
    path = _write(_project(tmp_path), "commit", "---\nname: commit\n---\nDo it.\n")

    with pytest.raises(ConfigError, match="needs a description"):
        load_skills(tmp_path)
    assert path.exists()


def test_a_skill_without_frontmatter_is_an_error(tmp_path: Path) -> None:
    _write(_project(tmp_path), "commit", "Do it.\n")

    with pytest.raises(ConfigError, match="no frontmatter"):
        load_skills(tmp_path)


def test_frontmatter_that_is_never_closed_is_an_error(tmp_path: Path) -> None:
    _write(_project(tmp_path), "commit", "---\ndescription: Write one\nDo it.\n")

    with pytest.raises(ConfigError, match="never closed"):
        load_skills(tmp_path)


def test_frontmatter_that_is_not_yaml_is_an_error(tmp_path: Path) -> None:
    _write(_project(tmp_path), "commit", "---\ndescription: [unclosed\n---\nDo it.\n")

    with pytest.raises(ConfigError, match="not valid YAML"):
        load_skills(tmp_path)


def test_frontmatter_that_is_not_a_table_is_an_error(tmp_path: Path) -> None:
    _write(_project(tmp_path), "commit", "---\n- one\n- two\n---\nDo it.\n")

    with pytest.raises(ConfigError, match="table of keys"):
        load_skills(tmp_path)


def test_keys_from_another_harness_are_ignored(tmp_path: Path) -> None:
    text = "---\nname: commit\ndescription: Write one\nversion: 2\nmodel: inherit\n---\nDo it.\n"
    _write(_project(tmp_path), "commit", text)

    (found,) = load_skills(tmp_path)

    assert found.name == "commit"


def test_a_long_body_is_cut(tmp_path: Path) -> None:
    _write(_project(tmp_path), "commit", _skill(body="x" * 400))
    config = Config(skills=SkillsConfig(max_bytes=100))

    (found,) = load_skills(tmp_path, config)

    assert len(found.body) < 400
    assert "truncated at skills.max_bytes=100" in found.body


# --- allowed-tools --------------------------------------------------------


def test_allowed_tools_take_the_tools_this_run_has(tmp_path: Path) -> None:
    text = (
        "---\nname: commit\ndescription: Write one\nallowed-tools: Bash(git:*), Read\n---\nDo it.\n"
    )
    _write(_project(tmp_path), "commit", text)

    (found,) = load_skills(tmp_path)

    assert found.allowed_tools == ("bash", "read")


def test_allowed_tools_may_be_a_list(tmp_path: Path) -> None:
    text = (
        "---\nname: commit\ndescription: Write one\nallowed-tools:\n  - Bash(git:*)\n  - "
        "WebFetch\n---\nDo it.\n"
    )
    _write(_project(tmp_path), "commit", text)

    (found,) = load_skills(tmp_path)

    assert found.allowed_tools == ("bash", "web_fetch")


def test_an_unknown_allowed_tool_is_said(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    text = "---\nname: commit\ndescription: Write one\nallowed-tools: Task, Read\n---\nDo it.\n"
    _write(_project(tmp_path), "commit", text)

    with caplog.at_level(logging.WARNING):
        (found,) = load_skills(tmp_path)

    assert found.allowed_tools == ("read",)
    assert "Task is not an enabled tool" in caplog.text


# --- the index ------------------------------------------------------------


def test_the_index_is_one_line_per_skill(tmp_path: Path) -> None:
    _write(_project(tmp_path), "commit", _skill("commit", "Write a commit message"))

    assert index(load_skills(tmp_path)) == "- commit — Write a commit message"


def test_the_index_of_nothing_is_empty() -> None:
    assert index(()) == ""


# --- the tool -------------------------------------------------------------


async def test_the_tool_returns_the_instructions(tmp_path: Path) -> None:
    _write(_project(tmp_path), "commit", _skill(body="Steps:\n1. Look"))

    result = await skill.execute({"name": "commit"}, _ctx(tmp_path))

    assert not result.is_error
    assert _text(result) == "# commit\n\nSteps:\n1. Look"


async def test_the_tool_names_the_other_files_beside_them(tmp_path: Path) -> None:
    _write(_project(tmp_path), "commit", _skill())
    reference = _project(tmp_path) / "commit" / "guide.md"
    reference.write_text("more", encoding="utf-8")

    result = await skill.execute({"name": "commit"}, _ctx(tmp_path))

    assert "- .lab-harness/skills/commit/guide.md" in _text(result)


async def test_the_tool_hands_the_allowed_tools_to_the_gate(tmp_path: Path) -> None:
    text = "---\nname: commit\ndescription: Write one\nallowed-tools: Bash(git:*)\n---\nDo it.\n"
    _write(_project(tmp_path), "commit", text)

    result = await skill.execute({"name": "commit"}, _ctx(tmp_path))

    assert result.details["allowed_tools"] == ["bash"]


async def test_an_unknown_name_says_what_is_available(tmp_path: Path) -> None:
    _write(_project(tmp_path), "commit", _skill())

    result = await skill.execute({"name": "release"}, _ctx(tmp_path))

    assert result.is_error
    assert "No skill named 'release'. Available: commit" in _text(result)


async def test_an_unknown_name_with_no_skills_says_so(tmp_path: Path) -> None:
    result = await skill.execute({"name": "commit"}, _ctx(tmp_path))

    assert "and no skills are loaded" in _text(result)


async def test_the_tool_reads_from_the_calls_own_working_directory(tmp_path: Path) -> None:
    """The rule every tool follows: cwd, never the process's."""
    elsewhere = tmp_path / "elsewhere"
    _write(elsewhere / ".lab-harness" / "skills", "commit", _skill())

    assert not (await skill.execute({"name": "commit"}, _ctx(elsewhere))).is_error
    assert (await skill.execute({"name": "commit"}, _ctx(tmp_path))).is_error


# --- selection ------------------------------------------------------------


def test_selected_keeps_only_those_named(tmp_path: Path) -> None:
    _write(_project(tmp_path), "commit", _skill("commit"))
    _write(_home(tmp_path), "release", _skill("release"))
    config = Config(skills=SkillsConfig(selected=("commit",)))

    assert [s.name for s in load_skills(tmp_path, config)] == ["commit"]


def test_empty_selection_is_every_one_found(tmp_path: Path) -> None:
    _write(_project(tmp_path), "commit", _skill("commit"))
    _write(_home(tmp_path), "release", _skill("release"))

    assert [s.name for s in load_skills(tmp_path)] == ["commit", "release"]


def test_a_selected_name_nothing_provides_is_said(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A config file runs against directories that differ by machine, so a
    missing skill is a warning rather than a startup failure."""
    _write(_project(tmp_path), "commit", _skill("commit"))
    config = Config(skills=SkillsConfig(selected=("commit", "release")))

    with caplog.at_level(logging.WARNING):
        found = load_skills(tmp_path, config)

    assert [s.name for s in found] == ["commit"]
    assert "release is selected but was not found" in caplog.text
