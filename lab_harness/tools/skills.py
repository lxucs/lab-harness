"""Skills: instructions for one kind of task, loaded when one is asked for.

A skill is a directory holding a `SKILL.md`:

    ---
    name: commit-messages
    description: Write a commit message for staged changes
    allowed-tools: Bash(git:*), Read
    ---
    Steps, examples, and whatever else the task needs.

The frontmatter is YAML. `description` is required — it is the whole basis on
which one is picked — and `name` defaults to the directory's. Unknown keys are
ignored, so a file written for another harness still loads.

**Only the name and the description ride in every request**, as a list in the
system prompt; the instructions come back when the model asks for them by name.
That is the point of the split: a directory of skills costs a list, not the
text of all of them.

`allowed-tools` names tools this skill may use without asking. They are
resolved to the tools this run actually has, so a name from another harness's
vocabulary is dropped rather than granted, and they leave in the result's
`details` — for the approval gate to read, since a tool cannot reach it.

**Which apply is `skills.selected`**, empty meaning every one found. Nothing
else filters: the `skill` tool, the list in the system prompt and the
terminal's menu all call `load_skills`, which is why changing the selection in
one place changes it in all of them.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from lab_harness.config import PROJECT_DIRECTORY, Config, ConfigError, lab_harness_home
from lab_harness.tools.base import ToolContext, ToolResult, fail, ok, tool
from lab_harness.tools.fs import display

_log = logging.getLogger(__name__)

#: The file a skill's directory is recognized by.
SKILL_FILE = "SKILL.md"
#: What opens and closes its frontmatter.
_DELIMITER = "---"
#: Added to a body cut at `skills.max_bytes`, so the model knows it is partial.
_TRUNCATION_NOTE = "\n\n[truncated at skills.max_bytes={limit}]"


@dataclass(frozen=True, slots=True)
class Skill:
    """One skill, as read from disk."""

    #: How it is named and asked for. The directory's name unless the
    #: frontmatter says otherwise.
    name: str
    #: When to use it. The only part of it in every request.
    description: str
    #: The `SKILL.md`. Its directory, whose other files it may send the
    #: model to read, is `path.parent`.
    path: Path
    #: The instructions: everything after the frontmatter, stripped.
    body: str
    #: Tools it is allowed to use, as enabled names, in `tools.enabled` order.
    allowed_tools: tuple[str, ...] = ()


def directories(cwd: str | Path = ".", config: Config | None = None) -> tuple[Path, ...]:
    """Where skills are looked for: the working directory's, then home's."""
    config = config or Config()
    return (
        Path(cwd) / PROJECT_DIRECTORY / config.skills.directory_name,
        lab_harness_home(config.session) / config.skills.directory_name,
    )


def load_skills(cwd: str | Path = ".", config: Config | None = None) -> tuple[Skill, ...]:
    """Every skill found for `cwd`, by name.

    A name the working directory already used is not taken again from home, so
    a project overrides a skill of its own name rather than shadowing it by
    accident of ordering. A file that exists but is unusable raises, for the
    reason a malformed config file does: a skill you meant to have and that
    silently was not there is worse than a failure you can see.
    """
    config = config or Config()
    if not config.skills.enabled:
        _log.debug("Skills are off: skills.enabled is false")
        return ()
    found: dict[str, Skill] = {}
    for root in directories(cwd, config):
        for path in sorted(root.glob(f"*/{SKILL_FILE}")):
            skill = _read(path, config)
            if skill.name in found:
                _log.debug("%s: %s already found in an earlier directory", path, skill.name)
                continue
            found[skill.name] = skill
    selected = _selected(found, config)
    _log.info("Loaded %d skill(s) for %s", len(selected), cwd)
    return tuple(found[name] for name in sorted(selected))


def _selected(found: dict[str, Skill], config: Config) -> dict[str, Skill]:
    """The skills `skills.selected` keeps, which is every one when it is empty.

    A name it asks for and no directory provides is warned about rather than
    refused: the same config file runs against different directories, and a
    missing skill is a smaller surprise than a startup failure.
    """
    wanted = config.skills.selected
    if not wanted:
        return found
    kept = {name: found[name] for name in wanted if name in found}
    for name in wanted:
        if name not in found:
            _log.warning("Skill %s is selected but was not found", name)
    return kept


def index(skills: Iterable[Skill]) -> str:
    """The list `prompts.skills` is filled with: one line per skill."""
    return "\n".join(f"- {skill.name} — {skill.description}" for skill in skills)


class SkillParams(BaseModel):
    name: str = Field(description="Name of the skill, exactly as it is listed.")


@tool(
    name="skill",
    description=(
        "Load a skill: instructions for one kind of task, listed with what it "
        "is for in the system prompt. Ask for one whose description matches "
        "what has been asked for, before starting."
    ),
)
async def skill(p: SkillParams, ctx: ToolContext) -> ToolResult:
    found = load_skills(ctx.cwd, ctx.config)
    for one in found:
        if one.name == p.name:
            return ok(_body(one, ctx), allowed_tools=list(one.allowed_tools))
    available = ", ".join(one.name for one in found)
    return fail(
        f"No skill named {p.name!r}. Available: {available}"
        if available
        else f"No skill named {p.name!r}, and no skills are loaded."
    )


def _body(one: Skill, ctx: ToolContext) -> str:
    """The instructions, and what else the directory holds.

    The other files are named so the model can read the one it needs — a
    skill that ships a script or a long reference is not meant to have either
    inlined into the instructions.
    """
    lines = [f"# {one.name}", "", one.body]
    others = sorted(p for p in one.path.parent.rglob("*") if p.is_file() and p != one.path)
    if others:
        lines += ["", "Also in this skill's directory:"]
        lines += [f"- {display(path, ctx)}" for path in others]
    return "\n".join(lines)


def _read(path: Path, config: Config) -> Skill:
    """One `SKILL.md`, as a `Skill`."""
    limit = config.skills.max_bytes
    try:
        with path.open("rb") as handle:
            # Bounded, and for the reason project.py reads that way: the cap is
            # there to keep a huge file out of the context, and loading it all
            # first would concede half the point.
            raw = handle.read(limit + 1)
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read ({exc.strerror})") from exc
    # errors="replace" because the cut can land mid-codepoint.
    text = raw[:limit].decode("utf-8", errors="replace")
    frontmatter, body = _split(text, path)
    if len(raw) > limit:
        body += _TRUNCATION_NOTE.format(limit=limit)
    description = _text(frontmatter, "description", path)
    if not description:
        raise ConfigError(f"{path}: needs a description, saying when to use it")
    return Skill(
        name=_text(frontmatter, "name", path) or path.parent.name,
        description=description,
        path=path,
        body=body,
        allowed_tools=_allowed_tools(frontmatter.get("allowed-tools"), config, path),
    )


def _split(text: str, path: Path) -> tuple[dict[str, object], str]:
    """The frontmatter and the body of a `SKILL.md`, as YAML and as text."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != _DELIMITER:
        raise ConfigError(f"{path}: has no frontmatter; it begins with {_DELIMITER} on a line")
    for end in range(1, len(lines)):
        if lines[end].strip() != _DELIMITER:
            continue
        try:
            loaded = yaml.safe_load("\n".join(lines[1:end]))
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: frontmatter is not valid YAML ({exc})") from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{path}: frontmatter is a YAML table of keys, not {loaded!r}")
        return loaded, "\n".join(lines[end + 1 :]).strip()
    raise ConfigError(f"{path}: frontmatter is never closed with {_DELIMITER}")


def _text(frontmatter: dict[str, object], key: str, path: Path) -> str:
    """`key` of the frontmatter as stripped text, "" if it is absent or not text."""
    value = frontmatter.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ConfigError(f"{path}: {key} is text, not {value!r}")
    return value.strip()


def _allowed_tools(value: object, config: Config, path: Path) -> tuple[str, ...]:
    """`allowed-tools` as the enabled tools it names.

    An entry may carry an argument pattern, as in `Bash(git:*)`: what is before
    the parenthesis is the tool. Names are matched without case or underscores,
    so `WebFetch` finds `web_fetch`. One naming a tool this run does not have is
    dropped with a warning — a skill written for another harness names tools
    that are not here, and granting nothing is the only sane reading of those.
    """
    if value is None:
        return ()
    entries = value.split(",") if isinstance(value, str) else value
    if not isinstance(entries, (list, tuple)):
        raise ConfigError(f"{path}: allowed-tools is a list or a comma-separated string")
    resolved: list[str] = []
    for entry in entries:
        name = _named(str(entry), config.tools.enabled)
        if name is None:
            _log.warning("%s: %s is not an enabled tool; not allowed", path, entry)
            continue
        resolved.append(name)
    return tuple(dict.fromkeys(resolved))


def _named(entry: str, enabled: tuple[str, ...]) -> str | None:
    """The enabled tool `entry` names, or None."""
    wanted = entry.split("(")[0].strip().lower().replace("_", "")
    return next((name for name in enabled if name.replace("_", "") == wanted), None)
