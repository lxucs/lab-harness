"""Project instructions — what a repository tells the agent about itself.

A project documents its own conventions in a file at its root (`HARNESS.md`
or `AGENTS.md`, both pre-existing conventions rather than our invention).
Loading it into the system prompt saves the agent from rediscovering those
conventions from the code on every run.

Like `config.py`, this sits at the bottom of the dependency graph: it reads a
directory and returns data, knowing nothing about agents, tools, or the CLI.
`build_system_prompt` joins what it finds to the configured prompts, so Python
callers assemble the same system prompt the CLI sends.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from string import Formatter

from lab_harness.config import ConfigError, ProjectConfig, PromptsConfig

_log = logging.getLogger(__name__)

#: What `prompts.project_instructions` may refer to.
_PLACEHOLDERS = frozenset({"name", "text"})


@dataclass(frozen=True, slots=True)
class ProjectContext:
    """Instructions found in the working directory."""

    path: Path
    #: File contents, stripped. Never empty — a blank file yields no context
    #: at all rather than a `ProjectContext` with nothing in it.
    text: str
    #: Whether the file was cut at `max_bytes`.
    truncated: bool = False


def load_project_context(
    cwd: str | Path = ".", config: ProjectConfig | None = None
) -> ProjectContext | None:
    """Load the project's own instructions, or `None` if it has none.

    The first configured name that exists wins; the rest are not read. Two
    files would mean two sets of instructions with no stated precedence, and
    concatenating them would silently double any advice they share.

    "Exists" is the whole test — a blank `HARNESS.md` means *no* instructions
    rather than "fall through to `AGENTS.md`". One rule with no special case
    is easier to reason about, and it makes an empty file a usable way to opt
    a project out of an `AGENTS.md` written for some other tool.

    A file that exists but cannot be *read* raises rather than being skipped,
    for the reason `load_config` raises on a malformed file: instructions you
    meant to apply and that silently did nothing are worse than a startup
    failure you can see.
    """
    config = config or ProjectConfig()
    root = Path(cwd)
    for name in config.context_files:
        path = root / name
        if not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                # Bounded read, not read_bytes(): the cap is there to stop a
                # huge file reaching the context window, and loading it into
                # memory first would concede half the point.
                raw = handle.read(config.max_bytes + 1)
        except OSError as exc:
            raise ConfigError(f"{path}: cannot read project instructions ({exc.strerror})") from exc
        truncated = len(raw) > config.max_bytes
        # errors="replace" because the cut can land mid-codepoint, and a
        # mangled character is a better outcome than a crash over advisory text.
        text = raw[: config.max_bytes].decode("utf-8", errors="replace").strip()
        if not text:
            _log.debug("%s is empty: treating the project as having no instructions", path)
            return None
        if truncated:
            _log.warning("%s truncated to project.max_bytes=%d", path, config.max_bytes)
        _log.info("Project instructions from %s (%d characters)", path, len(text))
        return ProjectContext(path=path, text=text, truncated=truncated)
    _log.debug("No project instructions in %s (tried %s)", root, ", ".join(config.context_files))
    return None


def build_system_prompt(prompts: PromptsConfig, context: ProjectContext | None) -> str:
    """`prompts.system`, then the project's instructions when it has any.

    The template keeps the file name, so the model can cite and edit the
    source of a convention rather than treating it as an unattributable rule.
    It is checked even when there is nothing to fill it with: a broken one
    should fail in every directory, not only in those with an AGENTS.md.
    """
    _check_template(prompts.project_instructions)
    if context is None:
        return prompts.system
    block = prompts.project_instructions.format(name=context.path.name, text=context.text)
    return f"{prompts.system}\n\n{block}" if prompts.system else block


def _check_template(template: str) -> None:
    where = "prompts.project_instructions"
    try:
        fields = {field for _, field, _, _ in Formatter().parse(template) if field is not None}
        unknown = sorted(f"{{{field}}}" for field in fields - _PLACEHOLDERS)
        if not unknown:
            # A format spec the placeholders cannot take fails only here.
            template.format(name="", text="")
    except ValueError as exc:
        raise ConfigError(f"{where}: {exc}. Write {{{{ and }}}} for literal braces") from exc
    if unknown:
        raise ConfigError(
            f"{where}: unknown placeholder(s) {', '.join(unknown)}; the template "
            f"takes {{name}} and {{text}}. Write {{{{ and }}}} for literal braces"
        )
    if "text" not in fields:
        raise ConfigError(f"{where}: needs {{text}}, where the instructions go")
