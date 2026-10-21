"""The system prompt: the sections in `[prompts]`, filled in and joined.

This is the one place it is put together, so the CLI and a Python caller send
the same text. The sections, in the order they are sent:

1. `prompts.system` — `--system-prompt` and `--system-prompt-file` replace it
2. `prompts.harness`
3. `prompts.environment`, filled from an `Environment`
4. `prompts.context_management`
5. `prompts.skills`, wrapped around the list of skills found, when there are
   any
6. `prompts.project_instructions`, wrapped around the project's own file, when
   it has one

A blank line separates them, and an empty one is left out, so a config file
drops a section by setting it to "".

**Every section is a template**, filled from one set of variables: the fields
of `Environment`, plus `{skills}`, `{name}` and `{text}`. Writing one is
optional — a section that names none is sent as it stands. One naming a
variable that does not exist is an error at startup: a prompt that silently
says nothing where it meant to say something is worse than a failure you can
see.

A brace is a variable only when it holds a name and nothing else, which is
what keeps the rest of the brace-using world out of the way: `{"a": 1}`, `{}`
and `{ cwd }` are text, so a section quoting JSON needs no escaping. `{{` and
`}}` write a literal brace, for the one case that does — a `{cwd}` meant to
stay `{cwd}`.
"""

import re
from dataclasses import asdict, fields

from lab_harness.config import ConfigError, PromptsConfig
from lab_harness.prompts.environment import Environment
from lab_harness.prompts.project import ProjectContext

#: What a section may name. Every section is a template, so this is the whole
#: vocabulary of all of them.
VARIABLES = frozenset(f.name for f in fields(Environment)) | frozenset({"skills", "name", "text"})
#: A variable, or a doubled brace, which stands for one. The only three shapes
#: a brace takes, so anything else in a section is text.
_TOKEN = re.compile(r"\{\{|\}\}|\{([A-Za-z_]\w*)\}")
#: A variable, and nothing else: what that `{` has to be.
_WHOLE = re.compile(r"\{[A-Za-z_]\w*\}")
#: A `{` opening something name-shaped: it must be a whole variable, or a
#: mistake worth reporting — `{text:d}` was meant to be one, and is not.
_STRAY = re.compile(r"\{(?=[A-Za-z_])")


def build_system_prompt(
    prompts: PromptsConfig,
    environment: Environment,
    context: ProjectContext | None = None,
    skills: str = "",
) -> str:
    """The sections of `prompts`, filled from `environment`, with `context` last.

    `skills` is the list of skills found, already rendered — one line each, as
    `prompts.skills` expects it. Empty means none were found, or none are
    loadable, and the section is left out.

    The project's template is given the file's name rather than its path, so
    the model can cite and edit the source of a convention rather than treating
    it as an unattributable rule. Every section is checked even when there is
    nothing to fill one with: a broken one should fail in every directory, not
    only in those with an AGENTS.md or a skill.
    """
    values: dict[str, str] = asdict(environment) | {
        "git_enabled": str(environment.git_enabled).lower()
    }
    values["skills"] = skills
    values["name"] = context.path.name if context is not None else ""
    values["text"] = context.text if context is not None else ""
    _check_template("prompts.system", prompts.system)
    _check_template("prompts.harness", prompts.harness)
    _check_template("prompts.environment", prompts.environment)
    _check_template("prompts.context_management", prompts.context_management)
    # Only when there is a section to fill: as with every other one, "" is how
    # a config file drops it, and a dropped section needs no placeholder.
    if prompts.skills and "skills" not in _check_template("prompts.skills", prompts.skills):
        raise ConfigError("prompts.skills: needs {skills}, where the list goes")
    where = "prompts.project_instructions"
    if "text" not in _check_template(where, prompts.project_instructions):
        raise ConfigError(f"{where}: needs {{text}}, where the instructions go")

    sections = [
        prompts.system,
        prompts.harness,
        prompts.environment,
        prompts.context_management,
    ]
    if skills:
        sections.append(prompts.skills)
    if context is not None:
        sections.append(prompts.project_instructions)
    return "\n\n".join(
        filled for filled in (_fill(section, values) for section in sections) if filled
    )


def _check_template(where: str, template: str) -> set[str]:
    """Raise unless `template` names only variables; return those it names."""
    named = {match[1] for match in _TOKEN.finditer(template) if match[1] is not None}
    if unknown := sorted(f"{{{name}}}" for name in named - VARIABLES):
        raise ConfigError(
            f"{where}: unknown placeholder(s) {', '.join(unknown)}; the placeholders "
            f"are {_listing()}. Write {{{{ and }}}} for literal braces"
        )
    for match in _STRAY.finditer(template):
        if _WHOLE.match(template, match.start()) is None:
            raise ConfigError(
                f"{where}: {_fragment(template, match.start())} is not a placeholder. "
                f"Write {{{{ and }}}} for literal braces"
            )
    return named


def _fill(template: str, values: dict[str, str]) -> str:
    """`template` with each variable replaced, in one pass.

    One pass, so what a variable holds is never read as a template: project
    instructions quoting `{{` are not the template's to collapse.
    """
    return _TOKEN.sub(lambda m: values[m[1]] if m[1] is not None else m[0][0], template)


def _listing() -> str:
    """Every variable, in the order an error names them."""
    return ", ".join(sorted(f"{{{name}}}" for name in VARIABLES))


def _fragment(template: str, start: int) -> str:
    """The group of braces at `start`, to name in an error."""
    end = template.find("}", start)
    return template[start : end + 1] if end != -1 else template[start:]
