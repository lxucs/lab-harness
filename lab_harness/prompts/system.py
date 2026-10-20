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
drops a section by setting it to "". Only `environment`, `skills` and
`project_instructions` are templates. The others are sent as written, so a
brace in them, or in a prompt given with `--system-prompt`, needs no escaping.
"""

from dataclasses import asdict, fields
from string import Formatter

from lab_harness.config import ConfigError, PromptsConfig
from lab_harness.prompts.environment import Environment
from lab_harness.prompts.project import ProjectContext

#: What `prompts.environment` may refer to.
ENVIRONMENT_PLACEHOLDERS = frozenset(f.name for f in fields(Environment))
#: What `prompts.project_instructions` may refer to.
INSTRUCTIONS_PLACEHOLDERS = frozenset({"name", "text"})
#: What `prompts.skills` may refer to.
SKILLS_PLACEHOLDERS = frozenset({"skills"})


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
    it as an unattributable rule. Every template is checked even when there is
    nothing to fill one with: a broken one should fail in every directory, not
    only in those with an AGENTS.md or a skill.
    """
    _check_template("prompts.environment", prompts.environment, ENVIRONMENT_PLACEHOLDERS)
    where = "prompts.project_instructions"
    named = _check_template(where, prompts.project_instructions, INSTRUCTIONS_PLACEHOLDERS)
    if "text" not in named:
        raise ConfigError(f"{where}: needs {{text}}, where the instructions go")
    # Only when there is a section to fill: as with every other one, "" is how
    # a config file drops it.
    if prompts.skills and "skills" not in _check_template(
        "prompts.skills", prompts.skills, SKILLS_PLACEHOLDERS
    ):
        raise ConfigError("prompts.skills: needs {skills}, where the list goes")

    values = asdict(environment) | {"git_enabled": str(environment.git_enabled).lower()}
    sections = [
        prompts.system,
        prompts.harness,
        prompts.environment.format(**values),
        prompts.context_management,
    ]
    if skills:
        sections.append(prompts.skills.format(skills=skills))
    if context is not None:
        sections.append(
            prompts.project_instructions.format(name=context.path.name, text=context.text)
        )
    return "\n\n".join(section for section in sections if section)


def _check_template(where: str, template: str, placeholders: frozenset[str]) -> set[str]:
    """Raise unless `template` refers only to `placeholders`; return those it uses."""
    try:
        named = {field for _, field, _, _ in Formatter().parse(template) if field is not None}
        unknown = sorted(f"{{{field}}}" for field in named - placeholders)
        if not unknown:
            # A format spec the placeholders cannot take fails only here.
            template.format(**dict.fromkeys(placeholders, ""))
    except ValueError as exc:
        raise ConfigError(f"{where}: {exc}. Write {{{{ and }}}} for literal braces") from exc
    if unknown:
        takes = ", ".join(sorted(f"{{{name}}}" for name in placeholders))
        raise ConfigError(
            f"{where}: unknown placeholder(s) {', '.join(unknown)}; the template "
            f"takes {takes}. Write {{{{ and }}}} for literal braces"
        )
    return named
