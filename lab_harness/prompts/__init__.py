"""The system prompt: what the model is told besides the conversation and the tools.

`prompts.toml` holds the text, the default of the `[prompts]` config section.
`system.py` fills it in and joins it, from two inputs read beside it:
`environment.py`, where the agent runs, and `project.py`, the project's own
`AGENTS.md`.

The package imports only `config.py`. It reads a directory and the machine and
returns a string, knowing nothing about agents, tools, or the CLI; the agent
is handed the finished prompt. The CLI and a Python caller build it the same
way.
"""

from lab_harness.prompts.environment import Environment, detect_environment
from lab_harness.prompts.project import ProjectContext, load_project_context
from lab_harness.prompts.system import build_system_prompt

__all__ = [
    "Environment",
    "ProjectContext",
    "build_system_prompt",
    "detect_environment",
    "load_project_context",
]
