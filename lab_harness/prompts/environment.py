"""Where the agent runs — what `prompts.environment` is filled with.

Read from the machine once, when the prompt is built, and returned as plain
data, so a Python caller can pass a fixed `Environment` instead and send the
same prompt on any machine.
"""

import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Environment:
    """What `prompts.environment` is filled with. Each field is a placeholder."""

    #: The working directory, absolute.
    cwd: str
    #: Whether `cwd` is inside a git repository. Sent as "true" or "false".
    git_enabled: bool
    #: `sys.platform`: "darwin", "linux", "win32".
    system_platform: str
    #: The shell that runs commands.
    shell: str
    #: The OS and its release, e.g. "Darwin 23.4.0".
    os_version: str
    #: The id of the model the agent runs.
    agent_model: str


def detect_environment(cwd: str | Path, model: str) -> Environment:
    """The environment of an agent running `model` in `cwd`, on this machine.

    A Python caller that later changes `agent.state.model` builds the prompt
    again to keep the two in step.
    """
    directory = Path(cwd).resolve()
    return Environment(
        cwd=str(directory),
        # A file as well as a directory: a worktree's `.git` is a file.
        git_enabled=any((d / ".git").exists() for d in (directory, *directory.parents)),
        system_platform=sys.platform,
        shell=_shell(),
        os_version=f"{platform.system()} {platform.release()}",
        agent_model=model,
    )


def _shell() -> str:
    """The shell `asyncio.create_subprocess_shell` starts, which runs commands.

    Not `$SHELL`: that is the user's login shell, and a command written for zsh
    can fail in the /bin/sh that actually runs it. These are the subprocess
    module's own choices, so they are constants here rather than config.
    """
    if sys.platform == "win32":
        return os.environ.get("COMSPEC", "cmd.exe")
    return "/bin/sh"
