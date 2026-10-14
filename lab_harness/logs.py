"""Where diagnostics go and how loud they are.

Modules across the package take a logger the ordinary way
(`logging.getLogger(__name__)`) and emit records unconditionally. Nothing is
printed until an application calls `configure_logging`, and only `cli/` does.
That is the library/application split the stdlib prescribes, and it is why
importing `lab_harness` never installs a handler on anybody's root logger.

Named `logs` rather than `logging` on purpose. Absolute imports mean a module
here would not actually shadow the stdlib, but a reader seeing
`lab_harness/logging.py` has to work that out before trusting any `import logging`
in the package.

Like `config.py`, this is a leaf module: it reads the schema and the
environment, and knows nothing about agents, tools, or the CLI.
"""

import logging
import os
import sys
from typing import Protocol

from lab_harness.config import ConfigError, LoggingConfig

#: Accepted level names, mapped to their `logging` constants. Deliberately a
#: subset of the stdlib's: `critical` and `notset` are noise in a program whose
#: worst outcome is a failed run, and every extra name is one more thing a
#: config file can get wrong.
LEVELS: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

#: Root of our logger tree. Derived from the package rather than written out,
#: so configuration survives a rename. Configuring *this* logger rather than
#: the root one is what keeps httpx's and anthropic's own debug output off the
#: screen when ours is turned on — `--log-level debug` should mean this
#: program, not every library it imports.
ROOT_LOGGER = __package__ or "lab_harness"


class Stream(Protocol):
    """What records can be written to: a file, or anything with `write`."""

    def write(self, text: str, /) -> object: ...


#: Marks the handler this module installs. Repeated calls replace ours and
#: leave alone any handler an embedding application attached itself.
_MARKER = "_lab_harness_handler"


def resolve_level(requested: str | None = None, config: LoggingConfig | None = None) -> str:
    """The level to use, as a canonical lowercase name.

    Precedence, highest first: the explicit argument (`--log-level`), the env
    var named by `logging.level_env`, then `logging.level`. That is the
    project-wide order — a flag beats the environment, which beats a file.

    An unusable name raises rather than falling back to a default. A level you
    asked for and did not get is the same class of bug as a config key that
    silently does nothing.
    """
    config = config or LoggingConfig()
    # An empty env var counts as unset: `LAB_HARNESS_LOG_LEVEL=` left in a shell
    # profile should not become a startup error about a level named "".
    from_env = os.environ.get(config.level_env, "").strip()
    wanted = requested or from_env or config.level
    name = wanted.strip().lower()
    if name not in LEVELS:
        raise ConfigError(f"Unknown log level {wanted!r}. Available: {', '.join(LEVELS)}.")
    return name


def configure_logging(
    requested: str | None = None,
    config: LoggingConfig | None = None,
    *,
    stream: Stream | None = None,
) -> str:
    """Install the handler for our logger tree. Returns the level applied.

    Idempotent: our previous handler is replaced rather than added to, so
    calling this again — which `cli/main.py` does, once before config is loaded
    and once after, and once more to hand the terminal its records — leaves
    one handler and one copy of each line.
    """
    config = config or LoggingConfig()
    level = resolve_level(requested, config)

    # sys.stderr is read now rather than captured at import, so a caller that
    # has redirected it gets the redirection.
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter(config.format))
    setattr(handler, _MARKER, True)

    logger = logging.getLogger(ROOT_LOGGER)
    for existing in [h for h in logger.handlers if getattr(h, _MARKER, False)]:
        logger.removeHandler(existing)
        existing.close()
    logger.addHandler(handler)
    logger.setLevel(LEVELS[level])
    # Our records stop here. Propagating would also hand them to the root
    # logger, where an embedding application's handler would print them a
    # second time, on a stream we do not control.
    logger.propagate = False
    return level
