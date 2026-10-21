"""What this program decides for itself: how to ask, and where things are.

The agent's own settings — model, keys, prompts, tool set — are harness
config, read from `config.local.toml` beside this file. This reads the rest
from `search_agent.toml`: the request that opens every run, and the paths it
reads and writes.
"""

import tomllib
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

#: Configured paths are resolved from here, so `python -m search_agent` behaves
#: the same from any working directory.
HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "search_agent.toml"

#: Used when a settings file leaves the request out. `{query}` is the line from
#: the query file; the corpus is named in `[corpus]`, not here.
DEFAULT_REQUEST = """{query}

Answer from the corpus.
"""

DEFAULT_TOOLS = ("ls", "glob", "grep", "read", "search_docs")


@dataclass(frozen=True)
class Settings:
    """One run's settled choices."""

    request: str
    #: The agent's working directory: where the settings file lives, so a path
    #: in this file and a path in a tool call mean the same thing.
    cwd: Path
    docs: Path
    queries: Path
    out: Path
    tools: tuple[str, ...]

    @property
    def corpus(self) -> str:
        """The corpus as the agent sees it: `docs` relative to its directory."""
        path = self.docs
        with suppress(ValueError):
            path = path.relative_to(self.cwd)
        return str(path)

    def first_turn(self, query: str) -> str:
        """The message that opens a run: the request, with the query in it."""
        return self.request.format(query=query)


def load_settings(path: Path | None = None) -> Settings:
    """`search_agent.toml`, with the defaults for whatever it leaves out.

    Relative paths are read against the file they came from, so a copy of
    this program kept elsewhere can name its own `docs/` and `runs/`.
    """
    file = path or CONFIG_FILE
    data = tomllib.loads(file.read_text(encoding="utf-8")) if file.is_file() else {}
    request = data.get("request", {})
    corpus = data.get("corpus", {})
    run = data.get("run", {})
    return Settings(
        request=str(request.get("template", DEFAULT_REQUEST)),
        cwd=file.parent,
        docs=_under(corpus.get("docs", "data/docs"), file.parent),
        queries=_under(run.get("queries", "data/queries.jsonl"), file.parent),
        out=_under(run.get("out_dir", "runs"), file.parent),
        tools=tuple(run.get("tools", DEFAULT_TOOLS)),
    )


def _under(value: object, base: Path) -> Path:
    """A configured path: absolute as given, otherwise under its config file."""
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else base / path
