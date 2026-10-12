"""Configuration schema and defaults.

This module is the single list of every tunable in the system. Code reads
values from a resolved `Config`; it never hardcodes them. See
docs/contributing.md.

`load_config()` layers TOML files over these defaults, lowest precedence
first: `<home>/config.toml`, `./.harness/config.toml`, then
`./.harness/config.local.toml` — the last of which is gitignored and is the
only one that may hold a credential.

Environment variables sit above all of them, but there is deliberately no
generic env-overlay machinery here. Each env var is read where it is used
(`HARNESS_HOME` below, `HARNESS_MODEL` in llm/registry.py, credentials in the
providers and backends), and its *name* is itself a config key — so a
deployment that uses different names changes config rather than code.

This module is the bottom of the dependency graph and imports nothing from the
rest of the package.
"""

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AgentConfig:
    #: Hard ceiling on provider requests in one run. Guards against a model
    #: that calls tools forever.
    max_turns: int = 50
    #: Run independent tool calls from one message concurrently.
    parallel_tools: bool = True


@dataclass(frozen=True, slots=True)
class ReadConfig:
    #: Lines returned when the caller gives no explicit limit.
    default_limit: int = 2_000
    #: Longer lines are cut, so one minified file cannot flood the context.
    max_line_length: int = 2_000
    #: Files above this are refused rather than read — the model cannot use a
    #: megabyte of text, and sending it is expensive.
    max_bytes: int = 256_000


@dataclass(frozen=True, slots=True)
class ListConfig:
    #: Entries returned per directory listing.
    max_entries: int = 1_000


@dataclass(frozen=True, slots=True)
class BashConfig:
    #: Default wall-clock ceiling for one command.
    timeout_seconds: int = 120
    #: Highest timeout a call may request. A model asking for an hour is a bug.
    max_timeout_seconds: int = 600
    #: Combined stdout+stderr kept. The head and tail are preserved and the
    #: middle elided, because both ends usually matter and the middle rarely does.
    max_output_bytes: int = 30_000
    #: How often streaming output is forwarded to `on_update`, in seconds.
    update_interval_seconds: float = 0.5


@dataclass(frozen=True, slots=True)
class SearchConfig:
    #: Matches returned by one grep.
    max_results: int = 200
    #: Paths returned by one glob.
    max_files: int = 500
    #: Longer matching lines are clipped in grep output.
    max_line_length: int = 400
    #: Directory names never descended into. Searching .git or node_modules
    #: finds nothing useful and costs seconds.
    ignored_directories: tuple[str, ...] = (
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".tox",
    )


@dataclass(frozen=True, slots=True)
class ApprovalConfig:
    """Which tools must be approved before they run.

    The gate is a `before_tool_call` hook, not loop logic. An empty `required`
    means everything runs unprompted — which is the `-p` default, since a
    one-shot run has nobody at the keyboard to answer.
    """

    #: Tools that prompt before executing.
    required: tuple[str, ...] = ("bash", "write", "edit")
    #: Answering "always" is remembered for the session, never persisted.
    allow_remember: bool = True


@dataclass(frozen=True, slots=True)
class TavilyConfig:
    """A search API built for agents. The default `web_search` backend."""

    #: Env var consulted first for the key. The key may also be set as
    #: `api_key` below, but only in config.local.toml — never a committed file.
    api_key_env: str = "TAVILY_API_KEY"
    #: Empty means "not configured here"; look in the environment instead.
    api_key: str = ""
    base_url: str = "https://api.tavily.com"
    #: "basic" spends one API credit per search, "advanced" two for better
    #: extraction. Basic by default so a free tier lasts.
    search_depth: str = "basic"
    #: Request full page text beside each snippet. Off by default: raw content
    #: for five results is usually far more than the answer needs, and
    #: `web_fetch` can get it for the one result that matters.
    include_raw_content: bool = False


@dataclass(frozen=True, slots=True)
class DdgsConfig:
    """A keyless metasearch backend. Scrapes, so it is rate-limited and brittle."""

    #: Upstream engine to query — "auto" lets ddgs choose and rotate. Named
    #: `engine` rather than `backend` so it cannot be confused with
    #: `web.backend`, which selects ddgs itself.
    engine: str = "auto"
    #: Result region, e.g. "us-en", "uk-en", "ru-ru".
    region: str = "us-en"


@dataclass(frozen=True, slots=True)
class WebConfig:
    #: Which backend `web_search` uses: "tavily" or "ddgs".
    backend: str = "tavily"
    #: Results requested per search.
    max_results: int = 5
    #: Wall-clock limit for one search or one fetch.
    timeout_seconds: float = 30.0
    #: Largest response `web_fetch` will download, before extraction.
    max_bytes: int = 2_000_000
    #: Characters of extracted text handed to the model. A long article costs
    #: more context than it is usually worth.
    max_content_chars: int = 50_000
    #: Sent on every outbound request. Some sites refuse an unidentified client.
    user_agent: str = "harness/0.1 (+https://github.com/lxucs/harness)"
    tavily: TavilyConfig = field(default_factory=TavilyConfig)
    ddgs: DdgsConfig = field(default_factory=DdgsConfig)


@dataclass(frozen=True, slots=True)
class ToolsConfig:
    read: ReadConfig = field(default_factory=ReadConfig)
    list: ListConfig = field(default_factory=ListConfig)
    bash: BashConfig = field(default_factory=BashConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    web: WebConfig = field(default_factory=WebConfig)
    approval: ApprovalConfig = field(default_factory=ApprovalConfig)


@dataclass(frozen=True, slots=True)
class AnthropicConfig:
    #: Env vars consulted for credentials, in precedence order. The values
    #: themselves never live in config — see docs/contributing.md.
    auth_token_env: str = "ANTHROPIC_AUTH_TOKEN"
    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url_env: str = "ANTHROPIC_BASE_URL"
    #: Sampling ceiling when the model declares none.
    default_max_tokens: int = 8_192
    #: Thinking budget when reasoning is requested, in tokens.
    thinking_budget_tokens: int = 4_096


@dataclass(frozen=True, slots=True)
class ProvidersConfig:
    anthropic: AnthropicConfig = field(default_factory=AnthropicConfig)


@dataclass(frozen=True, slots=True)
class SessionConfig:
    #: Root for all harness state. `~` is expanded at use.
    home: str = "~/.harness"
    #: Env var that overrides `home`. The value is a machine path, so it lives
    #: in the environment rather than in a committed config file.
    home_env: str = "HARNESS_HOME"
    #: Subdirectory of `home` holding per-project session directories.
    directory_name: str = "sessions"
    #: Write transcripts to disk at all. Off means nothing is persisted.
    enabled: bool = True
    #: Schema version written into each file's header. Readers use it to refuse
    #: files they do not understand. A genuine protocol constant, not a knob.
    version: int = 1


@dataclass(frozen=True, slots=True)
class TerminalConfig:
    #: Tools whose default tool set the interactive terminal enables.
    enabled_tools: tuple[str, ...] = (
        "read",
        "write",
        "edit",
        "ls",
        "bash",
        "grep",
        "glob",
        "web_search",
        "web_fetch",
    )
    #: Lines of tool output shown inline before collapsing to a summary.
    max_result_lines: int = 12
    #: Render assistant text as markdown rather than plain text.
    markdown: bool = True


@dataclass(frozen=True, slots=True)
class Config:
    agent: AgentConfig = field(default_factory=AgentConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    terminal: TerminalConfig = field(default_factory=TerminalConfig)
    session: SessionConfig = field(default_factory=SessionConfig)


class ConfigError(Exception):
    """A config file was unreadable or contained something unusable."""


#: Per-project config directory, relative to the working directory.
PROJECT_DIRECTORY = ".harness"
CONFIG_FILE = "config.toml"
#: Gitignored sibling of CONFIG_FILE. The only file that may hold a credential.
LOCAL_CONFIG_FILE = "config.local.toml"


def harness_home(config: SessionConfig | None = None) -> Path:
    """The harness home directory, honoring the configured env override."""
    config = config or SessionConfig()
    override = os.environ.get(config.home_env)
    return Path(override).expanduser() if override else Path(config.home).expanduser()


def config_paths(cwd: str | Path = ".") -> tuple[Path, ...]:
    """The files `load_config` reads, in increasing order of precedence.

    Home is located from the *default* session config rather than from a
    loaded one. Resolving where to look for config using config that has not
    been loaded yet is circular; `HARNESS_HOME` is the escape hatch.
    """
    project = Path(cwd) / PROJECT_DIRECTORY
    return (
        harness_home() / CONFIG_FILE,
        project / CONFIG_FILE,
        project / LOCAL_CONFIG_FILE,
    )


def load_config(cwd: str | Path = ".") -> Config:
    """Resolve configuration from defaults plus any config files present.

    Missing files are normal and skipped. A file that exists but is malformed
    is an error: a config you *meant* to apply and that silently did nothing is
    worse than a startup failure.
    """
    config = Config()
    for path in config_paths(cwd):
        data = _read_toml(path)
        if data is not None:
            config = _apply(config, data, source=path, trail=())
    return config


def _read_toml(path: Path) -> Mapping[str, Any] | None:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read ({exc.strerror})") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML ({exc})") from exc


def _apply[T](instance: T, data: Mapping[str, Any], *, source: Path, trail: tuple[str, ...]) -> T:
    """Return a copy of `instance` with `data` applied, recursing into sections.

    Unknown keys raise rather than being ignored. A typo that silently leaves
    the default in place is the kind of bug found days later, by the setting
    apparently having no effect.
    """
    known = {f.name: f for f in fields(instance)}  # type: ignore[arg-type]
    updates: dict[str, object] = {}
    for key, value in data.items():
        where = ".".join((*trail, key))
        if key not in known:
            available = ", ".join(sorted(known)) or "(none)"
            raise ConfigError(f"{source}: unknown setting {where!r}. Available here: {available}")
        current = getattr(instance, key)
        if is_dataclass(current):
            if not isinstance(value, Mapping):
                raise ConfigError(f"{source}: {where!r} is a section; expected a TOML table")
            updates[key] = _apply(current, value, source=source, trail=(*trail, key))
        else:
            updates[key] = _coerce(current, value, where=where, source=source)
    return replace(instance, **updates)  # type: ignore[type-var]


#: Readable names for the types a config value can have, so messages read as
#: English rather than as `expects a int`.
_TYPE_NAMES: dict[type, str] = {
    bool: "a boolean",
    int: "an integer",
    float: "a number",
    str: "a string",
    tuple: "an array",
    list: "an array",
}


def _describe(value: object) -> str:
    return _TYPE_NAMES.get(type(value), f"a {type(value).__name__}")


def _coerce(current: object, value: object, *, where: str, source: Path) -> object:
    """Check a scalar against the default's type, widening int to float.

    TOML has no tuple and no float literal for whole numbers, so a list becomes
    a tuple and `timeout_seconds = 30` is accepted for a float field.
    """

    def reject(expected: str) -> ConfigError:
        return ConfigError(f"{source}: {where!r} expects {expected}, got {_describe(value)}")

    if isinstance(current, tuple):
        if not isinstance(value, list):
            raise reject("an array")
        if not all(isinstance(item, str) for item in value):
            raise reject("an array of strings")
        return tuple(value)
    # bool before int: in Python `bool` is a subclass of `int`, so an
    # unqualified isinstance would let `max_turns = true` through.
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise reject("a boolean")
        return value
    if isinstance(value, bool):
        raise reject(_describe(current))
    if isinstance(current, float):
        if not isinstance(value, int | float):
            raise reject("a number")
        return float(value)
    if not isinstance(value, type(current)):
        raise reject(_describe(current))
    return value
