"""Configuration schema and defaults.

This module is the single list of every tunable in the system. Code reads
values from a resolved `Config`; it never hardcodes them. See
docs/contributing.md.

`load_config()` layers TOML files over these defaults, lowest precedence
first: `<home>/config.toml`, `./.lab-harness/config.toml`, then
`./.lab-harness/config.local.toml` — the last of which is gitignored and is the
only one that may hold a credential.

Environment variables sit above all of them, but there is deliberately no
generic env-overlay machinery here. Each env var is read where it is used
(`LAB_HARNESS_HOME` below, `LAB_HARNESS_MODEL` in llm/registry.py, credentials in the
providers and backends), and its *name* is itself a config key — so a
deployment that uses different names changes config rather than code.

This module is the bottom of the dependency graph and imports nothing from the
rest of the package.
"""

import logging
import os
import tomllib
from collections.abc import Mapping
from dataclasses import Field, asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, get_args

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    #: Hard ceiling on provider requests in one run. Guards against a model
    #: that calls tools forever.
    max_turns: int = 50
    #: Run independent tool calls from one message concurrently.
    parallel_tools: bool = True
    #: Model used when neither `--model` nor the env override names one. An
    #: id absent from the catalog in llm/registry.py still works; it is
    #: assumed current and given conservative limits.
    model: str = "claude-sonnet-5"
    #: Env var that overrides `model`, named here for the same reason
    #: `session.home_env` is — so a deployment using a different variable
    #: changes config rather than code.
    model_env: str = "LAB_HARNESS_MODEL"
    #: Provider used for a model id nothing else routes — not in any profile's
    #: `models` list and not in the catalog. The last rung of the ladder in
    #: llm/providers/select_provider; `--provider` is the first.
    provider: str = "anthropic"


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """What a project tells the agent about itself.

    A repository documents its own conventions in a file at its root; without
    this the agent has to rediscover them from the code every run.
    """

    #: Searched in the working directory in order; the first that exists is
    #: loaded. Both names are existing conventions rather than our invention,
    #: so a project that already has one needs no setup.
    context_files: tuple[str, ...] = ("HARNESS.md", "AGENTS.md")
    #: A longer file is truncated. Project instructions ride in the system
    #: prompt on *every* request, so an unbounded file is an unbounded bill.
    max_bytes: int = 32_000


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
    user_agent: str = "lab-harness/0.1 (+https://github.com/lxucs/lab-harness)"
    tavily: TavilyConfig = field(default_factory=TavilyConfig)
    ddgs: DdgsConfig = field(default_factory=DdgsConfig)


@dataclass(frozen=True, slots=True)
class DiffConfig:
    """The diff `edit` and `write` report beside their result, for a renderer.

    Never sent to the model: it rides in the result's `details`.
    """

    #: Unchanged lines shown around each change. Three is git's default.
    context_lines: int = 3


@dataclass(frozen=True, slots=True)
class ToolsConfig:
    read: ReadConfig = field(default_factory=ReadConfig)
    list: ListConfig = field(default_factory=ListConfig)
    diff: DiffConfig = field(default_factory=DiffConfig)
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
    #: What comes back in a thinking block: "summarized" or "omitted". Opus 5
    #: and Sonnet 5 think whether asked to or not, but default to omitted —
    #: the blocks arrive empty, and a terminal has nothing to show. Billing is
    #: the same either way. Empty sends no preference, for a model that
    #: rejects the field.
    thinking_display: str = "summarized"
    #: Models that take the older thinking request, a token budget, matched by
    #: id prefix so a dated id and its alias agree: everything before 4.6,
    #: which introduced adaptive thinking. Any other reasoning model gets
    #: adaptive, as an uncatalogued id is assumed current — and the 5-series
    #: rejects a budget with a 400.
    budget_thinking_models: tuple[str, ...] = (
        "claude-haiku-4-5",
        "claude-sonnet-4-5",
        "claude-opus-4-5",
        "claude-opus-4-1",
        "claude-opus-4-0",
        "claude-opus-4-2025",
        "claude-sonnet-4-0",
        "claude-sonnet-4-2025",
        "claude-3",
    )
    #: Budget for those models, in tokens, when reasoning is requested. They
    #: do not think otherwise.
    thinking_budget_tokens: int = 4_096


@dataclass(frozen=True, slots=True)
class OpenAIConfig:
    """One endpoint in the OpenAI family, and which protocol it speaks.

    Two protocols, because OpenAI moved and the rest of the world did not:
    `responses` is OpenAI's current API, and the only way to carry an OpenAI
    reasoning model's chain of thought across a tool call; `chat_completions`
    is what DeepSeek, GLM, Groq, OpenRouter, vLLM and Ollama speak.
    """

    #: "responses" or "chat_completions". Defaults to the latter because it is
    #: what every non-OpenAI endpoint speaks; the shipped `openai` profile
    #: overrides it.
    api: str = "chat_completions"
    base_url: str = "https://api.openai.com/v1"
    #: Env var consulted first for the key.
    api_key_env: str = "OPENAI_API_KEY"
    #: Env var that overrides `base_url`. Useful for a gateway in front of a
    #: vendor, and for pointing a profile at a local server.
    base_url_env: str = "OPENAI_BASE_URL"
    #: Empty means "not configured here"; look in the environment instead.
    #: Only ever set this in config.local.toml, never a committed file.
    api_key: str = ""
    #: Model ids this profile serves. An id listed here routes to this profile,
    #: which is how `--model deepseek-flash` finds DeepSeek with no flag and how
    #: a new vendor needs no code — only config.
    models: tuple[str, ...] = ()
    #: Sampling ceiling when the model declares none.
    default_max_tokens: int = 8_192
    #: Wall-clock limit for one streamed request. Generous next to the web
    #: tools': a long answer with reasoning legitimately takes minutes.
    timeout_seconds: float = 120.0
    #: Ask for token counts in the stream. Not every compatible server
    #: implements `stream_options`; turn this off for one that rejects it.
    #: `chat_completions` only — Responses always reports usage.
    include_usage: bool = True
    #: `chat_completions` only. Field each earlier assistant turn's reasoning
    #: is sent back in; empty sends none. DeepSeek returns a 400 for a request
    #: with tools unless every earlier turn's `reasoning_content` comes back,
    #: and GLM asks for it unmodified. Off for a new profile, because a server
    #: that never asked for the field may reject it. Responses replays its own
    #: encrypted reasoning regardless of this.
    replay_reasoning_field: str = ""
    #: `chat_completions` only. Ask the server to keep earlier turns' reasoning
    #: in context rather than discarding it — GLM's Preserved Thinking, which
    #: is off on its standard endpoint. Replaying reasoning the server then
    #: throws away buys nothing, so the shipped `glm` profile turns this on.
    #: It sends GLM's own request shape, which another vendor would read
    #: differently or reject; and its `type: "enabled"` switches thinking on
    #: for GLM-4.6, the one hybrid model left in the lineup.
    preserve_thinking: bool = False
    #: `responses` only. Detail of the reasoning summary: "auto", "concise" or
    #: "detailed". Empty asks for none, which makes a reasoning model's
    #: thinking invisible — the raw chain is never returned, so the summary is
    #: the only readable form of it.
    reasoning_summary: str = "auto"
    #: `responses` only. Let the server keep the conversation and refer to it
    #: by id. Off, because this harness owns its transcript: a thread living on
    #: someone else's server is one `--resume` and the session files cannot
    #: see. Turning it on disables encrypted-reasoning replay.
    store: bool = False


def _default_openai_profiles() -> dict[str, OpenAIConfig]:
    """Endpoints shipped as defaults. All public, documented vendor URLs.

    A profile here is a starting point, not a restriction — adding a key to
    `[providers.openai]` in a config file creates a new one.
    """
    return {
        "openai": OpenAIConfig(api="responses"),
        "deepseek": OpenAIConfig(
            base_url="https://api.deepseek.com/v1",
            api_key_env="DEEPSEEK_API_KEY",
            base_url_env="DEEPSEEK_BASE_URL",
            models=("deepseek-flash", "deepseek-v4-pro"),
            replay_reasoning_field="reasoning_content",
        ),
        "glm": OpenAIConfig(
            # The mainland endpoint. The international one is
            # https://api.z.ai/api/paas/v4 — override base_url to use it.
            base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key_env="ZHIPUAI_API_KEY",
            base_url_env="ZHIPUAI_BASE_URL",
            # Z.ai's current lineup (2026-10), flagship first.
            models=(
                "glm-5.3",
                "glm-5.3-flash",
                "glm-5.3-flashx",
                "glm-5.2",
                "glm-5.1",
                "glm-5",
                "glm-4.7",
                "glm-4.6",
            ),
            replay_reasoning_field="reasoning_content",
            preserve_thinking=True,
        ),
    }


@dataclass(frozen=True, slots=True)
class ProvidersConfig:
    anthropic: AnthropicConfig = field(default_factory=AnthropicConfig)
    #: OpenAI-compatible endpoints, keyed by profile name. Like `pricing`, and
    #: unlike every other section, *new keys here are allowed* — that is the
    #: whole point, since it is how a vendor is added without touching code.
    #: Keys *inside* a profile are still checked.
    openai: dict[str, OpenAIConfig] = field(default_factory=_default_openai_profiles)


@dataclass(frozen=True, slots=True)
class PricingConfig:
    """What one model costs, per million tokens.

    The four rates mirror the four counts in `Usage`, so a request's cost is
    each count times its rate. A rate left unset is zero. Rates are in whatever
    currency you quote them in — `terminal.currency_symbol` is only a label —
    but it has to be one currency for every model, since a session that
    switches models adds the costs together.
    """

    input: float = 0.0
    output: float = 0.0
    #: Prompt tokens served from the cache — typically a tenth of `input`.
    cache_read: float = 0.0
    #: Prompt tokens written to the cache. Anthropic charges a premium for
    #: these; providers that cache implicitly report no writes, so this rate
    #: never applies to them.
    cache_write: float = 0.0


def _default_pricing() -> dict[str, PricingConfig]:
    """USD list prices: the catalogued Claude models, and every id a shipped
    profile lists in its `models`.

    Prices change — override an entry in config rather than trusting these.
    Every other model starts unpriced and shows no cost.
    """
    return {
        # Anthropic's first-party API rates as published mid-2026, with
        # `cache_write` at the 5-minute-TTL premium (1.25x input). Bedrock and
        # Vertex bill separately.
        "claude-opus-5": PricingConfig(input=5.0, output=25.0, cache_read=0.5, cache_write=6.25),
        "claude-sonnet-5": PricingConfig(input=2.0, output=10.0, cache_read=0.2, cache_write=2.5),
        "claude-haiku-4-5-20251001": PricingConfig(
            input=1.0, output=5.0, cache_read=0.1, cache_write=1.25
        ),
        # DeepSeek's PEAK rates (2026-10), which apply 01:00-04:00 and
        # 06:00-10:00 UTC on weekdays other than Chinese public holidays.
        # Every other hour costs half. One rate
        # per model cannot follow the clock, so this takes the higher one: a
        # shown cost may run high, never low. `input` is the cache-miss rate,
        # and there is no cache_write — DeepSeek charges nothing to fill it.
        "deepseek-flash": PricingConfig(input=0.3, output=1.2, cache_read=0.006),
        "deepseek-v4-pro": PricingConfig(input=1.32, output=3.96, cache_read=0.044),
        # Z.ai's international list rates (2026-10). The shipped `glm` profile
        # points at the mainland endpoint, which bills in CNY at its own rates
        # — override these if that is the one you use. Z.ai's cache storage is
        # "limited-time free", so there is no cache_write.
        "glm-5.3": PricingConfig(input=1.4, output=4.4, cache_read=0.26),
        "glm-5.3-flash": PricingConfig(input=0.15, output=0.5, cache_read=0.03),
        "glm-5.3-flashx": PricingConfig(input=0.37, output=1.25, cache_read=0.075),
        "glm-5.2": PricingConfig(input=1.4, output=4.4, cache_read=0.26),
        "glm-5.1": PricingConfig(input=1.4, output=4.4, cache_read=0.26),
        "glm-5": PricingConfig(input=1.0, output=3.2, cache_read=0.2),
        "glm-4.7": PricingConfig(input=0.6, output=2.2, cache_read=0.11),
        "glm-4.6": PricingConfig(input=0.6, output=2.2, cache_read=0.11),
    }


@dataclass(frozen=True, slots=True)
class SessionConfig:
    #: Root for all harness state. `~` is expanded at use.
    home: str = "~/.lab-harness"
    #: Env var that overrides `home`. The value is a machine path, so it lives
    #: in the environment rather than in a committed config file.
    home_env: str = "LAB_HARNESS_HOME"
    #: Subdirectory of `home` holding per-project session directories.
    directory_name: str = "sessions"
    #: Write transcripts to disk at all. Off means nothing is persisted.
    enabled: bool = True
    #: Schema version written into each file's header. Readers use it to refuse
    #: files they do not understand. A genuine protocol constant, not a knob.
    #: Version 2 added the `context` entry; version 1 files still read.
    version: int = 2


@dataclass(frozen=True, slots=True)
class ThemeConfig:
    """Styles for the interactive terminal, as rich style strings.

    Named by role rather than by color, so that "what does a failed tool look
    like" is one setting. Each is checked at startup — a typo fails there, not
    on the first message that uses it. The defaults follow Claude Code.
    """

    #: The spinner and other highlights. Claude Code's orange.
    accent: str = "#d77757"
    #: The input prompt's glyph. prompt_toolkit parses this one, not rich, so
    #: it takes prompt_toolkit's syntax — hex colors and bold/italic/underline
    #: work in both; rich's `rgb(...)` does not.
    prompt: str = "bold #d77757"
    #: Secondary text: tool arguments and output, notices, the usage line.
    muted: str = "dim"
    #: Reasoning, streamed or expanded.
    thinking: str = "italic dim"
    #: The tool's name in a call line.
    tool: str = "bold"
    #: The bullet of a tool call that finished cleanly.
    success: str = "green"
    #: Failed tool calls and errors.
    error: str = "red"
    #: Cancellations and warnings.
    warning: str = "yellow"
    #: Whole lines of a diff, added and removed. A foreground as well as the
    #: background, so they read the same on a light terminal as a dark one.
    diff_added: str = "#ffffff on #225c2b"
    diff_removed: str = "#ffffff on #7a2936"


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
    #: Lines of an edit's diff shown before collapsing. More than plain
    #: output gets, since the diff is the part of an edit worth reading.
    max_diff_lines: int = 40
    #: Render assistant text as markdown rather than plain text.
    markdown: bool = True
    #: What a model's reasoning leaves in scrollback once its message ends:
    #: "full", "collapse" to one line, or "hidden". The first two also show it
    #: as it streams; hidden shows only the spinner. Ctrl-O switches between
    #: full and this for the rest of the session.
    thinking: str = "collapse"
    #: Lines of reasoning shown while it streams, when it is going to
    #: collapse. "full" shows as much as fits.
    thinking_preview_lines: int = 4
    #: Print one line of token, cache and cost figures after each run.
    #: `:status` shows the session's totals either way.
    show_usage: bool = True
    #: Label for amounts computed from `pricing`. Only a label — nothing is
    #: converted, so it has to match the currency the rates are quoted in.
    currency_symbol: str = "$"
    theme: ThemeConfig = field(default_factory=ThemeConfig)


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Diagnostics about the harness itself — not about the conversation.

    Records go to stderr. stdout is reserved: plain-text mode writes the
    answer there and `--json` writes the event stream there, so a log line on
    stdout would corrupt a pipeline rather than inform one.
    """

    #: Lowest level printed; one of the names in `lab_harness/logs.py`. Warning by
    #: default, because a run that works should say nothing beyond its answer.
    level: str = "warning"
    #: Env var that overrides `level`, named here for the same reason
    #: `agent.model_env` is — so a deployment using a different variable
    #: changes config rather than code.
    level_env: str = "LAB_HARNESS_LOG_LEVEL"
    #: Format for one record. The logger name is included because the first
    #: question about a stray line is which module emitted it.
    format: str = "%(levelname)s %(name)s: %(message)s"


@dataclass(frozen=True, slots=True)
class Config:
    agent: AgentConfig = field(default_factory=AgentConfig)
    project: ProjectConfig = field(default_factory=ProjectConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    #: Per-million-token rates, keyed by model id. Like `providers.openai`, a
    #: new key is allowed — it prices one more model. A model with no entry
    #: shows no cost; an entry with every rate at zero counts as no entry,
    #: which is how a config file switches off a shipped one.
    pricing: dict[str, PricingConfig] = field(default_factory=_default_pricing)
    terminal: TerminalConfig = field(default_factory=TerminalConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


class ConfigError(Exception):
    """A config file was unreadable or contained something unusable."""


#: Field names that hold a credential. `snapshot` replaces their values, which
#: is what keeps a key out of a session transcript. A test fails when a field
#: named like a credential is missing from this set.
SECRET_FIELDS = frozenset({"api_key"})
#: What a credential that was set reads as in a snapshot. One that was not set
#: stays empty, so the record still says which were configured.
REDACTED = "[redacted]"


def snapshot(config: Config) -> dict[str, Any]:
    """The config as plain, JSON-safe data, with every credential replaced.

    For recording what a run was configured with. Credentials read from the
    environment never reach a `Config` at all; this covers the ones a
    config.local.toml supplied.
    """
    return _redact(asdict(config))


def _redact(data: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            redacted[key] = _redact(value)
        elif key in SECRET_FIELDS and value:
            redacted[key] = REDACTED
        elif isinstance(value, tuple):
            # As JSON reads it back, so a snapshot equals its own record.
            redacted[key] = list(value)
        else:
            redacted[key] = value
    return redacted


#: Per-project config directory, relative to the working directory.
PROJECT_DIRECTORY = ".lab-harness"
CONFIG_FILE = "config.toml"
#: Gitignored sibling of CONFIG_FILE. The only file that may hold a credential.
LOCAL_CONFIG_FILE = "config.local.toml"


def lab_harness_home(config: SessionConfig | None = None) -> Path:
    """The harness home directory, honoring the configured env override."""
    config = config or SessionConfig()
    override = os.environ.get(config.home_env)
    return Path(override).expanduser() if override else Path(config.home).expanduser()


def config_paths(cwd: str | Path = ".") -> tuple[Path, ...]:
    """The files `load_config` reads, in increasing order of precedence.

    Home is located from the *default* session config rather than from a
    loaded one. Resolving where to look for config using config that has not
    been loaded yet is circular; `LAB_HARNESS_HOME` is the escape hatch.
    """
    project = Path(cwd) / PROJECT_DIRECTORY
    return (
        lab_harness_home() / CONFIG_FILE,
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
        if data is None:
            _log.debug("No config file at %s", path)
            continue
        # Answers "which file is overriding my setting" without the user
        # having to reconstruct the precedence rules by hand.
        _log.debug("Applying config from %s", path)
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
        elif isinstance(current, dict):
            updates[key] = _apply_named(
                current, value, member=_member_type(known[key]), source=source, trail=(*trail, key)
            )
        else:
            updates[key] = _coerce(current, value, where=where, source=source)
    return replace(instance, **updates)  # type: ignore[type-var]


def _member_type(spec: Field[Any]) -> type:
    """The value type of a `dict[str, X]` field, as X.

    Read from the annotation rather than from an existing entry, so a profile
    table someone emptied still validates its replacements.
    """
    args = get_args(spec.type)
    # isinstance(..., type) as well as is_dataclass: the latter also accepts an
    # *instance* of a dataclass, and we need the class to construct with.
    if len(args) != 2 or not isinstance(args[1], type) or not is_dataclass(args[1]):
        raise TypeError(f"{spec.name}: expected dict[str, <dataclass>], got {spec.type!r}")
    return args[1]


def _apply_named(
    current: Mapping[str, Any],
    data: object,
    *,
    member: type,
    source: Path,
    trail: tuple[str, ...],
) -> dict[str, Any]:
    """Merge a table of *named* sections, e.g. `[providers.openai.deepseek]`.

    The only tables where an unknown key is not an error. Everywhere else a
    name the schema does not know is a typo; here it is a new entry — a vendor
    profile, or a model's price — which is how a deployment adds one without
    code. Keys *inside* a section are checked as strictly as anywhere else.
    """
    where = ".".join(trail)
    if not isinstance(data, Mapping):
        raise ConfigError(f"{source}: {where!r} is a table of named sections; expected TOML tables")
    merged = dict(current)
    for name, section in data.items():
        if not isinstance(section, Mapping):
            raise ConfigError(
                f"{source}: {where}.{name!r} is a section; expected a TOML table. "
                f"A setting shared by every entry in {where!r} does not exist — "
                f"name one explicitly."
            )
        base = merged.get(name)
        if base is None:
            base = member()
        merged[name] = _apply(base, section, source=source, trail=(*trail, name))
    return merged


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
