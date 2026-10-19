"""Concrete provider adapters, and the routing that picks one.

Three protocols cover every vendor the harness supports:

| Module | Protocol | Reaches |
|---|---|---|
| `anthropic` | Messages | Claude |
| `responses` | OpenAI Responses | OpenAI |
| `chat_completions` | Chat Completions | DeepSeek, GLM, Groq, OpenRouter, vLLM, Ollama |

Named for the protocol rather than the vendor, because the mapping is not
one-to-one: OpenAI has moved to Responses while everyone else still speaks the
dialect OpenAI invented. A new vendor is normally a profile in
`providers.openai`, not a module.
"""

import logging
from dataclasses import replace

from lab_harness.config import Config, ConfigError, OpenAIConfig
from lab_harness.llm.provider import Provider
from lab_harness.llm.providers.anthropic import AnthropicProvider
from lab_harness.llm.providers.chat_completions import ChatCompletionsProvider
from lab_harness.llm.providers.responses import ResponsesProvider
from lab_harness.llm.registry import resolve_model
from lab_harness.llm.types import Model

_log = logging.getLogger(__name__)

#: The one provider name that is not an OpenAI-family profile.
ANTHROPIC = "anthropic"

#: Every adapter shipped. Beyond `Provider`, each can say what it sends:
#: `request_body` is the JSON a request posts, and `reply_entries` a reply as
#: its dialect writes it into the next request's history.
Adapter = AnthropicProvider | ChatCompletionsProvider | ResponsesProvider

#: `api` value -> adapter. A profile names one of these keys.
PROTOCOLS: dict[str, type[ChatCompletionsProvider] | type[ResponsesProvider]] = {
    "chat_completions": ChatCompletionsProvider,
    "responses": ResponsesProvider,
}


def select_provider(model_id: str, config: Config, override: str | None = None) -> str:
    """Which provider serves `model_id`.

    Precedence, highest first:

    1. `override` — the `--provider` flag
    2. the `provider` of the id's `models` entry
    3. `agent.provider`

    Rung 2 is what makes `--model deepseek-flash` reach DeepSeek with no flag.
    It is config like any other, so a config file can re-point a catalogued id
    at a different endpoint — a gateway, or a local server serving the same
    weights — with `[models.<id>] provider = "<profile>"`.
    """
    if override:
        _log.info("Provider %s for %r: set by --provider", override, model_id)
        return override
    entry = config.models.get(model_id)
    if entry is not None and entry.provider:
        _log.info("Provider %s for %r: from its models entry", entry.provider, model_id)
        return entry.provider
    fallback = config.agent.provider
    _log.info("Provider %s for %r: agent.provider, as nothing else claims it", fallback, model_id)
    return fallback


def build_provider(name: str, config: Config) -> Adapter:
    """Construct the named provider.

    Raises `ConfigError` for a name that is neither `anthropic` nor a
    configured profile — a typo in `--provider` should fail at startup rather
    than as a confusing request failure on the first turn.
    """
    if name == ANTHROPIC:
        return AnthropicProvider(config.providers.anthropic)
    profile = config.providers.openai.get(name)
    if profile is None:
        available = ", ".join([ANTHROPIC, *sorted(config.providers.openai)])
        raise ConfigError(f"Unknown provider {name!r}. Available: {available}.")
    return build_protocol(profile, name)


def build_protocol(profile: OpenAIConfig, name: str) -> Adapter:
    """The adapter for a profile's `api`, or a `ConfigError` naming the options.

    Checked here rather than in the adapter so a typo fails at startup, like
    every other unusable config value.
    """
    adapter = PROTOCOLS.get(profile.api)
    if adapter is None:
        available = ", ".join(sorted(PROTOCOLS))
        raise ConfigError(
            f"providers.openai.{name}.api is {profile.api!r}. Available: {available}."
        )
    return adapter(profile, name)


def route(
    model_id: str | None, config: Config, override: str | None = None
) -> tuple[Model, Provider]:
    """Resolve a model id, decide which provider serves it, and build that.

    One call rather than three at the wiring site, because the three belong
    together: the id has to be resolved before routing can look at it, and the
    resolved `Model` carries a provider name that routing may well change.

    That rewrite matters: an id the catalog has never seen arrives with no
    provider at all, and `--provider` can override the one a catalogued id has.
    """
    model = resolve_model(model_id, config)
    name = select_provider(model.id, config, override)
    return replace(model, provider=name), build_provider(name, config)


__all__ = [
    "ANTHROPIC",
    "PROTOCOLS",
    "Adapter",
    "AnthropicProvider",
    "ChatCompletionsProvider",
    "ResponsesProvider",
    "build_protocol",
    "build_provider",
    "route",
    "select_provider",
]
