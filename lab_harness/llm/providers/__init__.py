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
from lab_harness.llm.registry import MODELS, Model, resolve_model

_log = logging.getLogger(__name__)

#: The one provider name that is not an OpenAI-family profile.
ANTHROPIC = "anthropic"

#: `api` value -> adapter. A profile names one of these keys.
PROTOCOLS = {
    "chat_completions": ChatCompletionsProvider,
    "responses": ResponsesProvider,
}


def select_provider(model_id: str, config: Config, override: str | None = None) -> str:
    """Which provider serves `model_id`.

    Precedence, highest first:

    1. `override` — the `--provider` flag
    2. a profile whose `models` list names this id
    3. the catalog's `Model.provider`
    4. `agent.provider`

    Rung 2 is what makes `--model deepseek-flash` reach DeepSeek with no flag,
    and it is checked before the catalog so that a config file can re-point a
    model id at a different endpoint — a gateway, or a local server serving the
    same weights — without the catalog having to agree.
    """
    if override:
        _log.info("Provider %s for %r: set by --provider", override, model_id)
        return override
    for name, profile in config.providers.openai.items():
        if model_id in profile.models:
            _log.info("Provider %s for %r: listed in providers.openai.%s", name, model_id, name)
            return name
    known = MODELS.get(model_id)
    if known is not None:
        _log.info("Provider %s for %r: from the model catalog", known.provider, model_id)
        return known.provider
    fallback = config.agent.provider
    _log.info("Provider %s for %r: agent.provider, as nothing else claims it", fallback, model_id)
    return fallback


def build_provider(name: str, config: Config) -> Provider:
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


def build_protocol(profile: OpenAIConfig, name: str) -> Provider:
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

    That rewrite matters. An id the catalog has never seen is cloned from an
    Anthropic entry for its limits, so it arrives here claiming
    `provider="anthropic"` no matter which endpoint will actually serve it.
    """
    model = resolve_model(model_id, config.agent)
    name = select_provider(model.id, config, override)
    return replace(model, provider=name), build_provider(name, config)


__all__ = [
    "ANTHROPIC",
    "PROTOCOLS",
    "AnthropicProvider",
    "ChatCompletionsProvider",
    "ResponsesProvider",
    "build_protocol",
    "build_provider",
    "route",
    "select_provider",
]
