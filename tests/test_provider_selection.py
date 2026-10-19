"""Which provider serves a model, and how one gets built.

The ladder is the point: `--provider`, then the provider the id's `models`
entry names, then `agent.provider`. Each rung is pinned separately, because a
routing bug reads as "it used the wrong endpoint" long after the fact.
"""

from dataclasses import replace

import pytest

from lab_harness.config import (
    AgentConfig,
    Config,
    ConfigError,
    ModelConfig,
    OpenAIConfig,
    ProvidersConfig,
)
from lab_harness.llm.providers import (
    PROTOCOLS,
    build_protocol,
    build_provider,
    route,
    select_provider,
)
from lab_harness.llm.providers.anthropic import AnthropicProvider
from lab_harness.llm.providers.chat_completions import ChatCompletionsProvider
from lab_harness.llm.providers.responses import ResponsesProvider
from tests.entries import model_entry, profile_entry


def with_profiles(**profiles: OpenAIConfig) -> Config:
    return Config(providers=ProvidersConfig(openai=dict(profiles)))


def with_models(**models: ModelConfig) -> Config:
    """The shipped config, with these entries added to or replacing its own."""
    config = Config()
    return replace(config, models={**config.models, **models})


# --- the ladder -----------------------------------------------------------


def test_the_flag_wins_over_everything() -> None:
    assert select_provider("deepseek-flash", Config(), override="anthropic") == "anthropic"


def test_a_config_entry_can_repoint_a_catalogued_model() -> None:
    """A gateway, or a local server with the same weights, without a code change."""
    config = with_models(**{"claude-sonnet-5": model_entry(provider="local")})

    assert select_provider("claude-sonnet-5", config) == "local"


def test_an_entry_naming_no_provider_falls_through() -> None:
    """An entry may only describe limits or a price, and leave routing alone."""
    config = replace(
        with_models(**{"their-model": model_entry(max_tokens=4_096)}),
        agent=AgentConfig(provider="my-gateway"),
    )

    assert select_provider("their-model", config) == "my-gateway"


def test_an_unknown_id_falls_back_to_the_configured_default() -> None:
    config = replace(Config(), agent=AgentConfig(provider="my-gateway"))

    assert select_provider("something-released-last-week", config) == "my-gateway"


def test_the_shipped_catalog_routes_each_model_to_its_vendor() -> None:
    """The defaults have to actually work, not merely exist."""
    config = Config()

    assert select_provider("claude-fable-5", config) == "anthropic"
    assert select_provider("gpt-6-luna", config) == "openai"
    assert select_provider("gpt-6-astra", config) == "openai"
    assert select_provider("deepseek-flash", config) == "deepseek"
    assert select_provider("deepseek-v4-pro", config) == "deepseek"
    assert select_provider("glm-5.3", config) == "glm"
    assert select_provider("glm-4.6", config) == "glm"


# --- construction ---------------------------------------------------------


def test_an_unknown_provider_fails_at_startup() -> None:
    """A typo in --provider should not become a confusing first-turn failure."""
    with pytest.raises(ConfigError) as caught:
        build_provider("deepsek", Config())

    message = str(caught.value)
    assert "deepsek" in message
    # The message has to list the way out.
    assert "anthropic" in message and "deepseek" in message


# --- route(): the composition the CLI uses --------------------------------


def test_route_returns_a_matched_model_and_provider() -> None:
    model, provider = route("deepseek-flash", Config())

    assert model.id == "deepseek-flash"
    assert isinstance(provider, ChatCompletionsProvider)
    assert provider.name == "deepseek"


def test_route_writes_the_chosen_provider_onto_an_unknown_model() -> None:
    """An unlisted id resolves naming no provider; the Model must say what served it."""
    model, provider = route("something-released-last-week", Config())

    assert model.provider == "anthropic"
    assert isinstance(provider, AnthropicProvider)


def test_route_honors_the_flag() -> None:
    model, provider = route("deepseek-flash", Config(), override="anthropic")

    assert model.provider == "anthropic"
    assert isinstance(provider, AnthropicProvider)


def test_route_falls_back_to_the_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """No --model: agent.model decides, and routing follows it."""
    config = replace(Config(), agent=AgentConfig(model="deepseek-v4-pro"))

    model, provider = route(None, config)

    assert model.id == "deepseek-v4-pro"
    assert provider.name == "deepseek"


def test_route_defaults_to_anthropic_out_of_the_box() -> None:
    model, provider = route(None, Config())

    assert model.provider == "anthropic"
    assert isinstance(provider, AnthropicProvider)


# --- protocol dispatch -----------------------------------------------------
#
# OpenAI moved to the Responses API; everyone else still speaks the dialect
# OpenAI invented. So the protocol is a per-profile setting, not a global one.


def test_the_shipped_openai_profile_speaks_responses() -> None:
    assert isinstance(build_provider("openai", Config()), ResponsesProvider)


def test_the_other_shipped_profiles_speak_chat_completions() -> None:
    for name in ("deepseek", "glm"):
        provider = build_provider(name, Config())
        assert isinstance(provider, ChatCompletionsProvider), name


def test_chat_completions_is_the_default_for_a_new_profile() -> None:
    """A vendor added in config is far more likely to speak the old dialect."""
    config = with_profiles(groq=profile_entry())

    assert isinstance(build_provider("groq", config), ChatCompletionsProvider)


def test_a_profile_can_opt_into_responses() -> None:
    config = with_profiles(gateway=profile_entry(api="responses"))

    assert isinstance(build_provider("gateway", config), ResponsesProvider)


def test_the_profile_name_survives_either_protocol() -> None:
    """It lands in AssistantMessage.provider, so a transcript stays truthful."""
    assert build_protocol(profile_entry(api="responses"), "mine").name == "mine"
    assert build_protocol(profile_entry(api="chat_completions"), "mine").name == "mine"


def test_an_unknown_protocol_fails_at_startup() -> None:
    with pytest.raises(ConfigError) as caught:
        build_protocol(profile_entry(api="grpc"), "mine")

    message = str(caught.value)
    assert "grpc" in message
    assert "mine" in message, "the message must name the profile at fault"
    for known in PROTOCOLS:
        assert known in message, "the message must list the way out"


def test_an_unknown_protocol_in_config_fails_through_build_provider() -> None:
    config = with_profiles(broken=profile_entry(api="telepathy"))

    with pytest.raises(ConfigError, match="telepathy"):
        build_provider("broken", config)
