"""Model catalog and resolution.

The precedence chain is the whole point: a CLI flag beats the environment,
which beats config. Getting it backwards is invisible until someone is billed
for the wrong model, so each link is asserted on its own and then together.
"""

from dataclasses import replace
from pathlib import Path

import pytest

from lab_harness import config as config_module
from lab_harness.config import (
    AgentConfig,
    AnthropicConfig,
    Config,
    ConfigError,
    ModelConfig,
    OpenAIConfig,
    _catalog,
    _defaults,
    model_defaults,
)
from lab_harness.llm.providers import ANTHROPIC
from lab_harness.llm.registry import resolve_model
from tests.entries import model_entry


def agent(**settings: str) -> Config:
    return Config(agent=replace(AgentConfig(), **settings))


def test_a_catalog_id_resolves_to_its_entry() -> None:
    entry = Config().models["claude-opus-5"]
    model = resolve_model("claude-opus-5")
    assert (model.id, model.name, model.provider) == ("claude-opus-5", entry.name, entry.provider)
    assert (model.context_window, model.max_tokens) == (entry.context_window, entry.max_tokens)
    assert model.supports_reasoning is entry.supports_reasoning


def test_a_configured_entry_is_what_resolves() -> None:
    """The catalog is config: an entry a config file changed is the one used."""
    config = Config()
    models = {**config.models, "claude-opus-5": model_entry(max_tokens=1_000)}
    assert resolve_model("claude-opus-5", replace(config, models=models)).max_tokens == 1_000


# --- the shipped catalog --------------------------------------------------


def test_every_catalogued_model_names_a_provider_that_exists() -> None:
    """A typo in `provider` would route the model to an endpoint that is not there."""
    config = Config()
    known = {ANTHROPIC, *config.providers.openai}
    for model_id, entry in config.models.items():
        assert entry.provider in known, model_id


def test_every_catalogued_model_has_a_name_and_a_price() -> None:
    """So the terminal labels it, and `:status` shows a cost for it."""
    for model_id, entry in Config().models.items():
        assert entry.name, model_id
        assert entry.price.input and entry.price.output, model_id


def test_the_catalog_may_only_hold_llm_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """It is read like a config file, so a stray section fails rather than vanishing."""
    stray = tmp_path / "catalog.toml"
    stray.write_text("[agent]\nmax_turns = 3\n")
    monkeypatch.setattr(config_module, "CATALOG_FILE", stray)

    with pytest.raises(ConfigError, match=r"catalog\.toml: unknown setting 'agent'"):
        _catalog.__wrapped__()


def test_a_missing_catalog_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config_module, "CATALOG_FILE", tmp_path / "catalog.toml")

    with pytest.raises(ConfigError, match="package is incomplete"):
        _catalog.__wrapped__()


def shipped_without(line: str) -> str:
    """The shipped catalog with one line taken out."""
    shipped = config_module.CATALOG_FILE.read_text()
    assert line in shipped, "the line this test removes must exist"
    return shipped.replace(line, "", 1)


def test_an_llm_class_has_no_values_of_its_own() -> None:
    """Every value ships in the catalog. A bare one would be quietly half-set."""
    for cls in (AnthropicConfig, OpenAIConfig, ModelConfig):
        with pytest.raises(TypeError):
            cls()  # type: ignore[call-arg]


def test_a_defaults_table_missing_a_key_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nothing else supplies a key a defaults table leaves out."""
    catalog = tmp_path / "catalog.toml"
    catalog.write_text(shipped_without("timeout_seconds = 120.0\n"))
    monkeypatch.setattr(config_module, "CATALOG_FILE", catalog)

    with pytest.raises(
        ConfigError, match=r"'defaults\.profile' must set timeout_seconds.*package is incomplete"
    ):
        _defaults.__wrapped__()


def test_anthropic_missing_a_key_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    catalog = tmp_path / "catalog.toml"
    catalog.write_text(shipped_without("thinking_budget_tokens = 4_096\n"))
    monkeypatch.setattr(config_module, "CATALOG_FILE", catalog)

    with pytest.raises(
        ConfigError, match=r"'providers\.anthropic' must set thinking_budget_tokens"
    ):
        _catalog.__wrapped__()


# --- precedence -----------------------------------------------------------


def test_nothing_specified_uses_the_configured_model() -> None:
    assert resolve_model().id == AgentConfig().model


def test_the_configured_model_is_honored() -> None:
    config = agent(model="claude-opus-5")
    assert resolve_model(config=config).id == "claude-opus-5"


def test_the_environment_beats_config(monkeypatch: pytest.MonkeyPatch) -> None:
    config = agent(model="claude-opus-5")
    monkeypatch.setenv(config.agent.model_env, "claude-haiku-4-5-20251001")
    assert resolve_model(config=config).id == "claude-haiku-4-5-20251001"


def test_the_explicit_argument_beats_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    config = agent(model="claude-opus-5")
    monkeypatch.setenv(config.agent.model_env, "claude-haiku-4-5-20251001")
    assert resolve_model("claude-sonnet-5", config).id == "claude-sonnet-5"


def test_the_env_var_name_comes_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The *name* is a config key, so a deployment renames it without a patch."""
    config = agent(model_env="DEPLOYMENT_MODEL")
    monkeypatch.setenv("DEPLOYMENT_MODEL", "claude-opus-5")
    monkeypatch.setenv("LAB_HARNESS_MODEL", "claude-haiku-4-5-20251001")
    assert resolve_model(config=config).id == "claude-opus-5"


def test_an_empty_env_var_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """`LAB_HARNESS_MODEL=` is an unset variable, not a request for a nameless model."""
    config = agent(model="claude-opus-5")
    monkeypatch.setenv(config.agent.model_env, "")
    assert resolve_model(config=config).id == "claude-opus-5"


# --- ids the catalog has never heard of -----------------------------------


def test_an_unknown_id_is_not_an_error() -> None:
    """Model names ship faster than this catalog does."""
    model = resolve_model("claude-something-7")
    assert model.id == "claude-something-7"
    assert model.name == "claude-something-7"


def test_an_unknown_id_gets_conservative_limits() -> None:
    """Guessing high would fail the request; guessing low merely wastes room."""
    model = resolve_model("claude-something-7")
    assert model.context_window == 200_000
    assert model.max_tokens == 8_192


def test_an_unknown_id_gets_the_catalogs_model_defaults() -> None:
    model, defaults = resolve_model("claude-something-7"), model_defaults()
    assert (model.context_window, model.max_tokens) == (
        defaults.context_window,
        defaults.max_tokens,
    )
    assert model.supports_reasoning is defaults.supports_reasoning


def test_an_unknown_configured_model_still_resolves() -> None:
    """A config file naming next month's model must not break startup."""
    config = agent(model="claude-something-7")
    assert resolve_model(config=config).id == "claude-something-7"


def test_an_unknown_id_does_not_mutate_the_catalog() -> None:
    """Resolving reads the catalog; it never adds the guess to it."""
    before = Config().models
    resolve_model("claude-something-7")
    assert Config().models == before
    assert "claude-something-7" not in before


def test_an_unknown_id_names_no_provider() -> None:
    """Routing decides; the catalog has nothing to say about an id it lacks."""
    assert resolve_model("claude-something-7").provider == ""
