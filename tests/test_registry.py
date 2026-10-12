"""Model catalog and resolution.

The precedence chain is the whole point: a CLI flag beats the environment,
which beats config. Getting it backwards is invisible until someone is billed
for the wrong model, so each link is asserted on its own and then together.
"""

import pytest

from harness.config import AgentConfig
from harness.llm.registry import MODELS, resolve_model


def test_a_catalog_id_resolves_to_its_entry() -> None:
    model = resolve_model("claude-opus-5")
    assert model is MODELS["claude-opus-5"]
    assert model.context_window == MODELS["claude-opus-5"].context_window


def test_the_catalog_entries_are_self_consistent() -> None:
    """A key that disagrees with its own `id` would break resolution silently."""
    for key, model in MODELS.items():
        assert model.id == key


# --- precedence -----------------------------------------------------------


def test_nothing_specified_uses_the_configured_model() -> None:
    assert resolve_model().id == AgentConfig().model


def test_the_configured_model_is_honored() -> None:
    config = AgentConfig(model="claude-opus-5")
    assert resolve_model(config=config).id == "claude-opus-5"


def test_the_environment_beats_config(monkeypatch: pytest.MonkeyPatch) -> None:
    config = AgentConfig(model="claude-opus-5")
    monkeypatch.setenv(config.model_env, "claude-haiku-4-5-20251001")
    assert resolve_model(config=config).id == "claude-haiku-4-5-20251001"


def test_the_explicit_argument_beats_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    config = AgentConfig(model="claude-opus-5")
    monkeypatch.setenv(config.model_env, "claude-haiku-4-5-20251001")
    assert resolve_model("claude-sonnet-5", config).id == "claude-sonnet-5"


def test_the_env_var_name_comes_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The *name* is a config key, so a deployment renames it without a patch."""
    config = AgentConfig(model_env="DEPLOYMENT_MODEL")
    monkeypatch.setenv("DEPLOYMENT_MODEL", "claude-opus-5")
    monkeypatch.setenv("HARNESS_MODEL", "claude-haiku-4-5-20251001")
    assert resolve_model(config=config).id == "claude-opus-5"


def test_an_empty_env_var_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """`HARNESS_MODEL=` is an unset variable, not a request for a nameless model."""
    config = AgentConfig(model="claude-opus-5")
    monkeypatch.setenv(config.model_env, "")
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


def test_an_unknown_configured_model_still_resolves() -> None:
    """A config file naming next month's model must not break startup."""
    config = AgentConfig(model="claude-something-7")
    assert resolve_model(config=config).id == "claude-something-7"


def test_an_unknown_id_does_not_mutate_the_catalog() -> None:
    """The template is cloned, not edited — `replace` on a frozen dataclass."""
    before = dict(MODELS)
    resolve_model("claude-something-7")
    assert before == MODELS
    assert "claude-something-7" not in MODELS
