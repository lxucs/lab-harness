"""Layered TOML config loading.

The interesting behavior is not "a value is read" but the three things that
make config trustworthy: later files win, a typo is an error rather than a
silent no-op, and a wrong type is caught at startup rather than at the call
site hours later.
"""

import json
import re
from dataclasses import is_dataclass
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

import pytest

from lab_harness.config import (
    CONFIG_FILE,
    LOCAL_CONFIG_FILE,
    PROJECT_DIRECTORY,
    REDACTED,
    SECRET_FIELDS,
    Config,
    ConfigError,
    config_paths,
    lab_harness_home,
    load_config,
    model_defaults,
    profile_defaults,
    snapshot,
)

#: Field names that hold a credential rather than, say, an env var's name
#: (`api_key_env`) or a count (`max_tokens`).
_CREDENTIAL_NAME = re.compile(r"(^|_)(key|token|secret|password)$")


def write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def project(tmp_path: Path, body: str, *, name: str = CONFIG_FILE) -> Path:
    return write(tmp_path / PROJECT_DIRECTORY / name, body)


# --- the baseline ---------------------------------------------------------


def test_no_files_means_defaults(tmp_path: Path) -> None:
    assert load_config(tmp_path) == Config()


def test_an_absent_file_is_not_an_error(tmp_path: Path) -> None:
    """Most projects have no config at all; that is the normal case."""
    for path in config_paths(tmp_path):
        assert not path.exists()
    assert load_config(tmp_path).agent.max_turns == 50


# --- reading values -------------------------------------------------------


def test_a_scalar_is_applied(tmp_path: Path) -> None:
    project(tmp_path, "[agent]\nmax_turns = 7\n")
    assert load_config(tmp_path).agent.max_turns == 7


def test_a_nested_section_is_applied(tmp_path: Path) -> None:
    project(tmp_path, '[tools.web]\nbackend = "ddgs"\nmax_results = 3\n')
    web = load_config(tmp_path).tools.web
    assert (web.backend, web.max_results) == ("ddgs", 3)


def test_a_doubly_nested_section_is_applied(tmp_path: Path) -> None:
    project(tmp_path, '[tools.web.tavily]\nsearch_depth = "advanced"\n')
    assert load_config(tmp_path).tools.web.tavily.search_depth == "advanced"


def test_unset_keys_keep_their_defaults(tmp_path: Path) -> None:
    """A partial file is a partial override, not a replacement."""
    project(tmp_path, "[agent]\nmax_turns = 7\n")
    config = load_config(tmp_path)
    assert config.agent.max_turns == 7
    assert config.agent.parallel_tools is Config().agent.parallel_tools
    assert config.tools.read.max_bytes == Config().tools.read.max_bytes


def test_an_array_becomes_a_tuple(tmp_path: Path) -> None:
    """Config types are tuples so a Config stays immutable and hashable."""
    project(tmp_path, '[tools.approval]\nrequired = ["bash"]\n')
    assert load_config(tmp_path).tools.approval.required == ("bash",)


def test_an_integer_is_accepted_for_a_float(tmp_path: Path) -> None:
    """TOML has no float literal for a whole number, so `30` must work."""
    project(tmp_path, "[tools.web]\ntimeout_seconds = 30\n")
    value = load_config(tmp_path).tools.web.timeout_seconds
    assert value == 30.0 and isinstance(value, float)


# --- precedence -----------------------------------------------------------


def test_the_project_file_overrides_the_home_file(tmp_path: Path) -> None:
    write(lab_harness_home() / CONFIG_FILE, "[agent]\nmax_turns = 1\n")
    project(tmp_path, "[agent]\nmax_turns = 2\n")
    assert load_config(tmp_path).agent.max_turns == 2


def test_the_local_file_overrides_the_project_file(tmp_path: Path) -> None:
    """config.local.toml is last so a machine can override a committed value."""
    project(tmp_path, "[agent]\nmax_turns = 2\n")
    project(tmp_path, "[agent]\nmax_turns = 3\n", name=LOCAL_CONFIG_FILE)
    assert load_config(tmp_path).agent.max_turns == 3


def test_layers_merge_rather_than_replace(tmp_path: Path) -> None:
    """A later file overrides the keys it names and leaves the rest alone."""
    write(lab_harness_home() / CONFIG_FILE, "[agent]\nmax_turns = 1\nparallel_tools = false\n")
    project(tmp_path, "[agent]\nmax_turns = 2\n")
    config = load_config(tmp_path)
    assert config.agent.max_turns == 2
    assert config.agent.parallel_tools is False


def test_config_paths_are_in_precedence_order(tmp_path: Path) -> None:
    paths = config_paths(tmp_path)
    assert paths[0] == lab_harness_home() / CONFIG_FILE
    assert paths[1] == tmp_path / PROJECT_DIRECTORY / CONFIG_FILE
    assert paths[2] == tmp_path / PROJECT_DIRECTORY / LOCAL_CONFIG_FILE


# --- refusing bad input ---------------------------------------------------


def test_an_unknown_key_is_an_error(tmp_path: Path) -> None:
    """Silently ignoring a typo makes the setting appear not to work."""
    project(tmp_path, "[agent]\nmax_turnz = 7\n")
    with pytest.raises(ConfigError, match=r"unknown setting 'agent\.max_turnz'"):
        load_config(tmp_path)


def test_an_unknown_key_lists_the_available_ones(tmp_path: Path) -> None:
    project(tmp_path, "[agent]\nmax_turnz = 7\n")
    with pytest.raises(ConfigError, match="max_turns"):
        load_config(tmp_path)


def test_an_unknown_section_is_an_error(tmp_path: Path) -> None:
    project(tmp_path, "[nope]\nx = 1\n")
    with pytest.raises(ConfigError, match="unknown setting 'nope'"):
        load_config(tmp_path)


def test_a_wrong_type_is_an_error(tmp_path: Path) -> None:
    project(tmp_path, '[agent]\nmax_turns = "lots"\n')
    with pytest.raises(ConfigError, match=r"'agent\.max_turns' expects an integer"):
        load_config(tmp_path)


def test_a_boolean_is_not_an_integer(tmp_path: Path) -> None:
    """In Python `bool` subclasses `int`; an unguarded check would allow this."""
    project(tmp_path, "[agent]\nmax_turns = true\n")
    with pytest.raises(ConfigError, match=r"'agent\.max_turns' expects an integer"):
        load_config(tmp_path)


def test_an_integer_is_not_a_boolean(tmp_path: Path) -> None:
    project(tmp_path, "[agent]\nparallel_tools = 1\n")
    with pytest.raises(ConfigError, match="expects a boolean"):
        load_config(tmp_path)


def test_a_scalar_where_a_section_belongs_is_an_error(tmp_path: Path) -> None:
    project(tmp_path, "agent = 3\n")
    with pytest.raises(ConfigError, match="is a section"):
        load_config(tmp_path)


def test_the_defaults_are_the_catalogs_alone(tmp_path: Path) -> None:
    """A config file overrides one entry at a time; no setting is shared by all."""
    project(tmp_path, "[defaults.model]\nmax_tokens = 4096\n")
    with pytest.raises(ConfigError, match="unknown setting 'defaults'"):
        load_config(tmp_path)


def test_an_array_of_non_strings_is_an_error(tmp_path: Path) -> None:
    project(tmp_path, "[tools.approval]\nrequired = [1, 2]\n")
    with pytest.raises(ConfigError, match="an array of strings"):
        load_config(tmp_path)


def test_a_scalar_where_an_array_belongs_is_an_error(tmp_path: Path) -> None:
    project(tmp_path, '[tools.approval]\nrequired = "bash"\n')
    with pytest.raises(ConfigError, match="expects an array"):
        load_config(tmp_path)


def test_malformed_toml_is_an_error_naming_the_file(tmp_path: Path) -> None:
    path = project(tmp_path, "[agent\nmax_turns = 7\n")
    with pytest.raises(ConfigError, match=f"{path}.*invalid TOML"):
        load_config(tmp_path)


def test_the_error_names_the_file_it_came_from(tmp_path: Path) -> None:
    """With three layers, "which file is wrong" is the whole question."""
    path = project(tmp_path, "[agent]\nmax_turnz = 7\n", name=LOCAL_CONFIG_FILE)
    with pytest.raises(ConfigError, match=str(path)):
        load_config(tmp_path)


# --- home resolution ------------------------------------------------------


def test_lab_harness_home_follows_the_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_HARNESS_HOME", "/tmp/elsewhere")
    assert lab_harness_home() == Path("/tmp/elsewhere")


def test_lab_harness_home_expands_a_tilde(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LAB_HARNESS_HOME", raising=False)
    assert "~" not in str(lab_harness_home())


# --- named provider profiles ----------------------------------------------
#
# `providers.openai` is the one table where an unknown key is NOT an error:
# a new key is a new vendor, which is how one is added without code. Keys
# *inside* a profile are checked as strictly as anywhere else.


def test_the_shipped_profiles_are_present_by_default(tmp_path: Path) -> None:
    profiles = load_config(tmp_path).providers.openai

    assert set(profiles) >= {"openai", "deepseek", "glm"}
    assert profiles["deepseek"].api_key_env == "DEEPSEEK_API_KEY"


def test_the_vendors_that_want_reasoning_back_get_it() -> None:
    """DeepSeek 400s a tool loop without it; GLM asks for it unmodified."""
    profiles = Config().providers.openai

    assert profiles["deepseek"].replay_reasoning_field == "reasoning_content"
    assert profiles["glm"].replay_reasoning_field == "reasoning_content"


def test_a_new_profile_does_not_replay_reasoning() -> None:
    """An unknown server may reject a field it never asked for."""
    assert profile_defaults().replay_reasoning_field == ""


def test_only_glm_asks_the_server_to_preserve_thinking() -> None:
    """Replaying reasoning GLM then discards server-side would buy nothing."""
    profiles = Config().providers.openai

    assert profiles["glm"].preserve_thinking is True
    assert profiles["deepseek"].preserve_thinking is False
    assert profile_defaults().preserve_thinking is False


def test_a_new_profile_can_be_added_in_config(tmp_path: Path) -> None:
    """The whole point of the dict: a vendor with no code change."""
    project(
        tmp_path,
        "[providers.openai.groq]\n"
        'base_url = "https://api.groq.com/openai/v1"\n'
        'api_key_env = "GROQ_API_KEY"\n',
    )

    groq = load_config(tmp_path).providers.openai["groq"]

    assert groq.base_url == "https://api.groq.com/openai/v1"
    assert groq.api_key_env == "GROQ_API_KEY"


def test_a_new_profile_keeps_the_shipped_ones(tmp_path: Path) -> None:
    """Adding is not replacing — the same rule as every other config layer."""
    project(tmp_path, '[providers.openai.groq]\nbase_url = "https://api.groq.com/openai/v1"\n')

    profiles = load_config(tmp_path).providers.openai

    assert set(profiles) >= {"openai", "deepseek", "glm", "groq"}


def test_an_unset_key_in_a_new_profile_takes_the_catalog_default(tmp_path: Path) -> None:
    project(tmp_path, '[providers.openai.groq]\nbase_url = "https://api.groq.com/openai/v1"\n')

    groq = load_config(tmp_path).providers.openai["groq"]

    assert groq.timeout_seconds == profile_defaults().timeout_seconds == 120.0
    assert groq.include_usage is profile_defaults().include_usage is True


def test_an_existing_profile_merges_rather_than_replaces(tmp_path: Path) -> None:
    project(tmp_path, "[providers.openai.deepseek]\ntimeout_seconds = 30\n")

    deepseek = load_config(tmp_path).providers.openai["deepseek"]

    assert deepseek.timeout_seconds == 30.0
    # Untouched keys survive.
    assert deepseek.base_url == "https://api.deepseek.com/v1"
    assert deepseek.api_key_env == "DEEPSEEK_API_KEY"


def test_a_typo_inside_a_profile_is_still_an_error(tmp_path: Path) -> None:
    """Permissiveness stops at the profile name."""
    project(tmp_path, "[providers.openai.deepseek]\ntimeoutt_seconds = 30\n")

    with pytest.raises(ConfigError) as caught:
        load_config(tmp_path)

    message = str(caught.value)
    assert "providers.openai.deepseek.timeoutt_seconds" in message
    assert "timeout_seconds" in message, "the message must list what was available"


def test_a_wrong_type_inside_a_profile_is_still_an_error(tmp_path: Path) -> None:
    project(tmp_path, '[providers.openai.deepseek]\ntimeout_seconds = "soon"\n')

    with pytest.raises(ConfigError, match="timeout_seconds"):
        load_config(tmp_path)


def test_a_scalar_where_a_profile_belongs_is_an_error(tmp_path: Path) -> None:
    """`[providers] openai = 3` names no profile, so it cannot mean anything."""
    project(tmp_path, "[providers]\nopenai = 3\n")

    with pytest.raises(ConfigError, match="named sections"):
        load_config(tmp_path)


def test_a_scalar_in_place_of_a_profile_body_is_an_error(tmp_path: Path) -> None:
    project(tmp_path, "[providers.openai]\ndeepseek = 3\n")

    with pytest.raises(ConfigError, match="expected a TOML table"):
        load_config(tmp_path)


def test_a_local_file_can_carry_a_profile_key(tmp_path: Path) -> None:
    """The gitignored layer is where a credential belongs."""
    project(tmp_path, "[providers.openai.deepseek]\ntimeout_seconds = 30\n")
    project(
        tmp_path,
        '[providers.openai.deepseek]\napi_key = "placeholder-not-real"\n',
        name=LOCAL_CONFIG_FILE,
    )

    deepseek = load_config(tmp_path).providers.openai["deepseek"]

    assert deepseek.api_key == "placeholder-not-real"
    assert deepseek.timeout_seconds == 30.0


# --- models ---------------------------------------------------------------
#
# The other table of named sections, whose default is the shipped catalog. A
# key is a model id; an entry gives any of who serves it, its limits and flags,
# and its price.


def test_a_model_can_be_added_in_config(tmp_path: Path) -> None:
    # Quoted: a bare TOML key cannot contain the dot in "my-model-1.5".
    project(
        tmp_path,
        '[models."my-model-1.5"]\nprovider = "groq"\nmax_tokens = 4096\n'
        "price = { input = 0.6, output = 2.2 }\n",
    )

    entry = load_config(tmp_path).models["my-model-1.5"]

    assert (entry.provider, entry.max_tokens) == ("groq", 4_096)
    assert entry.context_window == model_defaults().context_window, "an unset key is the default"
    assert (entry.price.input, entry.price.output) == (0.6, 2.2)
    assert (entry.price.cache_read, entry.price.cache_write) == (0.0, 0.0), "an unset rate is zero"


def test_adding_a_model_keeps_the_catalog(tmp_path: Path) -> None:
    project(tmp_path, '[models."my-model-1.5"]\nprovider = "groq"\n')

    assert set(load_config(tmp_path).models) >= {"claude-sonnet-5", "glm-5.3", "my-model-1.5"}


def test_a_catalogued_model_merges_rather_than_replaces(tmp_path: Path) -> None:
    project(tmp_path, "[models.claude-sonnet-5.price]\ninput = 2.5\n")

    shipped = Config().models["claude-sonnet-5"]
    entry = load_config(tmp_path).models["claude-sonnet-5"]

    assert entry.price.input == 2.5
    assert entry.price.output == shipped.price.output, "the other rates survive"
    assert entry.max_tokens == shipped.max_tokens, "and so does the rest of the entry"


def test_a_typo_inside_a_model_is_an_error(tmp_path: Path) -> None:
    project(tmp_path, "[models.claude-sonnet-5]\nmax_tokenz = 1\n")

    with pytest.raises(ConfigError, match=r"models\.claude-sonnet-5\.max_tokenz"):
        load_config(tmp_path)


def test_a_typo_inside_a_price_is_an_error(tmp_path: Path) -> None:
    project(tmp_path, "[models.claude-sonnet-5.price]\ninptu = 2.5\n")

    with pytest.raises(ConfigError, match=r"models\.claude-sonnet-5\.price\.inptu"):
        load_config(tmp_path)


def test_a_setting_where_a_model_belongs_is_an_error(tmp_path: Path) -> None:
    """`[models] max_tokens = 2` sets nothing: every setting belongs to a model."""
    project(tmp_path, "[models]\nmax_tokens = 2\n")

    with pytest.raises(ConfigError, match="name one explicitly"):
        load_config(tmp_path)


def test_the_old_pricing_table_is_an_error(tmp_path: Path) -> None:
    """Prices moved into each model's entry; a table that does nothing must say so."""
    project(tmp_path, "[pricing.claude-sonnet-5]\ninput = 2.5\n")

    with pytest.raises(ConfigError, match="unknown setting 'pricing'"):
        load_config(tmp_path)


# --- snapshot -------------------------------------------------------------


def _field_names(cls: type) -> set[str]:
    """Every leaf field name in a config dataclass, through nested sections and
    named tables (`dict[str, Section]`)."""
    names: set[str] = set()
    for name, hint in get_type_hints(cls).items():
        target = get_args(hint)[-1] if get_origin(hint) is dict else hint
        if isinstance(target, type) and is_dataclass(target):
            names |= _field_names(target)
        else:
            names.add(name)
    return names


def test_a_snapshot_is_plain_data_with_credentials_replaced(tmp_path: Path) -> None:
    project(
        tmp_path,
        '[providers.openai.deepseek]\napi_key = "sk-from-a-local-file"\n',
        name=LOCAL_CONFIG_FILE,
    )

    recorded = snapshot(load_config(tmp_path))

    assert "sk-from-a-local-file" not in json.dumps(recorded)
    assert recorded["providers"]["openai"]["deepseek"]["api_key"] == REDACTED
    assert recorded["agent"]["max_turns"] == Config().agent.max_turns


def test_every_credential_field_is_redacted() -> None:
    """A new field named like a credential has to join SECRET_FIELDS, or the next
    session transcript records its value."""
    looks_secret = {n for n in _field_names(Config) if _CREDENTIAL_NAME.search(n)}

    assert looks_secret, "the pattern should find the credential fields that exist"
    assert looks_secret <= SECRET_FIELDS
