"""Layered TOML config loading.

The interesting behavior is not "a value is read" but the three things that
make config trustworthy: later files win, a typo is an error rather than a
silent no-op, and a wrong type is caught at startup rather than at the call
site hours later.
"""

from pathlib import Path

import pytest

from harness.config import (
    CONFIG_FILE,
    LOCAL_CONFIG_FILE,
    PROJECT_DIRECTORY,
    Config,
    ConfigError,
    config_paths,
    harness_home,
    load_config,
)


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
    write(harness_home() / CONFIG_FILE, "[agent]\nmax_turns = 1\n")
    project(tmp_path, "[agent]\nmax_turns = 2\n")
    assert load_config(tmp_path).agent.max_turns == 2


def test_the_local_file_overrides_the_project_file(tmp_path: Path) -> None:
    """config.local.toml is last so a machine can override a committed value."""
    project(tmp_path, "[agent]\nmax_turns = 2\n")
    project(tmp_path, "[agent]\nmax_turns = 3\n", name=LOCAL_CONFIG_FILE)
    assert load_config(tmp_path).agent.max_turns == 3


def test_layers_merge_rather_than_replace(tmp_path: Path) -> None:
    """A later file overrides the keys it names and leaves the rest alone."""
    write(harness_home() / CONFIG_FILE, "[agent]\nmax_turns = 1\nparallel_tools = false\n")
    project(tmp_path, "[agent]\nmax_turns = 2\n")
    config = load_config(tmp_path)
    assert config.agent.max_turns == 2
    assert config.agent.parallel_tools is False


def test_config_paths_are_in_precedence_order(tmp_path: Path) -> None:
    paths = config_paths(tmp_path)
    assert paths[0] == harness_home() / CONFIG_FILE
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


def test_harness_home_follows_the_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_HOME", "/tmp/elsewhere")
    assert harness_home() == Path("/tmp/elsewhere")


def test_harness_home_expands_a_tilde(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HARNESS_HOME", raising=False)
    assert "~" not in str(harness_home())
