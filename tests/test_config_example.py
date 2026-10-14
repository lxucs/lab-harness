"""The committed config example must stay true to the schema.

An example that drifts is worse than none: it documents keys that do nothing
and omits the ones you need. Now that config files actually load, the example
can be checked by *loading* it rather than by eyeballing it — if it applies
cleanly and changes nothing, every key exists and every value is the default.
"""

import tomllib
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import pytest

from lab_harness.config import Config, _apply

EXAMPLE = Path(__file__).resolve().parent.parent / ".lab-harness" / "config.toml.example"

#: Keys deliberately absent from the example, and why. Anything else missing is
#: an undocumented setting.
INTENTIONALLY_ABSENT = {
    # A credential. The example documents `api_key_env` instead; the value
    # belongs in the environment or in gitignored config.local.toml.
    "tools.web.tavily.api_key",
    # A machine-specific endpoint, for the same reason.
    "tools.web.tavily.base_url",
    # Same rule, once per shipped OpenAI-compatible profile.
    "providers.openai.openai.api_key",
    "providers.openai.deepseek.api_key",
    "providers.openai.glm.api_key",
}

#: Substrings that would mean a real secret leaked into a committed file.
SECRET_MARKERS = ("sk-", "tvly-", "Bearer ")


def example_data() -> dict[str, Any]:
    with EXAMPLE.open("rb") as handle:
        return tomllib.load(handle)


def leaf_paths(instance: object, trail: tuple[str, ...] = ()) -> set[str]:
    """Every scalar setting in the schema, as dotted paths.

    A dict field is a table of *named* sections (`providers.openai.deepseek`),
    so it recurses one level deeper than a plain one — the names are data, but
    the keys inside each section are schema and must be documented like any
    other.
    """
    found: set[str] = set()
    for field in fields(instance):  # type: ignore[arg-type]
        value = getattr(instance, field.name)
        where = (*trail, field.name)
        if is_dataclass(value):
            found |= leaf_paths(value, where)
        elif isinstance(value, dict):
            for name, section in value.items():
                found |= leaf_paths(section, (*where, name))
        else:
            found.add(".".join(where))
    return found


def documented_paths(data: dict[str, Any], trail: tuple[str, ...] = ()) -> set[str]:
    found: set[str] = set()
    for key, value in data.items():
        where = (*trail, key)
        if isinstance(value, dict):
            found |= documented_paths(value, where)
        else:
            found.add(".".join(where))
    return found


def test_the_example_exists() -> None:
    assert EXAMPLE.is_file()


def test_the_example_is_valid_toml() -> None:
    assert example_data()


def test_the_example_applies_without_error() -> None:
    """Catches a key that was renamed in the schema but not in the example."""
    _apply(Config(), example_data(), source=EXAMPLE, trail=())


def test_every_documented_value_is_the_current_default() -> None:
    """The example claims these are the defaults, so applying it must be a no-op."""
    assert _apply(Config(), example_data(), source=EXAMPLE, trail=()) == Config()


def test_every_schema_key_is_documented() -> None:
    missing = leaf_paths(Config()) - documented_paths(example_data()) - INTENTIONALLY_ABSENT
    assert not missing, f"Undocumented settings: {sorted(missing)}"


def test_nothing_documented_is_absent_from_the_schema() -> None:
    """The reverse drift: a key removed from the schema but left in the example."""
    extra = documented_paths(example_data()) - leaf_paths(Config())
    assert not extra, f"Example documents settings that do not exist: {sorted(extra)}"


def test_the_example_contains_no_secrets() -> None:
    body = EXAMPLE.read_text()
    found = [marker for marker in SECRET_MARKERS if marker in body]
    assert not found, f"Possible credential in a committed file: {found}"


@pytest.mark.parametrize("key", sorted(INTENTIONALLY_ABSENT))
def test_intentionally_absent_keys_really_are_absent(key: str) -> None:
    """Keeps the allowlist honest: an entry that got documented should be removed."""
    assert key not in documented_paths(example_data())
