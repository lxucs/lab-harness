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

from lab_harness.config import Config, _apply, _member_type

EXAMPLE = Path(__file__).resolve().parent.parent / ".lab-harness" / "config.example.toml"

#: Stands in for the entry name in a table of named sections, whose keys are
#: schema while the names are data: one documented entry covers them all.
NAME = "<name>"

#: Keys deliberately absent from the example, and why. Anything else missing is
#: an undocumented setting.
INTENTIONALLY_ABSENT = {
    # A credential. The example documents `api_key_env` instead; the value
    # belongs in the environment or in gitignored config.local.toml.
    "tools.web.tavily.api_key",
    # A machine-specific endpoint, for the same reason.
    "tools.web.tavily.base_url",
    # Same rule, for an OpenAI-compatible profile.
    f"providers.openai.{NAME}.api_key",
    # Prose that ships in prompts.toml, where it is documented. A copy here
    # would have to change with every edit to the default prompt.
    "prompts.system",
    "prompts.project_instructions",
}

#: Substrings that would mean a real secret leaked into a committed file.
SECRET_MARKERS = ("sk-", "tvly-", "Bearer ")


def example_data() -> dict[str, Any]:
    with EXAMPLE.open("rb") as handle:
        return tomllib.load(handle)


def leaf_paths(instance: object, trail: tuple[str, ...] = ()) -> set[str]:
    """Every scalar setting in the schema, as dotted paths.

    A dict field is a table of *named* sections (`providers.openai.deepseek`).
    Its keys come from the entry type, not from whatever entries ship: the
    names are data, but the keys inside a section are schema and must be
    documented like any other.
    """
    found: set[str] = set()
    for field in fields(instance):  # type: ignore[arg-type]
        value = getattr(instance, field.name)
        where = (*trail, field.name)
        if is_dataclass(value):
            found |= leaf_paths(value, where)
        elif isinstance(value, dict):
            found |= leaf_paths(_member_type(field)(), (*where, NAME))
        else:
            found.add(".".join(where))
    return found


def documented_paths(
    data: dict[str, Any], schema: object = None, trail: tuple[str, ...] = ()
) -> set[str]:
    """Every scalar the example sets, as dotted paths, named entries as `NAME`."""
    schema = Config() if schema is None else schema
    known = {field.name: field for field in fields(schema)}  # type: ignore[arg-type]
    found: set[str] = set()
    for key, value in data.items():
        where = (*trail, key)
        current = getattr(schema, key, None)
        if isinstance(current, dict):
            member = _member_type(known[key])
            for section in value.values():
                found |= documented_paths(section, member(), (*where, NAME))
        elif is_dataclass(current):
            found |= documented_paths(value, current, where)
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
