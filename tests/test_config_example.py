"""The committed config example must stay true to the schema.

An example that drifts is worse than none: it documents keys that do nothing
and omits the ones you need. Now that config files actually load, the example
can be checked by *loading* it rather than by eyeballing it — if it applies
cleanly and changes nothing, every key exists and every value is the default.

The LLM sections are the exception. Every LLM setting is in
lab_harness/llm/catalog.toml, whose loader refuses a defaults table that leaves
a key out — so the catalog documents each one, and the example none.
"""

import tomllib
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import pytest

from lab_harness.config import Config, _apply

EXAMPLE = Path(__file__).resolve().parent.parent / ".lab-harness" / "config.example.toml"

#: Sections documented in the catalog, and only there.
LLM_SECTIONS = frozenset({"providers", "models"})

#: Keys deliberately absent from the example, and why. Anything else missing is
#: an undocumented setting.
INTENTIONALLY_ABSENT = {
    # A credential. The example documents `api_key_env` instead; the value
    # belongs in the environment or in gitignored config.local.toml.
    "tools.web.tavily.api_key",
    # A machine-specific endpoint, for the same reason.
    "tools.web.tavily.base_url",
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
    """Every scalar setting in the schema outside the LLM sections, as dotted paths."""
    found: set[str] = set()
    for field in fields(instance):  # type: ignore[arg-type]
        if not trail and field.name in LLM_SECTIONS:
            continue
        value = getattr(instance, field.name)
        where = (*trail, field.name)
        if is_dataclass(value):
            found |= leaf_paths(value, where)
        else:
            found.add(".".join(where))
    return found


def documented_paths(
    data: dict[str, Any], schema: object = None, trail: tuple[str, ...] = ()
) -> set[str]:
    """Every scalar the example sets, as dotted paths."""
    schema = Config() if schema is None else schema
    found: set[str] = set()
    for key, value in data.items():
        where = (*trail, key)
        current = getattr(schema, key, None)
        if is_dataclass(current):
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


def test_llm_settings_are_left_to_the_catalog() -> None:
    """One place for every LLM setting, so the example holds none of them."""
    present = sorted(LLM_SECTIONS & example_data().keys())
    assert not present, f"Document these in lab_harness/llm/catalog.toml instead: {present}"
