"""The tool contract: schema derivation, validation, and error conversion."""

from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from lab_harness.llm.types import TextBlock
from lab_harness.tools.base import ToolContext, ToolResult, fail, ok, tool
from lab_harness.tools.registry import ToolRegistry


class Params(BaseModel):
    name: str = Field(description="Who to greet")
    times: int = Field(default=1, ge=1, description="How many times")


@tool(name="greet", description="Greet someone.")
async def greet(p: Params, ctx: ToolContext) -> ToolResult:
    return ok(" ".join([f"hi {p.name}"] * p.times))


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=str(tmp_path), call_id="c1")


def _text(result: ToolResult) -> str:
    return "".join(b.text for b in result.content if isinstance(b, TextBlock))


def test_decorator_derives_schema_from_the_model() -> None:
    schema = greet.parameters
    assert schema["type"] == "object"
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == {"name", "times"}
    assert properties["name"]["description"] == "Who to greet"
    assert schema["required"] == ["name"]


def test_decorator_carries_name_description_and_mode() -> None:
    assert greet.name == "greet"
    assert greet.description == "Greet someone."
    assert greet.execution_mode == "parallel"


async def test_valid_arguments_reach_the_function_parsed(tmp_path: Path) -> None:
    result = await greet.execute({"name": "ada", "times": 2}, _ctx(tmp_path))
    assert _text(result) == "hi ada hi ada"
    assert not result.is_error


async def test_defaults_apply_when_omitted(tmp_path: Path) -> None:
    result = await greet.execute({"name": "ada"}, _ctx(tmp_path))
    assert _text(result) == "hi ada"


async def test_invalid_arguments_become_an_error_result_not_an_exception(tmp_path: Path) -> None:
    """The model must be able to read the failure and reissue the call."""
    result = await greet.execute({"times": 2}, _ctx(tmp_path))

    assert result.is_error
    message = _text(result)
    assert "greet" in message
    assert "name" in message


async def test_constraint_violations_are_reported_with_the_field(tmp_path: Path) -> None:
    result = await greet.execute({"name": "ada", "times": 0}, _ctx(tmp_path))

    assert result.is_error
    assert "times" in _text(result)


async def _not_a_model(p: str, ctx: ToolContext) -> ToolResult:
    return ok("never")


async def _wrong_arity(p: Params) -> ToolResult:
    return ok("never")


def test_a_function_without_a_pydantic_first_arg_is_rejected_at_definition() -> None:
    """A bad signature fails when the tool is defined, not when the model calls it."""
    with pytest.raises(TypeError, match="Pydantic model"):
        tool(name="bad", description="bad")(_not_a_model)  # pyright: ignore[reportArgumentType]


def test_a_function_with_the_wrong_arity_is_rejected_at_definition() -> None:
    with pytest.raises(TypeError, match="exactly"):
        tool(name="bad", description="bad")(_wrong_arity)  # pyright: ignore[reportArgumentType]


def test_ok_and_fail_set_the_error_flag() -> None:
    assert not ok("fine").is_error
    assert fail("broken").is_error


def test_details_are_carried_but_stay_off_the_content() -> None:
    """details is for renderers; the model sees content only."""
    result = ok("done", path="/x", lines=3)
    assert result.details == {"path": "/x", "lines": 3}
    assert _text(result) == "done"


# --- registry -------------------------------------------------------------


def test_registry_resolves_by_name() -> None:
    registry = ToolRegistry([greet])
    assert registry.get("greet") is greet
    assert registry.get("absent") is None


def test_duplicate_registration_is_an_error() -> None:
    registry = ToolRegistry([greet])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(greet)


def test_select_returns_tools_in_the_requested_order() -> None:
    @tool(name="other", description="other")
    async def other(p: Params, ctx: ToolContext) -> ToolResult:
        return ok("other")

    registry = ToolRegistry([greet, other])
    assert registry.select(["other", "greet"]) == (other, greet)


def test_select_rejects_unknown_names_loudly() -> None:
    """A typo in configured tool names must fail at wiring, not at runtime."""
    registry = ToolRegistry([greet])
    with pytest.raises(ValueError, match="Unknown tool"):
        registry.select(["greet", "typo"])


def test_registry_supports_iteration_and_membership() -> None:
    registry = ToolRegistry([greet])
    assert [t.name for t in registry] == ["greet"]
    assert "greet" in registry
    assert len(registry) == 1
