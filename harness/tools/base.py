"""The tool contract, and the decorator that makes satisfying it one line.

A tool is anything matching `AgentTool`: a name, a description, a JSON Schema
for its parameters, an execution mode, and an async `execute`. The loop knows
nothing else about it, which is why a tool can live anywhere.

These types live here rather than in `agent/` because dependencies point
downward: `agent/` consumes tools, so `tools/` must not import from it.

`@tool` is the ergonomic path. Declare parameters as a Pydantic model and the
decorator derives the schema, validates arguments, and turns a validation
failure into an error result the model can read and correct — rather than an
exception the loop has to launder.
"""

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol, get_type_hints

from pydantic import BaseModel, ValidationError

from harness.config import Config
from harness.llm.types import ContentBlock, text

ExecutionMode = Literal["parallel", "sequential"]


@dataclass(frozen=True, slots=True)
class ToolResult:
    content: tuple[ContentBlock, ...]
    is_error: bool = False
    #: Structured data for renderers. Never sent to the model.
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolContext:
    """What a tool receives. Tools never reach for globals."""

    cwd: str
    call_id: str
    #: Resolved configuration. Tools read their limits from here rather than
    #: hardcoding them — see docs/contributing.md.
    config: Config = field(default_factory=Config)
    signal: asyncio.Event | None = None
    on_update: Callable[[ToolResult], None] | None = None


class AgentTool(Protocol):
    name: str
    description: str
    parameters: dict[str, object]
    execution_mode: ExecutionMode

    async def execute(self, args: dict[str, object], ctx: ToolContext) -> ToolResult: ...


def ok(message: str, **details: object) -> ToolResult:
    """A successful single-text-block result."""
    return ToolResult(content=text(message), details=details)


def fail(message: str, **details: object) -> ToolResult:
    """A failed result. The message is what the model sees, so make it actionable."""
    return ToolResult(content=text(message), is_error=True, details=details)


class FunctionTool[P: BaseModel]:
    """An `AgentTool` built from an async function plus a Pydantic parameter model.

    Produced by `@tool`; there is rarely a reason to construct one directly.
    """

    def __init__(
        self,
        fn: Callable[[P, ToolContext], Awaitable[ToolResult]],
        *,
        name: str,
        description: str,
        params: type[P],
        execution_mode: ExecutionMode,
    ) -> None:
        self._fn = fn
        self._params = params
        self.name = name
        self.description = description
        self.execution_mode: ExecutionMode = execution_mode
        self.parameters: dict[str, object] = params.model_json_schema()

    async def execute(self, args: dict[str, object], ctx: ToolContext) -> ToolResult:
        try:
            parsed = self._params.model_validate(args)
        except ValidationError as exc:
            # A validation failure is feedback, not a crash: the model gets the
            # field-level reasons and can reissue the call.
            return fail(f"Invalid arguments for {self.name!r}:\n{_explain(exc)}")
        return await self._fn(parsed, ctx)

    def __repr__(self) -> str:
        return f"<tool {self.name}>"


def tool[P: BaseModel](
    *,
    name: str,
    description: str,
    execution_mode: ExecutionMode = "parallel",
) -> Callable[[Callable[[P, ToolContext], Awaitable[ToolResult]]], FunctionTool[P]]:
    """Turn an async function into a registrable tool.

    The parameter model is read from the first argument's annotation, so the
    schema and the implementation cannot drift apart.

        @tool(name="read", description="Read a file.")
        async def read(p: ReadParams, ctx: ToolContext) -> ToolResult: ...
    """

    def decorate(fn: Callable[[P, ToolContext], Awaitable[ToolResult]]) -> FunctionTool[P]:
        return FunctionTool(
            fn,
            name=name,
            description=description,
            params=_params_model(fn),
            execution_mode=execution_mode,
        )

    return decorate


def _params_model[P: BaseModel](fn: Callable[[P, ToolContext], Awaitable[ToolResult]]) -> type[P]:
    """Read the parameter model off the function's first annotation."""
    signature = inspect.signature(fn)
    names = list(signature.parameters)
    if len(names) != 2:
        raise TypeError(f"{fn.__name__} must take exactly (params, ctx), got {names}")
    annotation = get_type_hints(fn).get(names[0])
    if not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
        raise TypeError(
            f"{fn.__name__}'s first parameter must be annotated with a Pydantic "
            f"model; got {annotation!r}"
        )
    return annotation  # type: ignore[return-value]


def _explain(exc: ValidationError) -> str:
    """Render a ValidationError as flat `field: reason` lines for the model."""
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        lines.append(f"  {location}: {error['msg']}")
    return "\n".join(lines)
