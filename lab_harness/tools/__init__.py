"""Built-in capabilities and the registry that declares them to the model."""

from lab_harness.tools.base import (
    AgentTool,
    ExecutionMode,
    FunctionTool,
    ToolContext,
    ToolResult,
    fail,
    ok,
    tool,
)
from lab_harness.tools.fs import FILESYSTEM_TOOLS
from lab_harness.tools.registry import ToolRegistry
from lab_harness.tools.search import SEARCH_TOOLS
from lab_harness.tools.shell import SHELL_TOOLS
from lab_harness.tools.web import WEB_TOOLS

#: Everything built in. The CLI narrows this with the configured tool list.
BUILTIN_TOOLS: tuple[AgentTool, ...] = FILESYSTEM_TOOLS + SHELL_TOOLS + SEARCH_TOOLS + WEB_TOOLS


def default_registry() -> ToolRegistry:
    """A registry holding every built-in tool."""
    return ToolRegistry(BUILTIN_TOOLS)


__all__ = [
    "BUILTIN_TOOLS",
    "AgentTool",
    "ExecutionMode",
    "FunctionTool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "default_registry",
    "fail",
    "ok",
    "tool",
]
