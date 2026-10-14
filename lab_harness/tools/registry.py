"""Name-to-tool lookup.

A registry is an ordinary object, constructed per run and passed down. There is
deliberately no module-level default instance: a global registry is the thing
that makes two tests with different tool sets interfere.
"""

from collections.abc import Iterable, Iterator, Sequence

from lab_harness.tools.base import AgentTool


class ToolRegistry:
    """An ordered, name-unique collection of executable tools."""

    def __init__(self, tools: Iterable[AgentTool] = ()) -> None:
        self._tools: dict[str, AgentTool] = {}
        for item in tools:
            self.register(item)

    def register(self, tool: AgentTool) -> None:
        """Add a tool. Duplicate names are a wiring bug, not a silent override."""
        if tool.name in self._tools:
            raise ValueError(f"A tool named {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> AgentTool | None:
        return self._tools.get(name)

    def select(self, names: Sequence[str]) -> tuple[AgentTool, ...]:
        """Resolve configured tool names, in the order given.

        An unknown name raises: a typo in config that silently disabled a tool
        would be found much later, by the model failing to do its job.
        """
        unknown = [name for name in names if name not in self._tools]
        if unknown:
            available = ", ".join(sorted(self._tools))
            raise ValueError(f"Unknown tool(s): {', '.join(unknown)}. Available: {available}")
        return tuple(self._tools[name] for name in names)

    def __iter__(self) -> Iterator[AgentTool]:
        return iter(self._tools.values())

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
