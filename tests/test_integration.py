"""End-to-end through the public API: a scripted model drives a real tool."""

from harness.agent import Agent, ToolContext, ToolResult
from harness.agent.types import AgentEvent, ExecutionMode
from harness.llm.fake import FAKE_MODEL, FakeProvider, text_response, tool_call_response
from harness.llm.types import TextBlock, ToolCallBlock, text


class CounterTool:
    name = "increment"
    description = "Add one to a number"
    execution_mode: ExecutionMode = "parallel"

    def __init__(self) -> None:
        self.parameters: dict[str, object] = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        }

    async def execute(self, args: dict[str, object], ctx: ToolContext) -> ToolResult:
        return ToolResult(content=text(str(int(str(args["value"])) + 1)))


async def test_full_tool_calling_round_trip() -> None:
    call = ToolCallBlock(id="c1", name="increment", arguments={"value": 41})
    provider = FakeProvider(
        [
            tool_call_response([call], preamble="Let me compute that."),
            text_response("The answer is 42."),
        ]
    )
    events: list[AgentEvent] = []

    agent = Agent(
        model=FAKE_MODEL,
        provider=provider,
        system_prompt="You are a harness.",
        tools=(CounterTool(),),
    )
    agent.subscribe(lambda e: _record(events, e))

    await agent.prompt("What is 41 plus one?")

    last_block = agent.state.messages[-1].content[0]
    assert isinstance(last_block, TextBlock)
    assert last_block.text == "The answer is 42."
    roles = [m.role for m in agent.state.messages]
    assert roles == ["user", "assistant", "tool_result", "assistant"]

    kinds = [e.type for e in events]
    assert kinds.count("turn_start") == 2
    assert "tool_execution_end" in kinds


async def _record(sink: list[AgentEvent], event: AgentEvent) -> None:
    sink.append(event)
