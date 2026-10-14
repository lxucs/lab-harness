"""The real loop, driven by a scripted model: no API key, no network.

    uv run python examples/scripted.py

`FakeProvider` stands in for the model and replays one canned response per
request — here, a turn that calls `ls` and a turn that answers. Everything else
is real: the loop, the tool, the events. That makes it the cheap way to try a
hook or a tool before spending tokens on it, and it is how the test suite runs
offline.
"""

import asyncio

from lab_harness.agent import Agent, AgentEvent
from lab_harness.config import load_config
from lab_harness.llm.fake import FAKE_MODEL, FakeProvider, text_response, tool_call_response
from lab_harness.llm.types import AssistantMessage, ContentBlock, TextBlock, ToolCallBlock
from lab_harness.tools import default_registry

SCRIPT = [
    tool_call_response(
        [ToolCallBlock(id="call-1", name="ls", arguments={"path": "."})],
        preamble="I'll list the directory first.",
    ),
    text_response("Those are the files in the working directory."),
]


def joined(content: tuple[ContentBlock, ...]) -> str:
    return "".join(block.text for block in content if isinstance(block, TextBlock))


async def show(event: AgentEvent) -> None:
    """Print each finished assistant message and each tool result."""
    if event.type == "message_end" and isinstance(event.message, AssistantMessage):
        print(f"model: {joined(event.message.content)}")
    elif event.type == "tool_execution_end":
        print(f"{event.tool_name} returned:\n{joined(event.content)}")


async def main() -> None:
    provider = FakeProvider(SCRIPT)
    agent = Agent(
        model=FAKE_MODEL,
        provider=provider,
        tools=default_registry().select(["ls"]),
        config=load_config(),
    )
    agent.subscribe(show)
    await agent.prompt("What is in this directory?")

    # Every request the loop made, as a real provider would have received it.
    for number, request in enumerate(provider.requests, 1):
        print(f"request {number}: {', '.join(m.role for m in request.messages)}")


if __name__ == "__main__":
    asyncio.run(main())
