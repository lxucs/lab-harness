"""Drive the agent from Python: stream its answer and show each tool call.

    uv run python examples/quickstart.py
    uv run python examples/quickstart.py "Where does the agent loop stop?"

The model comes from config — `$LAB_HARNESS_MODEL`, else `agent.model` — so the
same script runs against any configured vendor, given that vendor's API key
(`ANTHROPIC_API_KEY` for the default). The tools are read-only: nothing on disk
changes.
"""

import asyncio
import sys

from lab_harness.agent import Agent, AgentEvent
from lab_harness.config import load_config
from lab_harness.llm.providers import route
from lab_harness.llm.types import AssistantMessage
from lab_harness.tools import default_registry

PROMPT = "What does this project do? Answer in three sentences."
SYSTEM_PROMPT = "You are a concise assistant exploring a code base."
READ_ONLY_TOOLS = ["read", "ls", "grep", "glob"]


async def show(event: AgentEvent) -> None:
    """Print the answer as it streams, and one line per tool call."""
    if event.type == "message_update" and event.kind == "text":
        print(event.delta, end="", flush=True)
    elif event.type == "tool_execution_start":
        print(f"\n→ {event.tool_name}({event.arguments})", flush=True)


async def main(prompt: str) -> None:
    config = load_config()
    model, provider = route(None, config)  # what the CLI does when --model is not given
    agent = Agent(
        model=model,
        provider=provider,
        system_prompt=SYSTEM_PROMPT,
        tools=default_registry().select(READ_ONLY_TOOLS),
        config=config,
    )
    agent.subscribe(show)
    produced = await agent.prompt(prompt)

    # `produced` is this run's messages; `agent.state.messages` is the whole
    # conversation, and a second `prompt()` continues it.
    responses = [m for m in produced if isinstance(m, AssistantMessage)]
    tokens = sum(m.usage.total for m in responses)
    print(f"\n\n[{model.id} via {model.provider}: {len(responses)} request(s), {tokens} tokens]")


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or PROMPT))
