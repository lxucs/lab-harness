"""A new tool: one decorated function, registered beside a few built-ins.

    uv run python examples/custom_tool.py

The parameters are a Pydantic model, so the JSON schema the model is shown,
the validation of what it sends back, and the type hints are one declaration.
Invalid arguments never reach the function: the loop answers them with an
error the model can correct itself from. More in docs/tools.md.
"""

import asyncio
import re
import sys
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from lab_harness.agent import Agent, AgentEvent
from lab_harness.config import load_config
from lab_harness.llm.providers import route
from lab_harness.tools import (
    ToolContext,
    ToolRegistry,
    ToolResult,
    default_registry,
    fail,
    ok,
    tool,
)

PROMPT = "Which words does README.md use most, leaving out the obvious ones?"


class WordCountParams(BaseModel):
    path: str = Field(description="Text file to count, relative to the working directory")
    top: int = Field(10, ge=1, le=100, description="How many of the most common words to list")


@tool(name="word_count", description="Count the words in a text file and list the most common.")
async def word_count(p: WordCountParams, ctx: ToolContext) -> ToolResult:
    # Resolve against ctx.cwd, never the process's working directory: that is
    # what lets two agents with different roots share one process.
    path = Path(ctx.cwd, p.path)
    try:
        words = re.findall(r"[a-z']+", path.read_text(encoding="utf-8").lower())
    # Error text is written for the model: what went wrong, and what to try.
    except OSError as exc:
        return fail(f"Cannot read {p.path}: {exc.strerror}. Use ls to find the file.")
    except UnicodeDecodeError:
        return fail(f"{p.path} is not UTF-8 text. Pick a text file.")
    lines = [f"{count}\t{word}" for word, count in Counter(words).most_common(p.top)]
    # `details` reach renderers and --json logs; the model never sees them.
    return ok(f"{len(words)} words in {p.path}\n" + "\n".join(lines), words=len(words))


async def show(event: AgentEvent) -> None:
    if event.type == "message_update" and event.kind == "text":
        print(event.delta, end="", flush=True)
    elif event.type == "tool_execution_start":
        print(f"\n→ {event.tool_name}({event.arguments})", flush=True)


async def main(prompt: str) -> None:
    config = load_config()
    model, provider = route(None, config)
    registry = ToolRegistry(default_registry().select(["ls", "glob"]))
    registry.register(word_count)
    agent = Agent(model=model, provider=provider, tools=tuple(registry), config=config)
    agent.subscribe(show)
    await agent.prompt(prompt)
    print()


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or PROMPT))
