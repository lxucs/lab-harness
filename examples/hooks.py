"""All four hooks, as the methods of one experiment object.

    uv run python examples/hooks.py

| Hook | Runs | Used here to |
|---|---|---|
| `before_tool_call` | before each tool call | hold the run to a tool-call budget |
| `after_tool_call` | after it, before anything sees the result | redact email addresses |
| `prepare_request` | before each provider request | log the request's size |
| `finish_turn` | when the model stops without calling a tool | send it back for one self-check |

A hook that returns None changes nothing. A feature that fits a hook does not
belong in the loop — the CLI's approval prompt is a `before_tool_call` hook.
More in docs/architecture.md.
"""

import asyncio
import re
import sys

from lab_harness.agent import Agent, AgentEvent
from lab_harness.agent.types import (
    AfterToolCallContext,
    AfterToolCallResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    FinishTurnContext,
    FinishTurnResult,
    PrepareRequestContext,
    RequestUpdate,
)
from lab_harness.config import load_config
from lab_harness.llm.providers import route
from lab_harness.llm.types import AssistantMessage, TextBlock, UserMessage, text
from lab_harness.tools import default_registry

PROMPT = "Find where the agent loop decides a run is finished, and explain it in two sentences."
READ_ONLY_TOOLS = ["read", "ls", "grep", "glob"]
TOOL_BUDGET = 8
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
SELF_CHECK = (
    "Before you finish, check your answer against what the tools returned and "
    "correct anything they do not support. Then give the final answer."
)


class Experiment:
    """State the hooks share. Each bound method is a hook in its own right."""

    def __init__(self, tool_budget: int) -> None:
        self.tool_budget = tool_budget
        self.tool_calls = 0
        self.requests = 0
        self.checked = False

    async def before_tool_call(self, ctx: BeforeToolCallContext) -> BeforeToolCallResult | None:
        """Refuse calls past the budget. The reason is what the model reads."""
        self.tool_calls += 1
        if self.tool_calls <= self.tool_budget:
            return None
        reason = f"The budget of {self.tool_budget} tool calls is spent. Answer with what you have."
        return BeforeToolCallResult(block=True, reason=reason)

    async def after_tool_call(self, ctx: AfterToolCallContext) -> AfterToolCallResult | None:
        """Redact email addresses before the model, a renderer or the transcript sees them."""
        content = tuple(
            TextBlock(text=EMAIL.sub("[email]", b.text)) if isinstance(b, TextBlock) else b
            for b in ctx.result.content
        )
        if content == ctx.result.content:
            return None
        # `details` reach every renderer, and an edit's diff there would show
        # the text again, so a redacting hook drops them too.
        return AfterToolCallResult(content=content, details={})

    async def prepare_request(self, ctx: PrepareRequestContext) -> RequestUpdate | None:
        """Log each request. Returning `RequestUpdate(messages=...)` would send a
        different history instead — the seam context compaction will use."""
        self.requests += 1
        print(f"[request {self.requests}: {len(ctx.messages)} message(s) to {ctx.model.id}]")
        return None

    async def finish_turn(self, ctx: FinishTurnContext) -> FinishTurnResult | None:
        """Send the model back once to check its answer. Whatever a hook does,
        `agent.max_turns` bounds the run."""
        if self.checked:
            return None
        self.checked = True
        return FinishTurnResult(continue_with=text(SELF_CHECK))


async def show(event: AgentEvent) -> None:
    """Print each finished message, and the outcome of each tool call."""
    if event.type == "message_end" and isinstance(event.message, UserMessage | AssistantMessage):
        body = "".join(b.text for b in event.message.content if isinstance(b, TextBlock))
        if body:
            print(f"{event.message.role}: {body}")
    elif event.type == "tool_execution_end":
        print(f"  {event.tool_name}: {'refused or failed' if event.is_error else 'ok'}")


async def main(prompt: str) -> None:
    config = load_config()
    model, provider = route(None, config)
    experiment = Experiment(TOOL_BUDGET)
    agent = Agent(
        model=model,
        provider=provider,
        tools=default_registry().select(READ_ONLY_TOOLS),
        config=config,
        before_tool_call=experiment.before_tool_call,
        after_tool_call=experiment.after_tool_call,
        prepare_request=experiment.prepare_request,
        finish_turn=experiment.finish_turn,
    )
    agent.subscribe(show)
    await agent.prompt(prompt)
    print(f"[{experiment.requests} request(s), {experiment.tool_calls} tool call(s)]")


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or PROMPT))
