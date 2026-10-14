"""JSONL event subscriber: one event per line, to any text stream.

Small on purpose, and load-bearing for three reasons. It makes the runtime
scriptable (`lab-harness -p … --json | jq`); it gives end-to-end tests a structured
surface to assert on instead of scraped terminal output; and — most of all — a
second consumer is what keeps the "events are plain serializable data" rule from
decaying. A rule with only one consumer is a rule nobody checks.
"""

import json
import sys
from collections.abc import Awaitable, Callable
from typing import TextIO

from lab_harness.agent.types import AgentEvent
from lab_harness.llm.serde import to_dict


def json_sink(stream: TextIO | None = None) -> Callable[[AgentEvent], Awaitable[None]]:
    """Build a subscriber that writes each event to `stream` as one JSON line.

    Defaults to stdout. Flushes per line so a piped consumer sees events as
    they happen rather than when the process exits.
    """
    target = stream if stream is not None else sys.stdout

    async def write(event: AgentEvent) -> None:
        target.write(json.dumps(to_dict(event)) + "\n")
        target.flush()

    return write
