# Events

The event stream is the contract between the runtime and anything watching it.
`Agent.subscribe(listener)` is the only way to observe a run.

The provider stream protocol lives in `harness/llm/events.py`, agent-level
events in `harness/agent/types.py`, and subscriber fan-out in
`harness/agent/agent.py`. The JSONL sink in `harness/cli/json_sink.py` is the
second consumer, which is what keeps the plain-data rule honest.

## The rule that makes this work

**Every event is a frozen dataclass of JSON primitives.** No live objects, no
file handles, no provider SDK types, no callables.

`tests/test_serialization.py` round-trips every event type through
`json.dumps` and fails if this is violated. The rule matters because it is what
lets a second consumer exist — the JSONL sink today, a frontend in another
language later — without the runtime changing.

A rule nobody executes decays. That test is the enforcement.

## Lifecycle

```
agent_start
  message_start/end          ← the user message that triggered this run
  turn_start
    message_start   (assistant)
    message_update  × N        ← streaming deltas
    message_end
    tool_execution_start  × N
    tool_execution_update × N  ← e.g. bash output as it arrives
    tool_execution_end    × N
    message_start/end     × N  ← tool results, in assistant source order
  turn_end
  …more turns while the model keeps requesting tools…
agent_end
```

## Event types

| Event | Fields | Notes |
|---|---|---|
| `agent_start` | — | One per run |
| `agent_end` | `messages` | Last event of a run |
| `turn_start` | — | One assistant response + its tool calls |
| `turn_end` | `message`, `tool_results` | |
| `message_start` | `message` | user, assistant, or tool result — see the note below |
| `message_update` | `block_index`, `delta`, `kind` | Assistant streaming only |
| `message_end` | `message` | |
| `tool_execution_start` | `tool_call_id`, `tool_name`, `arguments` | |
| `tool_execution_update` | `tool_call_id`, `tool_name`, `content` | Streaming partial results |
| `tool_execution_end` | `tool_call_id`, `tool_name`, `content`, `is_error`, `duration_ms` | |

### `message_start` on an assistant message is a placeholder

For an assistant message, `message_start.message` has **empty `content`**. The
assembled message does not exist until the provider's terminal event, so the
start event exists to tell you *when to open a buffer*, not what is in it.
`message_end` carries the authoritative message.

Tool-result `message_start` events are different — those carry the full message,
because it is already complete when emitted.

So do not read `message_start.message.content` for an assistant message.
Accumulate the `message_update` deltas, or read
`agent.state.streaming_message`.

### The user message is on the stream too

A run announces the user message that triggered it, right after `agent_start`.
Without it a subscriber sees only the assistant's side, and a session writer
records a transcript that cannot be resumed from — the model would be replayed
its own answers with none of the questions.

`Agent.prompt()` hands the message to the loop rather than appending it, so that
`_reduce` stays the single writer to `state.messages`. The consequence worth
knowing: **the event stream reproduces the transcript exactly**, in order. That
is what makes the session recorder a plain subscriber.

### Why `message_update` carries a delta

pi's equivalent event carries a full partial snapshot of the in-flight message.
We carry `delta: str` plus `block_index: int` instead — small and cheap to
serialize on every token.

Consumers that need the whole in-flight message read
`agent.state.streaming_message`, which the `Agent` maintains.

## Ordering guarantees

Two that matter when tools run in parallel:

- `tool_execution_end` fires in **completion** order.
- Tool-result *messages* append in **assistant source** order, because providers
  require results to match the order of the calls that requested them.

These differ on purpose: the first is for live UI feedback, the second is a
protocol requirement.

## Subscribers

```python
from harness.cli.json_sink import json_sink

agent.subscribe(json_sink())  # JSONL to stdout
agent.subscribe(json_sink(some_file))  # or any text stream
```

The sink is the whole implementation — about ten lines:

```python
async def write(event: AgentEvent) -> None:
    target.write(json.dumps(to_dict(event)) + "\n")
    target.flush()
```

It flushes per line so a piped consumer sees events as they happen rather than
when the process exits.

Listener exceptions are isolated — a failing subscriber must not break a run.
`tests/test_json_sink.py` proves it with a sink whose every write raises.
