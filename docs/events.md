# Events

The event stream is the contract between the runtime and anything watching it.
`Agent.subscribe(listener)` is the only way to observe a run.

> **Status:** The provider stream protocol (`StreamEvent` and its members) is
> implemented in `harness/llm/events.py` as of Task 1. Agent-level events
> (`AgentEvent` and the subscriber model described below) are implemented in
> `harness/agent/types.py` as of Task 3.

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
  turn_start
    message_start   (assistant)
    message_update  × N        ← streaming deltas
    message_end
    tool_execution_start  × N
    tool_execution_update × N  ← e.g. bash output as it arrives
    tool_execution_end    × N
    message_start/end     × N  ← tool results, in assistant source order
  turn_end
  …more turns while tool calls or queued messages remain…
agent_end
```

## Event types

| Event | Fields | Notes |
|---|---|---|
| `agent_start` | — | One per run |
| `agent_end` | `messages` | Last event of a run |
| `turn_start` | — | One assistant response + its tool calls |
| `turn_end` | `message`, `tool_results` | |
| `message_start` | `message` | system, user, assistant, or tool result |
| `message_update` | `block_index`, `delta`, `kind` | Assistant streaming only |
| `message_end` | `message` | |
| `tool_execution_start` | `tool_call_id`, `tool_name`, `arguments` | |
| `tool_execution_update` | `tool_call_id`, `tool_name`, `content` | Streaming partial results |
| `tool_execution_end` | `tool_call_id`, `tool_name`, `content`, `is_error`, `duration_ms` | |

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
agent.subscribe(terminal.on_event)  # rich rendering
agent.subscribe(json_sink)  # JSONL to stdout
```

The JSONL sink is about thirty lines:

```python
def json_sink(event: AgentEvent) -> None:
    print(json.dumps(event.to_dict()), flush=True)
```

Listener exceptions are isolated — a failing subscriber must not break a run.
