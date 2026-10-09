# Architecture

## Layering

Three layers, as subpackages of one installable `harness` package.
Dependencies point strictly downward.

```
         cli/          terminal, json sink  (subscribers)
          │
          ▼
        agent/         the loop, Agent, events
          │
    ┌─────┼─────┬──────────┐
    ▼     ▼     ▼          ▼
  llm/  tools/  session/  (context/ — deferred)
```

`llm/` has no upward dependencies at all.

### What each layer knows

**`llm/`** — messages, models, tool *declarations*, and how to talk to a
provider. It does not know that tools are executable, that a filesystem exists,
or that there is a UI.

**`agent/`** — the loop, tools-as-executables, and the event stream. It does not
know the filesystem layout or anything about rendering.

**`tools/`** — individual capabilities. A tool receives a `ToolContext` with its
`cwd`, a cancel token, and an update callback. Tools never reach for globals.

**`session/`** — the append-only transcript on disk.

**`cli/`** — a *subscriber* to the agent's event stream. The runtime never
imports from `cli/`. Deleting `cli/` leaves the runtime working.

If you need an upward import, the abstraction is in the wrong place. Add a hook
or inject the dependency instead.

## The two seams

Almost everything extensible in this codebase goes through one of two seams.

### 1. The event stream

`Agent.subscribe(listener)` is the only way the outside world observes a run.
The terminal is a subscriber. The JSONL writer is a subscriber. A future TUI in
another language would be a third, reading the serialized form.

This works only if events stay plain data — see [events.md](events.md).

### 2. The hooks

| Hook | When | Used by |
|---|---|---|
| `before_tool_call` | after args validate, before execute | approval gate |
| `after_tool_call` | after execute, before events emit | truncation, redaction |
| `prepare_request` | before every provider request | compaction (later) |
| `finish_turn` | after a turn's results, before `turn_end` | auto-continue (later) |

Hooks are optional async callables. A feature that fits a hook does not belong
in the loop.

## Relationship to pi

This follows [pi](https://github.com/earendil-works/pi)'s layering closely:
`llm/ ≈ pi-ai`, `agent/ ≈ pi-agent-core`, and the two-tier split of a pure loop
function plus a stateful `Agent` wrapper is pi's.

Deliberate deviations:

**Prompt and tools are not in the transcript.** pi stores the system prompt and
tool declarations as system messages inside the transcript, replaying them to
derive current state. We pass `Context(system_prompt, messages, tools)` to the
provider directly. We lose mid-session tool-change provenance; the core gets
several hundred lines smaller. For a harness whose tool set is fixed at startup,
that is the right trade.

**One message queue.** pi has separate steering and follow-up queues, each with
`all` and `one-at-a-time` drain modes. We have one queue drained at turn
boundaries, which covers "type while it's working" at a fraction of the cost.

**Dropped entirely:** the provider catalog, codemode, durable/RPC/server,
extensions-as-plugins, MCP, branch summarization.

## Deferred, with a place to land

**Context compaction** is not in v1. The seam it needs — `prepare_request` — is.
Compaction reads the context, decides whether to summarize, and returns a
replacement; nothing in the loop changes when it arrives. v1 also records
`Usage` per assistant message, defines `Model.context_window`, and reserves a
`compaction` entry type in the session schema, so the groundwork is laid.

See §7 of the design spec for the full plan.

## Build order

Each step is independently verifiable. Status is tracked here as implementation
proceeds.

| # | Step | Status |
|---|---|---|
| 1 | `llm/types.py`, `events.py`, serialization tests | Complete |
| 2 | `llm/fake.py`, `provider.py` | Not started |
| 3 | `agent/loop.py` against `FakeProvider` | Not started |
| 4 | `agent/agent.py` — state, subscribers, abort, queue | Not started |
| 5 | `tools/base.py`, `registry.py`, `fs.py` | Not started |
| 6 | `cli/json_sink.py` → first end-to-end run | Not started |
| 7 | `llm/providers/anthropic.py` → first real run | Not started |
| 8 | `cli/terminal.py` — interactive | Not started |
| 9 | `tools/shell.py`, `search.py`, approval gate | Not started |
| 10 | `session/` | Not started |
| 11 | `tools/web.py`, `native.py` | Not started |
| 12 | `llm/providers/openai.py` | Not started |
