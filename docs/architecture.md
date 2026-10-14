# Architecture

## Layering

Three layers, as subpackages of one installable `lab_harness` package.
Dependencies point strictly downward.

```
         cli/          terminal, json sink  (subscribers)
          │
          ▼
        agent/         the loop, Agent, events
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
  llm/  tools/  session/
    └─────┴─────┴──→  config.py, project.py, logs.py   (leaf modules)
```

`llm/` imports from no other *layer*. It does read `lab_harness.config` — every
layer does. The leaf modules import nothing from the package except each other
and `config.py`, so depending on one cannot create a cycle and does not make a
layer less replaceable. Treat "no upward dependencies" as being about the
boxes, not about these three.

`project.py` loads a project's `HARNESS.md` or `AGENTS.md`. It lives beside
`config.py` rather than inside `agent/` because it is the same kind of thing —
read a directory, return data — and because `agent/` is not supposed to know
the filesystem layout.

`logs.py` decides where diagnostics go and how loud they are. Every layer takes
a logger the ordinary way (`logging.getLogger(__name__)`) and emits records
unconditionally; only `cli/` calls `configure_logging`. That is the stdlib's
library/application split, and it is why importing `lab_harness` installs no
handler on an embedding application's root logger — and why the module is
`logs.py` rather than `logging.py`, which a reader would have to think about
before trusting any `import logging` in the package. The interactive terminal
hands it a stream of its own, `LogStream`, so records print above the live
region instead of through it.

### What each layer knows

**`llm/`** — messages, models, tool *declarations*, and how to talk to a
provider. It does not know that tools are executable, that a filesystem exists,
or that there is a UI.

**`agent/`** — the loop, tools-as-executables, and the event stream. It does not
know the filesystem layout or anything about rendering.

**`tools/`** — individual capabilities, *and the tool contract itself*. A tool
receives a `ToolContext` with its `cwd`, resolved config, a cancel token, and an
update callback. Tools never reach for globals.

`AgentTool`, `ToolResult`, and `ToolContext` are defined in `tools/base.py`
rather than in `agent/`, precisely because of the rule above: `agent/` consumes
tools, so a definition in `agent/` would force every tool module to import
upward. `agent/types.py` re-exports them for callers.

**`session/`** — the append-only transcript on disk, and the reader that
rebuilds every request from it. The recorder is a *subscriber*, so persistence
is wiring rather than a branch in the loop: `cli/` decides whether to attach
one, and hands it the prompt, tools and config to record.

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
| `before_tool_call` | after args validate, before execute | approval gate (`cli/approval.py`) |
| `after_tool_call` | after execute, before events emit | truncation, redaction — of `content` and `details` both |
| `prepare_request` | before every provider request | compaction (TODO) |
| `finish_turn` | when a turn would cleanly end the run | auto-continue |

Hooks are optional async callables. A feature that fits a hook does not belong
in the loop.

**`finish_turn`** returns `FinishTurnResult(continue_with=…)` to append a user
message and keep going, or `None` to let the run end. Three properties make it
safe to hand to arbitrary code:

- It sits inside the `max_turns` loop, so auto-continue is bounded by
  construction — a hook that always continues wastes a budget, it does not hang.
- The injected message is appended to the transcript *and* announced through
  `MessageStart`/`MessageEnd`.
- A hook that raises ends the run rather than continuing — the mirror of the
  approval gate failing closed. Treating a crash as "continue" would burn the
  whole budget on code already known to be broken.

It is not consulted for a turn that errored or was aborted: "try again" is a
decision for whoever handles the error.

## Relationship to pi

This follows [pi](https://github.com/earendil-works/pi)'s layering closely:
`llm/ ≈ pi-ai`, `agent/ ≈ pi-agent-core`, and the two-tier split of a pure loop
function plus a stateful `Agent` wrapper is pi's.

Deliberate deviations:

**Prompt and tools are recorded, not replayed.** pi stores the system prompt
and tool declarations as system messages inside the transcript, replaying them
to derive current state. We pass `Context(system_prompt, messages, tools)` to
the provider directly, and the session writes them down once per invocation as
a `context` entry — a record, never read back into the agent. That keeps
enough to rebuild every request (see [sessions.md](sessions.md#trajectories))
while the core stays several hundred lines smaller. A tool set that changes
*mid*-invocation would go unrecorded, but ours is fixed at startup.

**One run at a time, no queue.** pi has separate steering and follow-up queues,
each with `all` and `one-at-a-time` drain modes, so a message can be injected
into a run already in flight. We have none of it: `prompt()` refuses while a run
is active. A frontend that wants to accept typing mid-run buffers the text
itself and prompts again when the run returns.

**Dropped entirely:** codemode, durable/RPC/server, extensions-as-plugins, MCP,
branch summarization.

**pi's generated provider catalog** shrinks to `llm/catalog.toml`: a hand-kept
table of the endpoints we ship and the models they serve, in config-file
format, so adding a model is a few lines of TOML and a config file can add or
override one the same way.

## TODO, with a place to land

**Context compaction** is the one piece of the original design still unbuilt.
The seam it needs — `prepare_request` — already exists and is already called
before every provider request. Compaction reads the context, decides whether to
summarize, and returns a replacement; nothing in the loop changes when it
arrives. The runtime already records `Usage` per assistant message, defines
`Model.context_window`, and reserves a `compaction` entry type in the session
schema, so the groundwork is laid. The entry is not optional when it lands:
`prepare_request` rewrites what is sent without touching the transcript, so
without a record of the summary, rebuilt
[trajectories](sessions.md#trajectories) would show the uncompacted history.

See §7 of the design spec for the full plan.

## Reading order

All of it is built. This is the order it was built in, which is also the least
confusing order to read it in — each step only depends on the ones above.

| # | Where |
|---|---|
| 1 | `llm/types.py`, `events.py` — the plain-data core |
| 2 | `llm/fake.py`, `provider.py` — the seam every test uses |
| 3 | `agent/loop.py` — the loop itself |
| 4 | `agent/agent.py` — state, subscribers, abort |
| 5 | `tools/base.py`, `registry.py`, `fs.py` |
| 6 | `cli/json_sink.py` — first end-to-end run |
| 7 | `llm/providers/anthropic.py` — first real run |
| 8 | `cli/terminal.py`, `usage.py` |
| 9 | `tools/shell.py`, `search.py`, `cli/approval.py` |
| 10 | `session/` |
| 11 | `tools/web.py`, `html_text.py`, `config.py` loading |
| 12 | `llm/providers/chat_completions.py`, `responses.py` |
