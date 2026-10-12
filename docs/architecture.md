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
    ┌─────┼─────┐
    ▼     ▼     ▼
  llm/  tools/  session/
    └─────┴─────┴──→  config.py, project.py, logs.py   (leaf modules)
```

`llm/` imports from no other *layer*. It does read `harness.config` — every
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
library/application split, and it is why importing `harness` installs no
handler on an embedding application's root logger — and why the module is
`logs.py` rather than `logging.py`, which a reader would have to think about
before trusting any `import logging` in the package.

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

**`session/`** — the append-only transcript on disk. The recorder is a
*subscriber*, so persistence is wiring rather than a branch in the loop: `cli/`
decides whether to attach one.

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
| `after_tool_call` | after execute, before events emit | truncation, redaction |
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
  `MessageStart`/`MessageEnd`. The message queue this replaces (below) got
  exactly that wrong.
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

**Prompt and tools are not in the transcript.** pi stores the system prompt and
tool declarations as system messages inside the transcript, replaying them to
derive current state. We pass `Context(system_prompt, messages, tools)` to the
provider directly. We lose mid-session tool-change provenance; the core gets
several hundred lines smaller. For a harness whose tool set is fixed at startup,
that is the right trade.

**One run at a time, no queue.** pi has separate steering and follow-up queues,
each with `all` and `one-at-a-time` drain modes, so a message can be injected
into a run already in flight. We have none of it: `prompt()` refuses while a run
is active. A frontend that wants to accept typing mid-run buffers the text
itself and prompts again when the run returns.

The first cut did have a queue, and review found it had quietly broken the
transcript: an injected message survived one request and then vanished from the
provider's view while remaining in `state.messages`, and a message queued on the
final turn was stranded entirely. Deleting the queue made both impossible by
construction rather than merely fixed — there is no second source of messages.
`prepare_request` went back to being what its docstring always claimed, a
per-request view transform.

**Dropped entirely:** the provider catalog, codemode, durable/RPC/server,
extensions-as-plugins, MCP, branch summarization.

## TODO, with a place to land

**Context compaction** is the one piece of the original design still unbuilt.
The seam it needs — `prepare_request` — already exists and is already called
before every provider request. Compaction reads the context, decides whether to
summarize, and returns a replacement; nothing in the loop changes when it
arrives. v1 also records `Usage` per assistant message, defines
`Model.context_window`, and reserves a `compaction` entry type in the session
schema, so the groundwork is laid.

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
| 8 | `cli/terminal.py` |
| 9 | `tools/shell.py`, `search.py`, `cli/approval.py` |
| 10 | `session/` |
| 11 | `tools/web.py`, `html_text.py`, `config.py` loading |
| 12 | `llm/providers/chat_completions.py`, `responses.py` |
