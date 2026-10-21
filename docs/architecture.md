# Architecture

## Layering

Three layers, as subpackages of one installable `lab_harness` package.
Dependencies point strictly downward.

```
         cli/          terminal, json sink  (subscribers)
          │
          ├─────────────────┐
          ▼                 ▼
        agent/            prompts/     the system prompt
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
  llm/  tools/  session/
    └─────┴─────┴──→  config.py, logs.py   (leaf modules)
```

`llm/` imports from no other *layer*. It does read `lab_harness.config` — every
layer does. The leaf modules import nothing from the package except each other,
so depending on one cannot create a cycle and does not make a layer less
replaceable. Treat "no upward dependencies" as being about the boxes, not about
these two.

`prompts/` sits beside `agent/`, not under it. `cli/` builds the system prompt
and hands `agent/` the finished string, so the runtime never reads a project's
files or the machine it runs on — `agent/` is not supposed to know the
filesystem layout. A Python caller builds the prompt the same way.

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

**`prompts/`** — what the model is told besides the conversation:
`prompts.toml` holds the text, `environment.py` reads where the agent runs,
`project.py` loads a project's `HARNESS.md` or `AGENTS.md`, and `system.py`
fills in and joins the sections. It imports only `config.py` and returns a
string; nothing but `cli/` and a Python caller imports it.

**`session/`** — the append-only transcript on disk, and the reader that
rebuilds every request from it. The recorder is a *subscriber*, so persistence
is wiring rather than a branch in the loop: `cli/` decides whether to attach
one, and hands it the prompt, tools and config to record.

**`cli/`** — a *subscriber* to the agent's event stream. The runtime never
imports from `cli/`. Deleting `cli/` leaves the runtime working. A resumed
session's history is the one thing it draws from `agent.state` rather than the
stream: the past is not a run, and sent through the stream it would be
recorded a second time.

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
the provider directly, and the session writes them down as a `context` entry —
a record, never read back into the agent. Each run's `agent_start` event
carries all three, so the recorder writes a new entry whenever one changes
between runs, and a `prepare_request` hook's rewrite of a single request is an
event and an entry of its own. That keeps enough to rebuild every request (see
[sessions.md](sessions.md#trajectories)) while the core stays several hundred
lines smaller.

**Prompts are data.** The system prompt pi assembles in code is
`lab_harness/prompts/prompts.toml` here, the default of the `[prompts]` config
section; code only fills in the variables and joins the sections, in
`prompts/system.py`. It names no tools: the declarations alone tell the model
what it can call, so the prompt stays true for any tool set.

**One run at a time, no queue.** pi has separate steering and follow-up queues,
each with `all` and `one-at-a-time` drain modes, so a message can be injected
into a run already in flight. We have none of it: `prompt()` refuses while a run
is active. A frontend that wants to accept typing mid-run buffers the text
itself and prompts again when the run returns.

**Dropped entirely:** codemode, durable/RPC/server, extensions-as-plugins, MCP,
branch summarization.

**pi's generated provider catalog** shrinks to `llm/catalog.toml`: a hand-kept
file of every LLM setting we ship — the endpoints, the models they serve, and
their defaults — in config-file format, so adding a model is a few lines of
TOML and a config file can add or override one the same way.

## TODO, with a place to land

**Context compaction** is the one planned piece still unbuilt.
The seam it needs — `prepare_request` — already exists and is already called
before every provider request. Compaction reads the context, decides whether to
summarize, and returns a replacement; nothing in the loop changes when it
arrives. The runtime already records `Usage` per assistant message, defines
`Model.context_window`, and reserves a `compaction` entry type in the session
schema, so the groundwork is laid. Rebuilt
[trajectories](sessions.md#trajectories) would already be right, since every
rewrite `prepare_request` makes is recorded — but as the whole view, on every
request. The `compaction` entry is for writing a summary once.

The plan is a `context/` package of two modules, `tokens.py` and
`compaction.py`: past a threshold, summarize the oldest turns with a cheap
model, splice in a `CompactionSummary` message, keep the most recent turns, and
never cut between a tool call and its result. It should stay small — pi's
version also covers branch summarization and structured artifact indexes,
which are not planned here.

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
| 8 | `cli/terminal.py`, `commands.py`, `usage.py`, `export.py` |
| 9 | `tools/shell.py`, `search.py`, `cli/approval.py` |
| 10 | `session/` |
| 11 | `tools/web.py`, `html_text.py`, `config.py` loading |
| 12 | `llm/providers/chat_completions.py`, `responses.py` |
| 13 | `prompts/` — the system prompt |
| 14 | `tools/skills.py` — skills, listed by name and loaded on demand |
