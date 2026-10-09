# harness

A lightweight agent harness. Python runtime, terminal frontend.

It runs a local agentic loop — the model reads files, runs commands, edits code,
and searches the web — and it is small enough to read in an afternoon.
Architecturally it follows [pi](https://github.com/earendil-works/pi): the same
layering, trimmed to the parts that earn their keep.

> **Status: pre-implementation.** The design is settled and documented; code is
> being written against the build order below. See
> [In progress](#in-progress) and [Planned](#planned).

## Why

Most agent harnesses are either too minimal to grow or too large to understand.
This one optimizes for a specific middle: a core you can hold in your head,
with the seams already in the right places so that adding a tool, a provider, or
a frontend is a small, local change rather than a refactor.

## Architecture

Three layers. Dependencies point strictly downward — `llm/` has no upward
dependencies at all.

```
         cli/          terminal, json sink  (subscribers)
          │
          ▼
        agent/         the loop, Agent, events
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
  llm/  tools/  session/
```

| Layer | Responsibility |
|---|---|
| `llm/` | Provider-neutral LLM access — messages, models, streaming, tool declarations |
| `agent/` | The agentic loop and a stateful `Agent` that owns the transcript and emits events |
| `tools/` | Built-in capabilities and the registry that declares them to the model |
| `session/` | Append-only JSONL transcripts, resume |
| `cli/` | Terminal and JSONL subscribers to the event stream |

Two seams carry almost everything extensible:

**The event stream.** `Agent.subscribe(listener)` is the only way to observe a
run. The terminal is a subscriber; so is the JSONL writer. Every event is a
frozen dataclass of JSON primitives, enforced by a round-trip test — which is
what lets a frontend in another language be a third subscriber later.

**The hooks.** `before_tool_call`, `after_tool_call`, `prepare_request`, and
`finish_turn`. A feature that fits a hook does not belong in the loop. Tool
approval is a hook; context compaction will be one.

Details in [docs/architecture.md](docs/architecture.md).

## Quick start

```bash
uv sync
uv run harness
```

```bash
uv run harness -p "summarize the config module"   # one-shot
uv run harness -p "..." --json                    # JSONL event stream
uv run harness --resume                           # continue last session
```

Configure a provider with `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`. See
[docs/configuration.md](docs/configuration.md).

## Documentation

| Page | Covers |
|---|---|
| [architecture.md](docs/architecture.md) | Layering, module boundaries, deviations from pi |
| [events.md](docs/events.md) | The event protocol between runtime and frontend |
| [providers.md](docs/providers.md) | Provider contract, adding a provider |
| [tools.md](docs/tools.md) | Tool contract, built-ins, adding a tool |
| [sessions.md](docs/sessions.md) | Transcript format, resume |
| [cli.md](docs/cli.md) | Terminal usage, flags, keybindings |
| [configuration.md](docs/configuration.md) | Config files and environment variables |
| [contributing.md](docs/contributing.md) | Code conventions, architectural invariants, config rules |

The full design rationale — including what was cut and why — is in
[the design spec](docs/superpowers/specs/2026-10-09-harness-runtime-design.md).

Contributing: [docs/contributing.md](docs/contributing.md). AI coding agents
should also read [AGENTS.md](AGENTS.md).

## Progress

### In progress

Nothing yet — implementation continues at step 2.

### Planned

Build order. Each step is independently verifiable.

- [x] 1 · `llm/` core types, event protocol, serialization tests
- [ ] 2 · `FakeProvider` and the `Provider` protocol
- [ ] 3 · `agent/loop.py` — the agentic loop
- [ ] 4 · `agent/agent.py` — state, subscribers, abort, message queue
- [ ] 5 · `tools/` base, registry, filesystem tools
- [ ] 6 · `cli/json_sink.py` — first end-to-end run
- [ ] 7 · Anthropic provider — first real run
- [ ] 8 · `cli/terminal.py` — interactive terminal
- [ ] 9 · `bash`, `grep`, `glob`, tool approval gate
- [ ] 10 · `session/` — persistence and resume
- [ ] 11 · Web search and fetch
- [ ] 12 · OpenAI-compatible provider

### Deferred

- **Context compaction.** Cut from v1 deliberately. `prepare_request` is called
  before every provider request, `Model.context_window` and per-message `Usage`
  exist, and the session schema reserves a `compaction` entry — so it lands
  later without touching the loop.
- **Not planned:** MCP, sub-agents, plan mode, RPC control channel,
  extensions-as-plugins, sandboxing, a multi-provider catalog.

## License

Not yet chosen.
