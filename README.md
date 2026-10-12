# harness

A lightweight agent harness. Python runtime, terminal frontend.

It runs a local agentic loop — the model reads files, runs commands, edits code,
and searches the web — and it is small enough to read in an afternoon.
Architecturally it follows [pi](https://github.com/earendil-works/pi): the same
layering, trimmed to the parts that earn their keep.

> **Status: usable.** The agentic loop, the full local tool set, web search and
> fetch, the Anthropic provider, an interactive terminal, tool approval,
> layered config, and session persistence are implemented and tested (431
> tests, offline). A second provider is next — see [Progress](#progress).

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
| `session/` | Append-only JSONL transcripts, resume — a subscriber, not a dependency |
| `cli/` | Terminal and JSONL subscribers to the event stream |

Two seams carry almost everything extensible:

**The event stream.** `Agent.subscribe(listener)` is the only way to observe a
run. The terminal is a subscriber; so is the JSONL writer. Every event is a
frozen dataclass of JSON primitives, enforced by a round-trip test — which is
what lets a frontend in another language be a third subscriber later.

**The hooks.** `before_tool_call`, `after_tool_call`, and `prepare_request`. A
feature that fits a hook does not belong in the loop — tool approval is one,
which is why `-p` and interactive modes differ without the loop branching.
Context compaction will be another.

Details in [docs/architecture.md](docs/architecture.md).

## Quick start

```bash
uv sync
export ANTHROPIC_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...       # optional, for web_search
uv run harness                       # interactive
```

```bash
uv run harness -p "summarize the config module"   # one-shot
uv run harness -p "..." --json                    # JSONL event stream
uv run harness --resume                           # continue where you left off
uv run harness --sessions                         # list saved sessions here
uv run harness -p "..." --model claude-opus-5
uv run harness --cwd ../other-project
```

In interactive mode, `Ctrl-C` cancels the turn in flight without ending the
session and `Ctrl-D` exits. Mutating tools (`bash`, `write`, `edit`) ask before
running. Conversations are saved per project under `~/.harness/sessions/` and
`--resume` picks the last one back up; `--no-save` skips persistence. Exit
status is `0` on a clean run, `1` when the final turn errored or was aborted.

Configuration and every environment variable: [docs/configuration.md](docs/configuration.md).

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

Steps 1–11 complete: a tested runtime core, the full local tool set
(`read`/`write`/`edit`/`ls`/`bash`/`grep`/`glob`), `web_search`/`web_fetch`
with pluggable backends, the Anthropic adapter, an interactive terminal, a
fail-closed approval gate, layered TOML config, and durable sessions.

Next: the OpenAI-compatible provider (step 12).

### Planned

Build order. Each step is independently verifiable.

- [x] 1 · `llm/` core types, event protocol, serialization tests
- [x] 2 · `FakeProvider` and the `Provider` protocol
- [x] 3 · `agent/loop.py` — the agentic loop
- [x] 4 · `agent/agent.py` — state, subscribers, abort
- [x] 5 · `tools/` base, registry, filesystem tools
- [x] 6 · `cli/json_sink.py` — first end-to-end run
- [x] 7 · Anthropic provider — first real run
- [x] 8 · `cli/terminal.py` — interactive terminal
- [x] 9 · `bash`, `grep`, `glob`, tool approval gate
- [x] 10 · `session/` — persistence and resume
- [x] 11 · Web search and fetch, layered TOML config
- [ ] 12 · OpenAI-compatible provider

**Worth knowing before you use it:** `-p` one-shot mode has **no approval
gate** — there is nobody at the keyboard to answer, so `bash`, `write`, and
`edit` run unprompted. Interactive mode asks, and fails closed on anything but
an explicit yes.

**Web search needs a key.** The default backend is
[Tavily](https://tavily.com) (free tier available); the keyless `ddgs`
alternative is implemented but did not return results from the development
machine — see [docs/tools.md](docs/tools.md#web-tools). `web_fetch` needs no
key either way. Forwarding search to Anthropic's server-side tool, which would
need no key at all, is designed but not built.

### Deferred

- **Context compaction.** Cut from v1 deliberately. `prepare_request` is called
  before every provider request, `Model.context_window` and per-message `Usage`
  exist, and the session schema reserves a `compaction` entry — so it lands
  later without touching the loop.
- **Not planned:** MCP, sub-agents, plan mode, RPC control channel,
  extensions-as-plugins, sandboxing, a multi-provider catalog.

## License

Not yet chosen.
