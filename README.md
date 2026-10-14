# lab-harness

**A minimal agent harness in Python, built for research.** Small enough to read
in an afternoon, complete enough to run real experiments on.

It runs a local agentic loop — the model reads files, runs commands, edits
code, and searches the web — from an interactive terminal, a one-shot command,
or your own Python. Architecturally it follows
[pi](https://github.com/earendil-works/pi): the same layering, trimmed to the
parts that earn their keep.

- **Minimal.** The stateless agent loop is ~470 lines of code, in
  [`agent/loop.py`](lab_harness/agent/loop.py); the stateful `Agent` that wraps
  it — transcript, subscribers, abort — is ~130, in
  [`agent/agent.py`](lab_harness/agent/agent.py). Not counting comments and
  docstrings, which explain the why. Five runtime dependencies, no agent
  framework.
- **Every request reproducible.** A session records the system prompt, tools,
  model and config each invocation ran with, so every request it made can be
  rebuilt exactly as the provider received it — for replay, analysis, or
  training data. See [trajectories](docs/sessions.md#trajectories).
- **One loop, many models.** Claude, OpenAI, DeepSeek, GLM, or any
  OpenAI-compatible endpoint, with reasoning carried across tool calls the way
  each vendor requires. A new vendor is a config section, not code.
- **Observable.** Every step is a typed event of plain JSON data. `--json`
  streams them as JSONL, and `--log-level debug` traces each request, response
  and tool call by shape and id.
- **Hackable at two seams.** Four hooks and the event stream. A new tool is one
  file and one registry entry.
- **Tested offline.** A scripted provider drives the real loop with no network
  — 872 tests, sockets blocked.

> **Status: usable, and feature-complete against its design.** Three provider
> adapters, the full local tool set, an interactive terminal, layered config and
> durable sessions — 872 tests, all offline. Details in
> [Status](#status).

## Why

Agent research needs two things a product harness rarely gives you: the exact
request behind every response, and the freedom to change one piece — a tool, a
hook, the provider — without forking the rest. Most harnesses are either too
minimal to grow or too large to understand. This one optimizes for the middle:
a core you can hold in your head, with the seams already in the right places so
that adding a tool, a provider, or a frontend is a small, local change rather
than a refactor.

## Architecture

Three layers. Dependencies point strictly downward — `llm/` imports from no
other layer. (Every layer reads the leaf modules `config.py`, `project.py` and
`logs.py`, which import nothing from the package themselves.)

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
| `session/` | Append-only JSONL transcripts, resume, rebuilding requests — a subscriber, not a dependency |
| `cli/` | Terminal and JSONL subscribers to the event stream |

Two seams carry almost everything extensible:

**The event stream.** `Agent.subscribe(listener)` is the only way to observe a
run. The terminal is a subscriber; so is the JSONL writer. Every event is a
frozen dataclass of JSON primitives, enforced by a round-trip test — which is
what lets a frontend in another language be a third subscriber later.

**The hooks.** `before_tool_call`, `after_tool_call`, `prepare_request`, and
`finish_turn`. A feature that fits a hook does not belong in the loop — tool
approval is one, which is why `-p` and interactive modes differ without the
loop branching. Context compaction will be another.

Details in [docs/architecture.md](docs/architecture.md).

## Quick start

```bash
uv sync
export ANTHROPIC_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...       # optional, for web_search
uv run lab-harness                   # interactive
```

```bash
uv run lab-harness -p "summarize the config module"   # one-shot
uv run lab-harness -p "..." --json                    # JSONL event stream
uv run lab-harness --resume                           # continue where you left off
uv run lab-harness --sessions                         # list saved sessions here
uv run lab-harness -p "..." --model claude-opus-5
uv run lab-harness --cwd ../other-project
uv run lab-harness --log-level debug             # diagnostics on stderr
uv run lab-harness --model deepseek-flash        # routed to DeepSeek by config
uv run lab-harness --model gpt-5                 # OpenAI, via the Responses API
```

In interactive mode, `Ctrl-C` cancels the turn in flight without ending the
session, `Ctrl-O` expands or collapses the model's thinking, and `Ctrl-D`
exits. Reasoning streams live — DeepSeek's and GLM's raw chain, OpenAI's and
Claude's summaries — and collapses to one line when its answer starts. Edits
show as red and green diffs, Claude Code style, both when they ask for approval
and once they land. Mutating tools (`bash`, `write`, `edit`) ask before
running. Conversations are saved per project under `~/.lab-harness/sessions/` and
`--resume` picks the last one back up; `--no-save` skips persistence. A saved
session records the system prompt, tools, model and config each invocation ran
with, so every request it made can be rebuilt — for replay or training. Exit
status is `0` on a clean run, `1` when the final turn errored or was aborted.

A `HARNESS.md` or `AGENTS.md` in the working directory is appended to the
system prompt, so a project's conventions do not have to be rediscovered every
run. Nothing to configure — if the file is there, it is used.

Configuration and every environment variable: [docs/configuration.md](docs/configuration.md).

## From Python

The CLI is a thin layer over a library, so an experiment can drive the agent
directly:

```python
import asyncio

from lab_harness.agent import Agent
from lab_harness.config import load_config
from lab_harness.llm.providers import route
from lab_harness.tools import default_registry

config = load_config()
model, provider = route("claude-sonnet-5", config)
agent = Agent(model=model, provider=provider, tools=tuple(default_registry()), config=config)


async def on_event(event):  # every delta, tool call and result, as it happens
    print(event.type)


agent.subscribe(on_event)
messages = asyncio.run(agent.prompt("What does this project do?"))
```

`agent.state.messages` is the whole transcript. Nothing is written to disk
unless you subscribe a recorder — `lab_harness.session.start_recorder`, as
`cli/main.py` does — after which
[`recorded_requests`](docs/sessions.md#trajectories) rebuilds every request
from the file. To script the model instead, `lab_harness.llm.fake.FakeProvider`
replays canned responses through the same loop; that is how the test suite runs
offline. Events: [docs/events.md](docs/events.md). Hooks:
[docs/architecture.md](docs/architecture.md).

## Documentation

| Page | Covers |
|---|---|
| [architecture.md](docs/architecture.md) | Layering, module boundaries, deviations from pi |
| [events.md](docs/events.md) | The event protocol between runtime and frontend |
| [providers.md](docs/providers.md) | Provider contract, profiles, adding a vendor |
| [tools.md](docs/tools.md) | Tool contract, built-ins, adding a tool |
| [sessions.md](docs/sessions.md) | Transcript format, resume, rebuilding trajectories |
| [cli.md](docs/cli.md) | Terminal, flags, keybindings, the usage and cost readout |
| [configuration.md](docs/configuration.md) | Config files, environment variables, pricing |
| [contributing.md](docs/contributing.md) | Code conventions, architectural invariants, config rules |

The full design rationale — including what was cut and why — is in
[the design spec](docs/superpowers/specs/2026-10-09-lab-harness-runtime-design.md).

Contributing: [docs/contributing.md](docs/contributing.md). AI coding agents
should also read [AGENTS.md](AGENTS.md).

## Status

**Everything planned is built:** a tested runtime core, the full local tool set
(`read`/`write`/`edit`/`ls`/`bash`/`grep`/`glob`), `web_search`/`web_fetch` with
pluggable backends, three provider adapters (Messages, Responses, Chat
Completions), an interactive terminal with streamed, collapsible thinking,
colored edit diffs and a per-run token, cache and cost readout, a fail-closed
approval gate, layered TOML config, project instructions from
`HARNESS.md`/`AGENTS.md`, levelled diagnostics on stderr, and durable sessions
that record enough to rebuild every request they made. A step-by-step reading
order is in
[docs/architecture.md](docs/architecture.md#reading-order).

**Worth knowing before you use it:** `-p` one-shot mode has **no approval
gate** — there is nobody at the keyboard to answer, so `bash`, `write`, and
`edit` run unprompted. Interactive mode asks, and fails closed on anything but
an explicit yes.

**Web search needs a key.** The default backend is
[Tavily](https://tavily.com) (free tier available); the keyless `ddgs`
alternative is implemented but did not return results from the development
machine — see [docs/tools.md](docs/tools.md#web-tools). `web_fetch` needs no
key either way.

### TODO

- **Context compaction.** The one piece of the original design still unbuilt.
  `prepare_request` is called before every provider request,
  `Model.context_window` and per-message `Usage` exist, and the session schema
  reserves a `compaction` entry — so it lands without touching the loop. That
  entry has to record the summary, or rebuilt trajectories would show the
  uncompacted history.
- **Anthropic prompt caching.** The Messages adapter sends no `cache_control`
  markers yet, so Anthropic caches nothing and its cache hit rate reads 0%.
  Contained to one function in `anthropic.py`; the placement is planned in
  [docs/providers.md](docs/providers.md#the-anthropic-adapter).

### Not planned

MCP, sub-agents, plan mode, RPC control channel, extensions-as-plugins,
sandboxing, and provider-native `web_search` — that last one would add a second
tool-execution path only one of the three adapters could ever use.

## License

Not yet chosen.
