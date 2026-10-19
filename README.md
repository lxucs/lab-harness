# lab-harness

**A minimal agent harness in Python, built for research.** Small enough to read
in an afternoon, complete enough to run real experiments on.

It runs a local agentic loop — the model reads and writes files, runs commands,
and searches the web, with whichever of those tools you give it — from an
interactive terminal, a one-shot command, or your own Python. Architecturally it follows
[pi](https://github.com/earendil-works/pi): the same layering, trimmed to the core logics.

- **Minimal.** The stateless agent loop is under 500 lines of code, in
  [`agent/loop.py`](lab_harness/agent/loop.py); the stateful `Agent` that wraps
  it — transcript, subscribers, abort — is under 150 lines, in
  [`agent/agent.py`](lab_harness/agent/agent.py). No other framework dependencies.
- **Every request reproducible.** A session records the system prompt, tools,
  model and config each request was sent with — including changes mid-session
  and requests a hook rewrote — so every request it made can be rebuilt exactly
  as the provider received it, for replay, analysis, or training data. See
  [trajectories](docs/sessions.md#trajectories).
- **One loop, many models.** Claude, OpenAI, DeepSeek, GLM, or any
  OpenAI-compatible endpoint, with reasoning carried across tool calls the way
  each vendor requires. Every endpoint and model is a table in
  [`llm/catalog.toml`](lab_harness/llm/catalog.toml), so a new vendor or model
  is a few lines of TOML, not code.
- **Observable.** Every step is a typed event of plain JSON data. `--json`
  streams them as JSONL, and `--log-level debug` traces each request, response
  and tool call by shape and id.
- **Hackable at two seams.** Four hooks and the event stream. A new tool is one
  file and one registry entry.
- **Tested offline.** A scripted provider drives the real loop with no network,
  and the suite blocks sockets so no test can reach one by accident.

> **Status: usable, and feature-complete against its design.** Three provider
> adapters, the full local tool set, an interactive terminal, layered config and
> durable sessions, all tested offline. Details in [Status](#status).

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
other layer. (Every layer reads the leaf modules `config.py` and `logs.py`,
which import only each other.)

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
```

| Layer | Responsibility |
|---|---|
| `llm/` | Provider-neutral LLM access — messages, models, streaming, tool declarations |
| `agent/` | The agentic loop and a stateful `Agent` that owns the transcript and emits events |
| `tools/` | Built-in capabilities and the registry that declares them to the model |
| `session/` | Append-only JSONL transcripts, resume, rebuilding requests — a subscriber, not a dependency |
| `prompts/` | The system prompt: the configured sections, the environment, the project's `AGENTS.md` — built once and handed to the agent as a string |
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
export TAVILY_API_KEY=tvly-...                        # optional, for web_search
uv run lab-harness                                    # interactive
uv run lab-harness -p "summarize the config module"   # one-shot
```

Interactive mode asks before `bash`, `write` or `edit` runs; `-p` has nobody to
ask, so they run unprompted — give it fewer tools with `--tools read,ls,grep,glob`
to rule them out. Sessions, JSONL events, other models, the system prompt and
debug logs: [examples/terminal.md](examples/terminal.md).

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
model, provider = route(None, config)  # $LAB_HARNESS_MODEL, else agent.model
tools = default_registry().select(["read", "ls", "grep", "glob"])
agent = Agent(model=model, provider=provider, tools=tools, config=config)
messages = asyncio.run(agent.prompt("What does this project do?"))
```

That agent has no system prompt. `build_system_prompt` from `lab_harness.prompts`
builds the one the CLI sends, as [quickstart.py](examples/quickstart.py) shows.

[examples/](examples/) goes further: streaming events, a custom tool, all four
hooks, a scripted model that needs no key, and exporting a session's requests
for training. Configuration and every environment variable:
[docs/configuration.md](docs/configuration.md).

## Documentation

| Page | Covers |
|---|---|
| [examples/](examples/) | Runnable usage, from the terminal and from Python |
| [architecture.md](docs/architecture.md) | Layering, module boundaries, deviations from pi |
| [events.md](docs/events.md) | The event protocol between runtime and frontend |
| [providers.md](docs/providers.md) | Provider contract, profiles, adding a vendor |
| [tools.md](docs/tools.md) | Tool contract, built-ins, adding a tool |
| [sessions.md](docs/sessions.md) | Transcript format, resume, rebuilding trajectories |
| [cli.md](docs/cli.md) | Terminal, flags, keybindings, the usage and cost readout |
| [configuration.md](docs/configuration.md) | Config files, environment variables, models and prices, prompts |
| [contributing.md](docs/contributing.md) | Code conventions, architectural invariants, config rules |

Contributing: [docs/contributing.md](docs/contributing.md). AI coding agents
should also read [AGENTS.md](AGENTS.md).

## Status

**Everything planned is built:** a tested runtime core, the full local tool set
(`read`/`write`/`edit`/`ls`/`bash`/`grep`/`glob`), `web_search`/`web_fetch` with
pluggable backends, three provider adapters (Messages, Responses, Chat
Completions), an interactive terminal with streamed, collapsible thinking,
colored edit diffs and a per-run token, cache and cost readout, a fail-closed
approval gate, a tool set chosen in config or per run, layered TOML config with
the prompts and the model catalog as data, a system prompt built from
configurable sections with the environment filled in and the project's
`HARNESS.md`/`AGENTS.md` appended, levelled diagnostics on stderr, and durable
sessions that resume on screen where they left off and record enough to
rebuild every request they made, changes mid-session included. A step-by-step
reading order is in [docs/architecture.md](docs/architecture.md#reading-order).

**Worth knowing before you use it:** `-p` one-shot mode has **no approval
gate** — there is nobody at the keyboard to answer, so `bash`, `write`, and
`edit` run unprompted unless `--tools` or `tools.enabled` leaves them out.
Interactive mode asks, and fails closed on anything but an explicit yes.

**Web search needs a key.** The default backend is
[Tavily](https://tavily.com) (free tier available); the keyless `ddgs`
alternative is implemented but did not return results from the development
machine — see [docs/tools.md](docs/tools.md#web-tools). `web_fetch` needs no
key either way.

### TODO

- **Context compaction.** The one planned piece still unbuilt.
  `prepare_request` is called before every provider request,
  `Model.context_window` and per-message `Usage` exist, and the session schema
  reserves a `compaction` entry — so it lands without touching the loop. Its
  rewrites are recorded already; the entry is for writing a summary once rather
  than the whole view with every request.
- **Anthropic prompt caching.** The Messages adapter sends no `cache_control`
  markers yet, so Anthropic caches nothing and its cache hit rate reads 0%.
  Contained to one function in `anthropic.py`; the placement is planned in
  [docs/providers.md](docs/providers.md#the-anthropic-adapter).

### Not planned

MCP, sub-agents, plan mode, RPC control channel, extensions-as-plugins,
sandboxing, and provider-native `web_search` — that last one would add a second
tool-execution path only one of the three adapters could ever use.

## License

[MIT](LICENSE).
