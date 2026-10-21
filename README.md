# lab-harness

[![CI](https://github.com/lxucs/lab-harness/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lxucs/lab-harness/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**A minimal agent harness in Python, built for study and research.** Small
enough to read in an afternoon, complete enough to run full experiments on.

A lightweight Python alternative to [pi](https://github.com/earendil-works/pi),
with the same layering: `llm/` for pi-ai, `agent/` for pi-agent-core, a pure
loop wrapped by a stateful `Agent`. A hands-on option for running an agent
harness locally: read it end to end, change it in place, and see exactly what
it sends to the model.

At its core is a local agentic loop: the model reads and writes files, runs
commands and searches the web, with whichever of those tools you give it. On
top of it:

- **An interactive terminal**, to work with the agent directly — thinking and
  tool calls as they stream, a token, cache and cost readout, and sessions that
  resume and export as a web page.
- **A Python API**, to build on it — `from lab_harness.agent import Agent` is
  the whole integration. [`search_agent/`](search_agent/) is a complete search
  agent built that way.

A run in the terminal:

<img src="https://github.com/lxucs/lab-harness/raw/main/img/demo.gif" width="600" alt="lab-harness in the terminal: a question answered from a handful of documents, one tool call at a time">

The [trajectory report](https://lxucs.github.io/lab-harness/search_agent/runs/793/report.html) a run records — the whole agentic trace for easy reading:

<a href="https://lxucs.github.io/lab-harness/search_agent/runs/793/report.html"><img src="https://github.com/lxucs/lab-harness/raw/main/img/report.png" width="600" alt="The trajectory report of a search agent run: turns, requests, tool calls, tokens, cache and cost, then every step"></a>

## Features

Agent research needs what a product harness rarely gives: the exact request
behind every response, and the freedom to swap one piece — a tool, a hook, a
provider — without forking the rest. lab-harness keeps the core small enough to
hold in your head, with the seams where those changes go.

- **Minimal.** The stateless agent loop is about 500 lines of code, in
  [`agent/loop.py`](lab_harness/agent/loop.py); the stateful `Agent` that wraps
  it — transcript, subscribers, abort — about 150, in
  [`agent/agent.py`](lab_harness/agent/agent.py). No other framework dependencies.
- **An agent in a few lines.** `route` a provider, pick tools, build an
  `Agent`, `await agent.prompt(...)`. The run records itself as it goes, so the
  transcript, the [HTML report](https://lxucs.github.io/lab-harness/search_agent/runs/793/report.html) and the token
  and cost numbers come with it.
- **Every request reproducible.** A session records the system prompt, tools,
  model, reasoning effort and config each request was sent with.
  [`trajectories.py --wire`](examples/trajectories.py) exports them as the JSON
  bodies the endpoint got. See [trajectories](docs/sessions.md#trajectories).
- **One loop, many models.** Claude, OpenAI, DeepSeek, GLM, or any other
  OpenAI-compatible endpoint. Every endpoint and model is a table in
  [`llm/catalog.toml`](lab_harness/llm/catalog.toml), so a new vendor or model
  is a few lines of TOML, not code.

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
export OPENAI_API_KEY=sk-...                          # or api_key in .lab-harness/config.local.toml
export TAVILY_API_KEY=tvly-...                        # optional, for web_search
uv run lab-harness                                    # interactive terminal
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
builds the one the CLI sends, as [quickstart.py](examples/quickstart.py) shows;
see the `[prompts]` sections in [prompts.toml](lab_harness/prompts/prompts.toml).

[examples/](examples/) goes further: streaming events, a custom tool, all four
hooks, a scripted model that needs no key, and exporting a session's requests
for training. Configuration and every environment variable:
[docs/configuration.md](docs/configuration.md).

## Example: a search agent built on lab-harness

[`search_agent/`](search_agent/) is a search agent — an agent that answers a
question by searching and reading documents — running as a program you can start
with one command. The agent itself is a few lines:

```python
model, provider = route(None, config)  # the model, from config alone
tools = tools_for(("ls", "glob", "grep", "read", "search_docs"))
agent = Agent(
    model=model,
    provider=provider,
    tools=tools,
    config=config,
    system_prompt=SYSTEM_PROMPT,
    cwd=".",
)
messages = await agent.prompt(question)
```

`uv run python -m search_agent` runs it, and a real run is committed to look
at: [the report it produced](https://lxucs.github.io/lab-harness/search_agent/runs/793/report.html), with the
trajectory as JSON beside it.

## Documentation

| Page | Covers |
|---|---|
| [examples/](examples/) | Runnable usage, from the terminal and from Python |
| [search_agent/](search_agent/) | A search agent: a whole program in Python, and a committed run to read |
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

## Roadmap

- **Context compaction** — the one planned piece unbuilt. It fits the
  `prepare_request` hook, so it lands without touching the loop.
- **Anthropic prompt caching** — the Messages adapter sends no `cache_control`
  markers yet, so Anthropic's cache hit rate reads 0%.

Not planned: MCP, sub-agents, plan mode, an RPC control channel,
extensions-as-plugins, sandboxing, provider-native `web_search`. What each item
needs, and why those are left out: [docs/README.md](docs/README.md#todo).

## Citation

If lab-harness helps your research, GitHub's **Cite this repository** button
gives a citation in APA or BibTeX, from [CITATION.cff](CITATION.cff).

## License

[MIT](LICENSE).
