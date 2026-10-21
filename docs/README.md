# lab-harness Documentation

A minimal agent harness in Python, built for research — every request
reproducible from its transcript, one loop across Claude, OpenAI, DeepSeek and
GLM, every step a typed event. Modeled on
[pi](https://github.com/earendil-works/pi). The [README](../README.md) has the
pitch and a quick start, [examples/](../examples/) has runnable usage from the
terminal and from Python, [`search_agent/`](../search_agent/) is a complete
program built on the harness with a committed run to read, and these pages
have the details.

## Reading order

Start with [architecture.md](architecture.md) — it explains the layering that
everything else assumes, and its [reading order](architecture.md#reading-order)
walks the code a step at a time.

| Page | Covers |
|---|---|
| [architecture.md](architecture.md) | Layering, module boundaries, why the seams are where they are |
| [events.md](events.md) | The agent event protocol — the contract between runtime and frontend |
| [providers.md](providers.md) | The LLM layer, provider contract, adding a provider |
| [tools.md](tools.md) | Tool contract, built-in tools, skills, adding a tool |
| [sessions.md](sessions.md) | Transcript format, resume, rebuilding trajectories |
| [cli.md](cli.md) | Terminal, flags, keybindings, the usage and cost readout |
| [configuration.md](configuration.md) | Config files, environment variables, models and prices, prompts |
| [contributing.md](contributing.md) | Code conventions, architectural invariants, config rules |

Working agreements for AI coding agents are in [`../AGENTS.md`](../AGENTS.md).

## Status

Every numbered build step is complete, so these pages describe what the code
does rather than what it will do. At a glance:

- **Runtime** — a tested agent loop; the local tool set
  (`read`/`write`/`edit`/`ls`/`bash`/`grep`/`glob`/`skill`); `web_search` and
  `web_fetch` behind pluggable backends; three provider adapters — Messages,
  Responses, Chat Completions. Reasoning is carried across tool calls the way
  each vendor requires. [tools.md](tools.md) ·
  [providers.md](providers.md)
- **Terminal** — streamed, collapsible thinking; colored edit diffs; slash
  commands listed as they are typed; a per-run token, cache and cost readout;
  a fail-closed approval gate. The tool set, the skill set and the reasoning
  effort are each chosen in config, per run, or mid-session, and the system
  prompt is rebuilt around them as they change. [cli.md](cli.md)
- **Config as data** — layered TOML, with the prompts and the whole model
  catalog as config files. The system prompt is configurable sections, filled
  from the environment, with the project's `HARNESS.md`/`AGENTS.md` appended.
  [configuration.md](configuration.md)
- **Sessions** — durable transcripts that resume on screen where they left
  off, export as a searchable web page, and record enough to rebuild every
  request they made, changes mid-session included.
  [sessions.md](sessions.md)

### TODO

- **Context compaction.** The one planned piece still unbuilt, marked TODO in
  [architecture.md](architecture.md). `prepare_request` is called before every
  provider request, `Model.context_window` and per-message `Usage` exist, and
  the session schema reserves a `compaction` entry — so it lands without
  touching the loop. Its rewrites are recorded already; the entry is for
  writing a summary once rather than the whole view with every request.
- **Anthropic prompt caching.** The Messages adapter sends no `cache_control`
  markers yet, so Anthropic caches nothing and its cache hit rate reads 0%.
  Contained to one function in `anthropic.py`; the placement is planned in
  [providers.md](providers.md#the-anthropic-adapter).

### Not planned

MCP, sub-agents, plan mode, RPC control channel, extensions-as-plugins,
sandboxing, and provider-native `web_search` — that last one would add a second
tool-execution path only one of the three adapters could ever use.
