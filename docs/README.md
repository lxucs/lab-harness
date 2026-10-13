# Harness Documentation

A lightweight agent harness. Python runtime, terminal frontend, modeled on
[pi](https://github.com/earendil-works/pi).

## Reading order

Start with [architecture.md](architecture.md) — it explains the layering that
everything else assumes.

| Page | Covers |
|---|---|
| [architecture.md](architecture.md) | Layering, module boundaries, why the seams are where they are |
| [events.md](events.md) | The agent event protocol — the contract between runtime and frontend |
| [providers.md](providers.md) | The LLM layer, provider contract, adding a provider |
| [tools.md](tools.md) | Tool contract, built-in tools, adding a tool |
| [sessions.md](sessions.md) | Transcript format, resume, rebuilding trajectories |
| [cli.md](cli.md) | Terminal, flags, keybindings, the usage and cost readout |
| [configuration.md](configuration.md) | Config files, environment variables, pricing |
| [contributing.md](contributing.md) | Code conventions, architectural invariants, config rules |

Design rationale and the decisions behind v1 scope live in
[superpowers/specs/2026-10-09-harness-runtime-design.md](superpowers/specs/2026-10-09-harness-runtime-design.md).

Working agreements for AI coding agents are in [`../AGENTS.md`](../AGENTS.md).

## Status

Every numbered build step is complete, so these pages describe what the code
does rather than what it will do. The one piece of the original design still
unbuilt is context compaction, marked TODO in
[architecture.md](architecture.md). The remaining TODOs — compaction and Anthropic
prompt caching — and what is deliberately not planned are listed in
[the README](../README.md#todo).
