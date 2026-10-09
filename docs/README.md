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
| [sessions.md](sessions.md) | Transcript format, resume |
| [cli.md](cli.md) | Terminal usage, flags, keybindings |
| [configuration.md](configuration.md) | Config file, environment variables |
| [contributing.md](contributing.md) | Code conventions, architectural invariants, config rules |

Design rationale and the decisions behind v1 scope live in
[superpowers/specs/2026-10-09-harness-runtime-design.md](superpowers/specs/2026-10-09-harness-runtime-design.md).

Working agreements for AI coding agents are in [`../AGENTS.md`](../AGENTS.md).

## Status

Pre-implementation. The design is settled; these pages are filled in alongside
the code they describe, per the build order in the design spec. Sections marked
*Not yet implemented* are intentional placeholders, not oversights.
