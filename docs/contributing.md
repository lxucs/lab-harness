# Contributing

Code conventions and the rules that keep the architecture intact. Working
agreements for AI coding agents — environment, testing, git — are in
[`../AGENTS.md`](../AGENTS.md).

## Code conventions

- Python 3.12+, `async`/`await` throughout the runtime.
- Frozen dataclasses for protocol types (messages, events); Pydantic models for
  tool parameters, where one model gives schema + validation + type hints.
- Full type annotations. `pyright` must be clean.
- `ruff check` and `ruff format` before calling work done.
- Match the comment density of surrounding code. Comment *why*, not *what*.
- Prefer small, focused modules. A file past ~400 lines is usually doing two
  things.

## The invariants

Four rules the architecture depends on. Each is enforced at exactly one
boundary, which is why the rest of the code has so few defensive checks.

### 1. Dependencies point downward

```
cli/  →  agent/  →  llm/
         tools/      (llm has no upward deps)
         session/
```

`cli/` is a subscriber; the runtime never imports from it. Deleting `cli/` must
leave the runtime working. If you need an upward import, the abstraction is in
the wrong place — add a hook or inject the dependency.

See [architecture.md](architecture.md).

### 2. Events stay serializable

Every `AgentEvent` is a frozen dataclass of JSON primitives. No live objects,
file handles, SDK types, or callables. `tests/test_serialization.py`
round-trips every event type and fails if this breaks.

This is what lets a second consumer exist — the JSONL sink today, a frontend in
another language later. See [events.md](events.md).

### 3. Providers never raise

Request, model, and runtime failures terminate the stream with an `error` event
carrying an `AssistantMessage` with `stop_reason="error"`. This is why the loop
has almost no try/except. See [providers.md](providers.md).

### 4. Tools never kill the loop

A tool returns `ToolResult(is_error=True)` or raises; the loop converts a raise
into an error result. See [tools.md](tools.md).

## Configuration

### No hardcoded values

Anything tunable comes from config, not a literal mid-function: models, base
URLs, timeouts, retry counts, truncation limits, token thresholds, directory
locations, default tool sets.

```python
# Wrong — buried, undiscoverable, unchangeable without an edit
if len(output) > 30_000:
    output = output[:30_000]

# Right
if len(output) > config.tools.bash.max_output_bytes:
    output = output[: config.tools.bash.max_output_bytes]
```

Defaults live in one place — the schema in `harness/config.py` — so reading the
schema tells you every knob the system has. Adding a tunable means adding it to
the schema with a default in the same change.

A literal is fine only for genuine protocol constants that would be wrong to
change: an API version header, a JSONL schema version, the `"\n"` a line
splitter splits on.

### Nothing sensitive is committed

No API keys, tokens, absolute paths (`/Users/...`, `/home/...`), private URLs,
hostnames, org IDs, or personal identifiers — in config, docs, tests, or
examples.

| File | Committed | Contents |
|---|---|---|
| `config.toml` | yes | Non-sensitive defaults |
| `config.local.toml` | **no** — gitignored | Keys, absolute paths, private endpoints |
| `config.toml.example` | yes | Template with placeholders |

Prefer environment variables for secrets. If you add a config location, add its
local variant to `.gitignore` in the same change.

In docs and examples use obvious placeholders — `sk-...`,
`https://api.example.com`, `/path/to/project` — never a value copied from a
running environment.

See [configuration.md](configuration.md).

## Documentation

Docs are written alongside the code, not after. A module whose doc page was not
updated is not finished.

| Change | Update |
|---|---|
| New or changed tool | [tools.md](tools.md) |
| New or changed provider | [providers.md](providers.md) |
| Event protocol change | [events.md](events.md) |
| Layering or module change | [architecture.md](architecture.md) |
| Transcript format change | [sessions.md](sessions.md) |
| New config key | [configuration.md](configuration.md) |

Also tick the build-order checklist in [`../README.md`](../README.md) as steps
complete.

## Checks

```bash
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest
```

Run them and read the output before claiming work is done.
