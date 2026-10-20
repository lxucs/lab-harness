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
- Prefer small, focused modules. Past ~400 lines *of code* a file is usually
  doing two things. Comments and docstrings do not count against that — several
  modules here run past 500 total lines and are still under the limit.

## The invariants

Four rules the architecture depends on. Each is enforced at exactly one
boundary, which is why the rest of the code has so few defensive checks.

### 1. Dependencies point downward

```
cli/  →  agent/  →  llm/
         tools/      (llm has no upward deps)
         session/
cli/  →  prompts/    (agent/ is handed the finished system prompt)
```

`cli/` is a subscriber; the runtime never imports from it. Deleting `cli/` must
leave the runtime working. If you need an upward import, the abstraction is in
the wrong place — add a hook or inject the dependency.

See [architecture.md](architecture.md).

### 2. Events stay serializable

Every `AgentEvent` is a frozen dataclass of JSON primitives. No live objects,
file handles, SDK types, or callables. `tests/test_serialization.py`
round-trips every event type and fails if this breaks. That includes a tool's
`ToolResult.details`, which ride on `tool_execution_end` — a `Path` or a
datetime put there would break every serializing subscriber.

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

Defaults live in one place — the schema in `lab_harness/config.py` — so reading the
schema tells you every knob the system has. Adding a tunable means adding it to
the schema with a default in the same change.

LLM settings are the exception: they are data, all in
`lab_harness/llm/catalog.toml`, in the same format a config file uses —
Claude's adapter settings, every OpenAI-family endpoint, every model, and the
`[defaults]` an endpoint or a model gets for a key it leaves out. Their classes
in `config.py` have types and no defaults, so a new LLM setting is a field
there and its value in the catalog, in the same change. Adding a model is one
`[models.<id>]` table, with its documented limits and price; no Python changes.

Prompts are data too. Text the model is sent — each section of the system
prompt, the wrapper around project instructions — lives in
`lab_harness/prompts/prompts.toml` and reaches code as `config.prompts`, never
as a string literal in Python; `prompts/system.py` beside it fills it in and
joins it. The system prompt names no tools: tool guidance belongs in the tool's own
description, which is only sent when the tool is. The one exception is text
that belongs to a tool or an error — a tool's description, a tool result's
message — which stays beside the code it describes.

A literal is fine only for genuine protocol constants that would be wrong to
change: an API version header, a JSONL schema version, the `"\n"` a line
splitter splits on.

### Nothing sensitive is committed

No API keys, tokens, absolute paths (`/Users/...`, `/home/...`), private URLs,
hostnames, org IDs, or personal identifiers — in config, the model catalog,
the prompts, docs, tests, or examples.

| File | Committed | Contents |
|---|---|---|
| `config.toml` | yes | Non-sensitive defaults |
| `config.local.toml` | **no** — gitignored | Keys, absolute paths, private endpoints |
| `config.example.toml` | yes | Template with placeholders |
| `lab_harness/llm/catalog.toml` | yes | Every LLM setting — public, documented vendor data only |
| `lab_harness/prompts/prompts.toml` | yes | Shipped prompts — no tool names, nothing about a particular machine |

Prefer environment variables for secrets. If you add a config location, add its
local variant to `.gitignore` in the same change.

In docs and examples use obvious placeholders — `sk-...`,
`https://api.example.com`, `/path/to/project` — never a value copied from a
running environment.

See [configuration.md](configuration.md).

### Nothing sensitive is logged

The same list, one step further. Debug output is pasted into issues, so a
credential that reaches a log record is a credential published.

```python
# Right — the name says which source won, and leaks nothing
_log.debug("Credentials taken from %s", cfg.api_key_env)

# Wrong
_log.debug("Using key %s", api_key)
```

This covers private endpoints too — `ANTHROPIC_BASE_URL` is often an internal
gateway. Log the variable's *name*, never its value.
`tests/test_logging.py` asserts a key set in the environment does not appear in
debug output; extend it when you add a provider.

### Nothing sensitive is recorded

Session transcripts hold a snapshot of the whole config, so a credential set in
`config.local.toml` would land in every one. `config.snapshot` replaces the
fields named in `SECRET_FIELDS`. A new config field that holds a credential
must be named like one (`api_key`, `*_token`, `*_secret`) and added there —
`tests/test_config_loading.py` fails on a credential-shaped name the set is
missing.

### Everything a request carries is recorded

A transcript rebuilds every request exactly, so anything that can change what
a request sends — the system prompt, the tools, the model (its reasoning
effort included), the messages — has to reach the event stream when it
changes: `agent_start` carries the first three for every run, and
`request_rewritten` carries a hook's rewrite of one request. A new way to
change one of them mid-session, such as the terminal's `/effort`, needs no
recorder change if it changes the agent's state between runs; anything subtler needs an event, or `recorded_requests` will rebuild the
request wrong. `tests/test_session.py` compares rebuilt requests against what
the provider was actually handed — extend it with the new case.

## Logging

Take a logger the ordinary way and emit unconditionally:

```python
_log = logging.getLogger(__name__)
```

Name it `_log`, and never call `basicConfig` or add a handler — only `cli/`
configures logging, through `lab_harness/logs.py`. A library that configures
logging hijacks the application embedding it.

Pick the level by who needs the line:

| Level | For |
|---|---|
| `warning` | A problem with no other channel. It prints by default, so it must be worth interrupting a working run |
| `info` | One line per significant decision — the model in use, which rule picked the provider, the transcript path — and one per run |
| `debug` | The trace: config files read, request shapes, response ids, tool outcomes, durations, outbound URLs |

Never log message content or tool output. The transcript holds the
conversation; a log that duplicates it is both a privacy problem and unreadable.
Log *shape* instead — counts, ids, durations. An id is what lets a line stand
in for the content: a tool call's id finds its arguments in the transcript.
A request goes through `llm/providers/_trace.py`, which logs the keys in
`SETTINGS` by value and every other key by size, so a key added later stays
private until someone decides it is a setting.

**Keep each call on one line** where it fits in the 100 columns — a short
message, `key=value` for several facts, and a helper that builds the facts when
there are many (`request_shape`, or `_shape` and `_outcome` in the loop):

```python
# Right
_log.debug("Tool call %s: %s", call.id, _outcome(result, duration_ms))

# Wrong — five lines of logging for one fact
_log.debug(
    "Tool %r finished in %dms%s",
    call.name,
    duration_ms,
    " (error)" if result.is_error else "",
)
```

Several lines are fine when the message cannot get shorter without losing what
it says.

Levels are documented in [configuration.md](configuration.md#logging), so a new
record at a level that page does not describe means one of the two is wrong.

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
| New config key, env var, or log level | [configuration.md](configuration.md) |
| New prompt, or a change to the default one | `lab_harness/prompts/prompts.toml` documents itself; a new section is also a field of `PromptsConfig`, a place in `prompts/system.py`'s order, and a row in [configuration.md](configuration.md#prompts) |

Keep [`../README.md`](../README.md)'s Status section honest when something
lands or is abandoned.

## Checks

```bash
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest
```

Run them and read the output before claiming work is done.
