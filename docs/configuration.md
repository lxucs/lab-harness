# Configuration

> **Status:** Not yet implemented. This page describes the designed surface and
> is updated as `config.py` lands.

## Principles

**Nothing tunable is hardcoded.** Models, timeouts, retry counts, truncation
limits, directory locations, default tool sets — all of it resolves from config.
The schema in `harness/config.py` is the single list of every knob the system
has, and every default lives there.

**Nothing sensitive is committed.** Keys, absolute paths, and private endpoints
go in `config.local.toml` (gitignored) or environment variables — never in
`config.toml`, docs, or tests. See the hygiene rules in
[`../AGENTS.md`](../AGENTS.md).

## Resolution order

Lowest to highest precedence:

1. Schema defaults in `harness/config.py`
2. `~/.harness/config.toml` — user config
3. `./.harness/config.toml` — project config, committed
4. `./.harness/config.local.toml` — project-local, **gitignored**
5. Environment variables
6. CLI flags

## Files

| File | Committed | Contents |
|---|---|---|
| `config.toml` | yes | Non-sensitive defaults — tool lists, feature flags, limits |
| `config.local.toml` | **no** | Keys, absolute paths, private endpoints |
| `config.toml.example` | yes | Template documenting every key, placeholders only |

### `config.toml` — safe to commit

```toml
model = "claude-sonnet-5"

[tools]
enabled = ["read", "write", "edit", "ls", "bash", "grep", "glob", "web_search"]
approval = ["bash", "write", "edit"]

[tools.bash]
timeout_seconds = 120
max_output_bytes = 30_000

[limits]
max_tool_output_lines = 2_000
```

### `config.local.toml` — never committed

```toml
[providers.anthropic]
base_url = "https://internal-gateway.example.com"   # placeholder

[providers.openai]
api_key = "sk-..."   # prefer the environment variable instead
```

Prefer environment variables for secrets. `config.local.toml` is for
machine-specific non-secret values, and for keys when env vars are inconvenient.

## Environment variables

### Providers

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic key |
| `ANTHROPIC_AUTH_TOKEN` | Bearer token; takes precedence over the key |
| `ANTHROPIC_BASE_URL` | Override the endpoint |
| `OPENAI_API_KEY` | OpenAI-compatible key |
| `OPENAI_BASE_URL` | Endpoint — point at DeepSeek, Groq, vLLM, Ollama, … |

### Web search

| Variable | Purpose |
|---|---|
| `HARNESS_SEARCH_PROVIDER` | `tavily` \| `brave` \| `exa` |
| `TAVILY_API_KEY` / `BRAVE_API_KEY` / `EXA_API_KEY` | Backend key |

Only needed when the active model lacks provider-native search — see
[tools.md](tools.md).

### Harness

| Variable | Purpose |
|---|---|
| `HARNESS_HOME` | Override `~/.harness` |
| `HARNESS_MODEL` | Override the configured model |
| `HARNESS_LOG_LEVEL` | `debug` \| `info` \| `warning` \| `error` |

## Reading config in code

Config is resolved once at startup and passed down. Modules receive what they
need rather than importing a global:

```python
# Right — the limit is a named, discoverable setting
if len(output) > config.tools.bash.max_output_bytes:
    ...

# Wrong — a magic number nobody can find or change
if len(output) > 30_000:
    ...
```

A literal is acceptable only for genuine protocol constants that would be wrong
to change — an API version header, a schema version number.

## Project context

A `HARNESS.md` or `AGENTS.md` in the working directory is loaded into the system
prompt as project instructions, following the convention pi and other harnesses
use.
