# Configuration

> **Status:** Implemented. Layered TOML loading, the resolution order below,
> and every environment variable marked "works" are live.

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

A missing file is normal and skipped. A file that *exists* but is malformed is
a startup error: config you meant to apply and that silently did nothing is
worse than a failure you can see.

**Layers merge, they do not replace.** A later file overrides the keys it names
and leaves the rest alone, so a one-line `config.local.toml` does not reset
everything else to defaults.

**Unknown keys are an error.** A typo that silently leaves the default in place
is found days later, by the setting apparently having no effect. The message
names the file, the full key path, and the keys that *are* available at that
level. Wrong types are rejected the same way — including `max_turns = true`,
which an unguarded check would accept, since `bool` subclasses `int` in Python.

There is deliberately no generic environment-variable overlay. Each variable is
read where it is used, and its *name* is itself a config key, so a deployment
using different names changes config rather than code.

Files are located relative to `--cwd`, so `harness --cwd ../other-project`
picks up that project's `.harness/`, not this one's.

## Files

| File | Committed | Contents |
|---|---|---|
| `config.toml` | yes | Non-sensitive defaults — tool lists, feature flags, limits |
| `config.local.toml` | **no** | Keys, absolute paths, private endpoints |
| `config.toml.example` | yes | Template documenting every key, placeholders only |

### `config.toml` — safe to commit

Keys mirror the dataclass fields in `harness/config.py`. Every value shown is
the current default, so a key you do not set behaves exactly as written.

```toml
[agent]
max_turns = 50          # ceiling on provider requests in one run
parallel_tools = true   # run independent tool calls concurrently

[tools.read]
default_limit = 2000    # lines returned when the model requests no limit
max_line_length = 2000  # where a long line is clipped
max_bytes = 256000      # largest file `read` will open

[tools.list]
max_entries = 1000      # entries per `ls`

[tools.bash]
timeout_seconds = 120       # default wall-clock limit
max_timeout_seconds = 600   # ceiling on a requested timeout
max_output_bytes = 30000    # output kept before eliding the middle

[tools.search]
max_results = 200       # matches per grep
max_files = 500         # paths per glob
ignored_directories = [".git", "node_modules", "__pycache__"]  # …and more

[tools.web]
backend = "tavily"        # "tavily" (needs a key) or "ddgs" (keyless)
max_results = 5           # results per search
timeout_seconds = 30.0    # per search or fetch
max_bytes = 2000000       # largest response web_fetch downloads
max_content_chars = 50000 # extracted text handed to the model

[tools.web.tavily]
api_key_env = "TAVILY_API_KEY"  # NAME of the variable; never the value here
search_depth = "basic"          # "advanced" costs two credits instead of one
include_raw_content = false     # full page text beside each snippet

[tools.approval]
required = ["bash", "write", "edit"]  # prompt before these run
allow_remember = true                 # whether "always" is honored

[terminal]
enabled_tools = ["read", "write", "edit", "ls", "bash", "grep", "glob"]
max_result_lines = 12   # tool output shown before collapsing
markdown = true         # render assistant text as markdown

[session]
home = "~/.harness"     # root for state; $HARNESS_HOME overrides
enabled = true          # write transcripts at all
version = 1             # transcript schema version

[providers.anthropic]
default_max_tokens = 8192     # when the model declares none
thinking_budget_tokens = 4096 # when reasoning is requested
```

The full list, with every default, is in
[`.harness/config.toml.example`](../.harness/config.toml.example).

### `config.local.toml` — never committed

```toml
[tools.web.tavily]
api_key = "tvly-..."   # placeholder

[providers.anthropic]
base_url = "https://internal-gateway.example.com"   # placeholder
```

Prefer environment variables for secrets. `config.local.toml` is for
machine-specific non-secret values, and for keys when env vars are inconvenient
— a key that should survive across shells without living in a dotfile, say.

Environment wins over this file, so a key here can always be overridden.

## Environment variables

### Providers

| Variable | Purpose | Status |
|---|---|---|
| `ANTHROPIC_API_KEY` | Anthropic key | works |
| `ANTHROPIC_AUTH_TOKEN` | Bearer token; takes precedence over the key | works |
| `ANTHROPIC_BASE_URL` | Override the endpoint | works |
| `OPENAI_API_KEY` | OpenAI-compatible key | step 12 |
| `OPENAI_BASE_URL` | Endpoint — point at DeepSeek, Groq, vLLM, Ollama, … | step 12 |

The *names* of these variables are themselves config keys
(`providers.anthropic.api_key_env` and friends), so a deployment that uses
different names changes config rather than code. The values are never written to
a config file — see the hygiene rules below.

### Web search

| Variable | Purpose | Status |
|---|---|---|
| `TAVILY_API_KEY` | Key for the default search backend | works |

The backend is chosen by `tools.web.backend` in config, not by an environment
variable. Only Tavily needs a key; the `ddgs` backend needs none, and
`web_fetch` never does. See [tools.md](tools.md#web-tools).

### Harness

| Variable | Purpose | Status |
|---|---|---|
| `HARNESS_MODEL` | Override the configured model | works |
| `HARNESS_HOME` | Override `~/.harness` | works |
| `HARNESS_LOG_LEVEL` | `debug` \| `info` \| `warning` \| `error` | planned |

## Reading config in code

Config is resolved once at startup and passed down. Modules receive what they
need rather than importing a global. Tools read theirs from `ToolContext`:

```python
# Right — the limit is a named, discoverable setting
limits = ctx.config.tools.read
window = lines[start : start + (p.limit or limits.default_limit)]

# Wrong — a magic number nobody can find or change
window = lines[start : start + 2000]
```

Threading config through `ToolContext` rather than importing it is also what
makes the limits testable: two contexts with different values must produce
different behavior, and `tests/test_tools_fs.py` asserts exactly that.

A literal is acceptable only for genuine protocol constants that would be wrong
to change — an API version header, a schema version number.

## Project context

A `HARNESS.md` or `AGENTS.md` in the working directory is loaded into the system
prompt as project instructions, following the convention pi and other harnesses
use.
