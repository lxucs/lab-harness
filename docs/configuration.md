# Configuration

## Principles

**Nothing tunable is hardcoded.** Models, timeouts, retry counts, truncation
limits, directory locations, default tool sets, prompts — all of it resolves
from config. The schema in `lab_harness/config.py` is the single list of every
knob the system has, and every default lives there — except two kinds of data
the package ships in config-file format:
[`lab_harness/llm/catalog.toml`](../lab_harness/llm/catalog.toml), every LLM
setting, and [`lab_harness/prompts.toml`](../lab_harness/prompts.toml), the
prompts.

**Nothing sensitive is committed.** Keys, absolute paths, and private endpoints
go in a `config.local.toml` (never committed) or environment variables — never
in `config.toml`, docs, or tests. See the hygiene rules in
[`../AGENTS.md`](../AGENTS.md).

## Resolution order

Lowest to highest precedence:

1. Schema defaults in `lab_harness/config.py`, every LLM setting in
   `lab_harness/llm/catalog.toml`, and the prompts in `lab_harness/prompts.toml`
2. `~/.lab-harness/config.toml` — user config
3. `~/.lab-harness/config.local.toml` — user-local, for keys
4. `./.lab-harness/config.toml` — project config, committed
5. `./.lab-harness/config.local.toml` — project-local, **gitignored**
6. Environment variables
7. CLI flags

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

Project files are located relative to `--cwd`, so `lab-harness --cwd
../other-project` picks up that project's `.lab-harness/`, not this one's. The
two in `~/.lab-harness/` (or `$LAB_HARNESS_HOME`) apply in every directory:
settings and keys you want wherever the harness runs go there.

## Files

| File | Committed | Contents |
|---|---|---|
| `config.toml` | yes | Non-sensitive defaults — tool lists, feature flags, limits |
| `config.local.toml` | **no** | Keys, absolute paths, private endpoints |
| `config.example.toml` | yes | Template documenting every key but the LLM ones, placeholders only |

`config.toml` and `config.local.toml` are each read from home, then from the
project. Only a project's `config.toml` is committed.

### `config.toml` — safe to commit

Keys mirror the dataclass fields in `lab_harness/config.py`. Every value shown is
the current default, so a key you do not set behaves exactly as written.

```toml
[agent]
max_turns = 50                   # ceiling on provider requests in one run
parallel_tools = true            # run independent tool calls concurrently
model = "claude-sonnet-5"        # used when --model and $LAB_HARNESS_MODEL are absent
model_env = "LAB_HARNESS_MODEL"  # NAME of the variable that overrides `model`
provider = "anthropic"           # for a model id nothing else routes

[project]
context_files = ["HARNESS.md", "AGENTS.md"]  # first one that exists is loaded
max_bytes = 32000                            # longer files are truncated

[prompts]               # ships in lab_harness/prompts.toml; see "Prompts" below
system = "You are a helpful assistant running in a terminal. …"
project_instructions = "## Project instructions ({name})\n\n…{text}"

[tools]
enabled = [             # what the model may call, in both modes; --tools overrides
    "read", "write", "edit", "ls", "bash", "grep", "glob",
    "web_search", "web_fetch",
]

[tools.read]
default_limit = 2000    # lines returned when the model requests no limit
max_line_length = 2000  # where a long line is clipped
max_bytes = 256000      # largest file `read` will open

[tools.list]
max_entries = 1000      # entries per `ls`

[tools.diff]
context_lines = 3       # unchanged lines around each change in an edit's diff

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
max_result_lines = 12   # tool output shown before collapsing
max_diff_lines = 40     # lines of an edit's diff shown before collapsing
markdown = true         # render assistant text as markdown
thinking = "collapse"   # reasoning once its message ends: "full" | "collapse" | "hidden"
thinking_preview_lines = 4  # lines of reasoning shown while it streams, when collapsing
show_usage = true       # token, cache and cost line after each run
currency_symbol = "$"   # label for costs computed from model prices

[terminal.theme]        # rich style strings, by role; see "The terminal" below
accent = "#d77757"
prompt = "bold #d77757"

[session]
home = "~/.lab-harness" # root for state; $LAB_HARNESS_HOME overrides
enabled = true          # write transcripts at all
version = 3             # transcript schema version

# Every LLM setting — [providers.anthropic], [providers.openai.<name>] and
# [models.<id>] — ships in lab_harness/llm/catalog.toml, in this format; see
# "Provider profiles" and "Models" below.

[logging]
level = "warning"                    # debug | info | warning | error
level_env = "LAB_HARNESS_LOG_LEVEL"  # NAME of the variable that overrides `level`
format = "%(levelname)s %(name)s: %(message)s"
```

Every key, with its default, is in
[`.lab-harness/config.example.toml`](../.lab-harness/config.example.toml) —
except the LLM ones, which are all in
[`lab_harness/llm/catalog.toml`](../lab_harness/llm/catalog.toml).

### `config.local.toml` — never committed

```toml
[tools.web.tavily]
api_key = "tvly-..."   # placeholder

[providers.openai.deepseek]
api_key = "sk-..."     # literal key; the api_key_env variable still wins if set
base_url = "https://internal-gateway.example.com/v1"   # placeholder
```

Anthropic's endpoint has no `base_url` key: it is read from
`ANTHROPIC_BASE_URL` (named by `providers.anthropic.base_url_env`), so a
gateway in front of it is an environment setting.

Prefer environment variables for secrets. `config.local.toml` is for
machine-specific non-secret values, and for keys when env vars are inconvenient
— a key that should survive across shells without living in a dotfile, say.
A key you want in every project goes in `~/.lab-harness/config.local.toml`; one
for this project alone, in `.lab-harness/config.local.toml`.

Environment wins over both files, so a key in either can always be overridden.

## Environment variables

### Providers

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic key |
| `ANTHROPIC_AUTH_TOKEN` | Bearer token; takes precedence over the key |
| `ANTHROPIC_BASE_URL` | Override the endpoint |
| `OPENAI_API_KEY` | Key for the `openai` profile |
| `OPENAI_BASE_URL` | Endpoint for the `openai` profile |
| `DEEPSEEK_API_KEY` | Key for the `deepseek` profile |
| `ZHIPUAI_API_KEY` | Key for the `glm` profile |

Every OpenAI-compatible profile names its own variables, so a profile you add
brings its own (`providers.openai.<name>.api_key_env`).

The *names* of these variables are themselves config keys
(`providers.anthropic.api_key_env` and friends), so a deployment that uses
different names changes config rather than code. The values are never written to
a config file — see the hygiene rules below.

### Web search

| Variable | Purpose |
|---|---|
| `TAVILY_API_KEY` | Key for the default search backend |

The backend is chosen by `tools.web.backend` in config, not by an environment
variable. Only Tavily needs a key; the `ddgs` backend needs none, and
`web_fetch` never does. See [tools.md](tools.md#web-tools).

### lab-harness

| Variable | Purpose |
|---|---|
| `LAB_HARNESS_MODEL` | Override `agent.model`; `--model` still wins |
| `LAB_HARNESS_HOME` | Override `session.home` |
| `LAB_HARNESS_LOG_LEVEL` | Override `logging.level`; `--log-level` still wins |

Model resolution is the general precedence rule in miniature: `--model`, then
`$LAB_HARNESS_MODEL`, then `agent.model` from config, which defaults to
`claude-sonnet-5`. An id the catalog has never heard of is **not** an error —
it is given conservative limits rather than refused, because being unable to
use a model released last week is worse than a slightly wrong `context_window`.
Which provider serves it is decided separately; see [Models](#models).

## Provider profiles

`providers.openai` is a table of **named** endpoints. It is one of the two
places in the schema where an unknown key is not an error — a new key is a new
vendor (the other is [`models`](#models), where it is a new model). That is
how OpenAI, DeepSeek, GLM, Groq, OpenRouter, vLLM and Ollama are all supported
by one adapter. The shipped `openai`, `deepseek` and `glm` profiles are in
[`lab_harness/llm/catalog.toml`](../lab_harness/llm/catalog.toml).

```toml
[providers.openai.groq]
base_url = "https://api.groq.com/openai/v1"
api_key_env = "GROQ_API_KEY"

[models."llama-3.3-70b-versatile"]   # what routes to it — see "Models"
provider = "groq"

[providers.openai.local]
base_url = "http://localhost:11434/v1"   # Ollama
api_key_env = "OLLAMA_API_KEY"
```

**`api` picks the protocol.** `chat_completions` is the default because that is
what a new vendor almost certainly speaks; `responses` is OpenAI's current API
and the only way to carry an OpenAI reasoning model's chain of thought across a
tool call. The shipped `openai` profile sets `responses`. An unusable value
fails at startup, naming the profile at fault.

**`replay_reasoning_field` sends reasoning back** on Chat Completions, in the
named field of each earlier assistant message. DeepSeek returns a 400 for a
tool loop without it and GLM asks for it, so both shipped profiles set
`"reasoning_content"`. A profile you add starts at `""` — none sent — because a
server that never asked for the field may reject it. Set it only when the
vendor's docs ask; [providers.md](providers.md#how-thinking-is-handled-across-models)
has the per-vendor rules.

**`preserve_thinking` asks GLM to keep it.** GLM's standard endpoint discards
earlier reasoning unless the request says `thinking.clear_thinking: false`, so
replaying it there would buy nothing. The shipped `glm` profile turns this on;
it sends GLM's own request shape, so leave it off for any other vendor.

**Adding does not replace.** A profile named in a config file merges into the
shipped one; `[providers.openai.deepseek] timeout_seconds = 30` leaves its
`base_url` and `api_key_env` alone. A new profile starts from the catalog's
`[defaults.profile]`, which lists every key and what one left out gets —
OpenAI's endpoint and variables among them, so a profile for anyone else names
its own. Keys *inside* a profile are checked as strictly as anywhere else, so a
typo there still fails at startup.

## Models

`models` describes each model, keyed by id: which provider serves it, its
limits and capabilities, and what it costs. The ones lab-harness knows ship in
[`lab_harness/llm/catalog.toml`](../lab_harness/llm/catalog.toml), and a config
file adds a model or changes one with the same table:

```toml
[models."my-model-1.5"]    # quoted: a bare TOML key cannot contain a dot
provider = "groq"          # "anthropic" or a [providers.openai] profile
context_window = 128000
max_tokens = 8192
supports_reasoning = false
price = { input = 1.0, output = 4.0 }   # illustrative; take real rates from the vendor

[models.claude-sonnet-5]   # a catalogued model: only the keys given change
provider = "my-gateway"
```

| Key | Default | Meaning |
|---|---|---|
| `name` | the id | Display name |
| `provider` | `""` | Who serves it; empty leaves it to `agent.provider` |
| `context_window` | `200000` | Tokens the model reads; the terminal measures context against it |
| `max_tokens` | `8192` | Output ceiling sent with every request |
| `supports_reasoning` | `true` | Gates the thinking request an adapter sends |
| `supports_images`, `supports_server_tools` | `false` | Recorded; nothing acts on them |
| `price` | all zero | USD — or any one currency — per million tokens |

**The defaults are what an unlisted id gets.** They are the catalog's
`[defaults.model]`, which a config file cannot change — it overrides one entry
at a time. An id with no entry still works: it is assumed current, so it
reasons, and its limits are floors across current models, so guessing cannot
exceed what one accepts. An entry needs only the keys that differ.

**Which provider serves a model**, highest precedence first: `--provider`, then
the entry's `provider`, then `agent.provider`. So `--model deepseek-flash`
reaches DeepSeek with no flag, and pointing a catalogued model at a gateway or
a local server is one line. Full rationale:
[providers.md](providers.md#profiles).

```bash
lab-harness --model deepseek-flash           # routed by its models entry, no flag
lab-harness --model their-new-model --provider groq
```

### Prices

`price` exists only so the terminal can show a cost — nothing is enforced or
budgeted. Its four rates mirror `Usage`: `input`, `output`, `cache_read` and
`cache_write`.

**No price means no cost.** A model without one is unpriced, and the footer
and `:status` simply leave the cost out. The same goes for a session where
*any* request that used tokens ran on an unpriced model: a total that silently
skipped some requests would look like the whole bill.

**One formula prices every provider.** Every adapter reports `input` as the
uncached remainder, with cache reads and writes counted separately. A rate left
out is zero — for OpenAI, DeepSeek or GLM, which cache implicitly and report no
writes, `cache_write` is irrelevant anyway; leaving out `cache_read` would make
cached tokens free.

**Shipped prices**, all in USD:

- Claude, at Anthropic's first-party list rates (mid-2026; Fable 5 from
  2026-10), with `cache_write` at the 5-minute-TTL premium. Bedrock and Vertex
  bill differently.
- The GPT-6 models, at OpenAI's standard-tier rates (2026-10). A request over
  272K input tokens bills at OpenAI's long-context rates instead, which one
  rate cannot follow, so a cost shown for one runs low. `cache_write` is zero:
  the Responses adapter reports no cache writes, so those tokens count as
  `input`.
- `deepseek-flash` and `deepseek-v4-pro`, at DeepSeek's **peak** rates.
  DeepSeek halves its prices outside 01:00–04:00 and 06:00–10:00 UTC on
  weekdays, and one rate cannot follow the clock, so the higher one is shipped:
  a cost shown for DeepSeek may run high, never low. `input` is its cache-miss
  rate, and `cache_write` is zero — it charges nothing to fill the cache.
- GLM, at **Z.ai's international** rates. The shipped profile points at the
  *mainland* endpoint, which bills in CNY at its own rates — override the
  prices if that is the one you use. `cache_write` is zero: Z.ai lists cache
  storage as "limited-time free".

A config file merges over them — `[models.claude-sonnet-5.price] input = 2.5`
keeps the other three rates — so to switch one off, set all four to zero; an
all-zero price counts as none.

Amounts are labelled with `terminal.currency_symbol` (default `$`). It is only a
label: write every rate in the same currency, because a session that switches
models adds the costs together.

## Prompts

What the model is told besides the conversation and the tool declarations. The
text ships in [`lab_harness/prompts.toml`](../lab_harness/prompts.toml) — data,
not Python — and that file is the default of the `[prompts]` section, in the
same format. A config file replaces a prompt with the same key:

```toml
[prompts]
system = """\
You are a careful research assistant. Say when you are unsure. \
Keep responses concise."""
```

| Key | Holds |
|---|---|
| `system` | Opens every request's system prompt. Empty sends none. |
| `project_instructions` | Wraps the [project's instructions](#project-instructions), appended after a blank line. `{name}` is the file's name, `{text}` its contents. |

**The system prompt names no tools.** What the model can call, and how to use
each tool, comes from the tool declarations alone — so `tools.enabled` or
`--tools` can narrow the set without the prompt describing a tool that is not
there. Advice about one tool belongs in its description. A test holds the
shipped prompt to this; hold your own to it too.

**For one run**, `--system-prompt TEXT` or `--system-prompt-file PATH` replaces
`system`; the project's instructions are still appended. **Mid-session**, a
Python caller can assign `agent.state.system_prompt` between runs. Either way
the session records the prompt each request was sent with —
[sessions.md](sessions.md#changes-mid-session).

`project_instructions` is checked at startup, in every directory: a
placeholder other than `{name}` and `{text}`, or a template without `{text}`,
is an error. Write `{{` and `}}` for literal braces. A line ending in a
backslash inside `"""` continues the paragraph, so a prompt can be wrapped in
the file without breaking the text the model sees.

## The terminal

How the interactive terminal looks. What each element is:
[cli.md](cli.md#interactive).

`thinking` picks what a model's reasoning leaves in scrollback — `full`,
`collapse` (the default) or `hidden` — and Ctrl-O switches between `full` and
it for the session. [cli.md](cli.md#thinking) has the table.

`[terminal.theme]` styles the output by role rather than by color, so "what
does a failed tool look like" is one setting. Values are rich style strings —
a color name or hex code, plus any of `bold`, `dim`, `italic`, `underline`:

```toml
[terminal.theme]
accent = "#d77757"        # the spinner — Claude Code's orange
prompt = "bold #d77757"   # the › glyph
muted = "dim"             # tool arguments and output, notices, the usage line
thinking = "italic dim"   # reasoning, streamed or expanded
tool = "bold"             # the tool's name in a call line
success = "green"         # the bullet of a call that finished cleanly
error = "red"             # failed calls and errors
warning = "yellow"        # cancellations, warnings, calls awaiting approval
diff_added = "#ffffff on #225c2b"    # whole added lines of a diff
diff_removed = "#ffffff on #7a2936"  # whole removed lines
```

**Diff lines name a foreground as well as a background**, so they read the
same on a light terminal as a dark one. A background alone would leave the
terminal's own text color on it — near-black on a light theme.

**`prompt` is parsed by prompt_toolkit, not rich**, since the input line is
prompt_toolkit's. Hex colors and `bold`/`italic`/`underline` work in both;
rich's `rgb(…)` form does not.

**Every style is checked at startup.** A typo is a config error naming the
key, rather than a crash on the first message that uses it.

## Logging

Diagnostics about the harness itself — not about the conversation, which is
what the transcript and `--json` are for.

```bash
lab-harness --log-level debug -p "…"     # this run
LAB_HARNESS_LOG_LEVEL=debug lab-harness  # this shell
```

```toml
[logging]
level = "warning"
```

**Everything goes to stderr.** stdout carries the answer in plain-text mode and
the event stream under `--json`, so a log line on stdout would corrupt a
pipeline rather than inform one. `lab-harness -p "…" > answer.txt` still captures
the answer alone with logging turned all the way up.

The one exception is the interactive terminal when stderr *is* that terminal:
records are printed through the console, dimmed, above the live region. Written
to the terminal directly they would land inside the region and break its next
redraw. Redirect stderr — `lab-harness --log-level debug 2> lab-harness.log` — and they
go to the file as usual, out of the conversation.

| Level | What you get |
|---|---|
| `error` | Only failures the run could not absorb |
| `warning` *(default)* | …plus a model id not in the catalog, project instructions truncated, `max_turns` exhausted, a provider request that failed, and a resumed session's warnings in `-p` mode |
| `info` | …plus one line per significant decision — the model and where its id came from, which rule picked the provider, the tools and the system prompt and where each came from, the project instructions used, the transcript being written, approval being off, a `-p` resume — and one summary line per run |
| `debug` | …plus a trace: every config file read or skipped, the tools and prompt size, each request's shape and reasoning settings, each response's vendor id, stop reason as sent, sizes, tokens and timing, each tool call's argument names, approval answer, outcome and duration, and outbound web requests |

[cli.md](cli.md#diagnostics) has what a run looks like at `debug`.

**No conversation either.** A request is logged by shape: settings by value,
everything else — messages, the system prompt, tool declarations — by size. A
tool call is logged by its argument *names*, and its result by the details that
are numbers or flags; a command, a path or a diff stays out. What ties a line
to the rest is an id: a tool call's id finds its arguments in the transcript
and in `--json`, and a response id is what a vendor's support will ask for.

The default is `warning` because a run that works should print nothing beyond
its answer. The warnings are the ones with no other channel: interactive mode
shows a truncation notice and resume warnings on screen, but `-p` has no
screen, and a model id the catalog has never seen otherwise surfaces as an
opaque API error later.

**A level set in a config file cannot report that file being read.** Logging is
configured twice — once from the flag, the environment and the schema defaults
before `load_config` runs, and again from the loaded config. So to debug config
resolution itself, use `--log-level` or `LAB_HARNESS_LOG_LEVEL`; `[logging] level`
takes effect everywhere after loading.

An unusable level name is a startup error, like any other bad config value:

```
error: Unknown log level 'loud'. Available: debug, info, warning, error.
```

Only `lab_harness.*` loggers are configured. `--log-level debug` means this
program, not httpx and the Anthropic SDK as well. Importing `lab_harness` as a
library installs no handler at all — only the CLI calls `configure_logging`.

**Credential values are never logged.** The provider logs the *name* of the
variable a credential came from, which is the useful fact when two are set, and
the same applies to `ANTHROPIC_BASE_URL` — a private endpoint often enough to
treat as sensitive. A profile says where both came from the same way:
`Profile deepseek: key from DEEPSEEK_API_KEY, endpoint from base_url`. There are
tests asserting a key and an endpoint set in the environment do not appear in
debug output.

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

## Project instructions

A `HARNESS.md` or `AGENTS.md` in the working directory is appended to the
system prompt, following the convention pi and other harnesses use. A project
that already has one needs no setup; interactive mode prints a dim line naming
the file it loaded, so a feature that changes the agent's behavior is never
silent.

```toml
[project]
context_files = ["HARNESS.md", "AGENTS.md"]
max_bytes = 32000
```

**The first name that exists wins** — the rest are not read. Two files would
be two sets of instructions with no stated precedence, and concatenating them
would silently double any advice they share.

"Exists" is the whole test, so a **blank `HARNESS.md` means no instructions**
rather than "fall through to `AGENTS.md`". That is one rule with no special
case, and it doubles as the way to opt a project out of an `AGENTS.md` written
for some other tool.

The file is read from `--cwd`, like config, so `lab-harness --cwd ../other-project`
picks up that project's conventions.

**It is capped.** These instructions ride in the system prompt on *every*
request, so an unbounded file is an unbounded bill. Past `max_bytes` the file
is truncated and the terminal says so.

A file that exists but cannot be *read* is a startup error rather than a
silent skip — the same reasoning as a malformed config file. Instructions you
meant to apply and that quietly did nothing are the one failure this feature
must not have.

Changing the file changes the system prompt — so `--resume` on an older
session will warn that the prompt has moved. That is the warning working, not
a bug; the transcript records which prompt each request was sent with. How the
instructions are wrapped is `prompts.project_instructions` — see
[Prompts](#prompts).
