# CLI

## Modes

```bash
harness                      # interactive
harness -p "fix the lints"   # one-shot, plain text
harness -p "..." --json      # one-shot, JSONL event stream
harness --resume             # continue the most recent session
```

All of these are **subscribers to the same event stream**. The interactive
terminal and the JSONL writer differ only in what they do with events, not in
what the runtime does.

## Flags

| Flag | Purpose |
|---|---|
| `-p`, `--prompt` | Run one prompt and exit. Omit it to start the interactive terminal. |
| `--json` | Emit the raw event stream as JSONL instead of rendered text. |
| `--model` | Model id. Beats `$HARNESS_MODEL`, which beats `agent.model` in config. |
| `--provider` | `anthropic` or a configured `providers.openai` profile. Normally inferred from the model id. |
| `--cwd` | Working directory for tools, config discovery, *and* project instructions. Defaults to the current directory. |
| `--log-level` | `debug`, `info`, `warning` or `error`. Beats `$HARNESS_LOG_LEVEL`, which beats `logging.level` in config. |
| `--yes` | Skip tool approval prompts in interactive mode. |
| `--resume [ID]` | Continue a session. With no id, the most recent one here. |
| `--sessions` | List saved sessions for this directory and exit. |
| `--no-save` | Do not write a transcript for this run. |

**Exit status is meaningful.** A run whose final turn ended in `error` or
`aborted` exits `1`; a clean run exits `0`; `Ctrl-C` exits `130`. A script
piping this needs to tell a refusal from an outage.

In plain-text mode, assistant text goes to **stdout** while tool-call notices
and errors go to **stderr**, so `harness -p "…" > answer.txt` captures the
answer alone. Diagnostics follow the same rule — see
[Diagnostics](#diagnostics) below.

## Interactive

prompt_toolkit handles input: history, multi-line entry, and key bindings. rich
handles output: streaming text, markdown, and syntax-highlighted diffs.

Output is **inline** — it lands in real terminal scrollback rather than being
trapped inside a full-screen app, so scrolling back through a long session works
the way it does with any other command.

```
› read the config and summarize

● read(harness/config.py)
  └ 1   """Configuration schema and defaults.
    2
    3   This module is the single list of every tunable…
    … 38 more line(s)

The config defines three groups of settings…

› ▊
```

Streaming text renders through `rich.Live`, which redraws one region in place
and leaves the finished answer in scrollback. Printing deltas and then
re-rendering the final message would show the answer twice — output already
written to a scrollback terminal cannot be taken back.

| Key | Action |
|---|---|
| `Ctrl-C` | Cancel the in-flight turn; the session survives |
| `Ctrl-D` | Exit |
| `Enter` | Submit |
| `Alt-Enter` | Newline without submitting |

| Command | Action |
|---|---|
| `:status` | Token, cache and cost totals for the session |
| `:q`, `:quit`, `exit` | Exit |

The agent runs one turn sequence at a time. Typing while it works is buffered by
the terminal and submitted when the run finishes — the runtime itself refuses a
second concurrent `prompt()` rather than coordinating with the one in flight.

### Usage and cost

Each run ends with one dim line covering every request it made:

```
45.2k in · 1.3k out · 82% cached · context 15.1k of 1M (2%) · $0.0421
```

| Part | Meaning |
|---|---|
| `in` | Prompt tokens read across the run's requests, cached or not |
| `out` | Tokens generated |
| `cached` | Share of `in` served from the provider's prompt cache |
| `context` | Size of the latest request — prompt plus output — against the model's window |
| cost | From [`pricing`](configuration.md#pricing); left out for an unpriced model |

`context` is the latest request, not a sum: every request resends the history,
so adding them up would count it again on each turn. `in` *is* a sum, because
that is what is billed.

`:status` shows the same figures for the whole session, with the prompt split
into uncached, cache-read and cache-write tokens:

```
model      claude-sonnet-5 via anthropic
requests   12
input      450.4k (38.1k uncached · 392k cache read · 20.3k cache write)
output     5.2k
cache hit  87%
context    48k of 1M (5%)
cost       $0.3120
```

Both are computed from the transcript rather than counted as events arrive —
every assistant message carries its request's `Usage` — so after `--resume`
the totals include the restored history.

Set `terminal.show_usage = false` to drop the per-run line; `:status` still
works. The line is interactive-only: in `-p` mode stdout is the answer, and
`--json` already carries `usage` on every assistant `message_end`.

**Anthropic cache hits read 0% for now.** Anthropic only caches a prompt that
marks where to, and the adapter does not send those markers yet — see the
[TODO in providers.md](providers.md#the-anthropic-adapter). OpenAI and
DeepSeek cache repeated prefixes on their own and report the hits, so their
figures are real. Another Chat Completions vendor counts only if it reports
`prompt_tokens_details.cached_tokens`, the field the adapter reads.

## JSON mode

One event per line, flushed as it happens:

```bash
$ harness -p "what files are here?" --json
{"type": "agent_start"}
{"type": "turn_start"}
{"message": {"content": [], "model": "...", "stop_reason": "stop", ...}, "type": "message_start"}
{"tool_call_id": "c1", "tool_name": "ls", "arguments": {}, "type": "tool_execution_start"}
{"tool_call_id": "c1", "tool_name": "ls", "content": [...], "is_error": false, "duration_ms": 1, "type": "tool_execution_end"}
{"type": "turn_end", ...}
{"type": "agent_end", "messages": [...]}
```

Pipe it into `jq`, or use it as the assertion surface for end-to-end tests —
checking a structured event stream beats scraping rendered terminal output.

Every line round-trips back into real event objects through
`harness.llm.serde`, which is what makes a consumer in another language
possible. See [events.md](events.md).

## Sessions

Every run is saved, per project, under `~/.harness/sessions/`:

```bash
harness --sessions            # what is saved for this directory
harness --resume              # continue the most recent one
harness --resume 20261010T11  # a specific one; a unique prefix works
harness -p "…" --no-save      # run without persisting
```

Resume continues writing to the *same* file, so a conversation stays one
transcript across many invocations. Sessions are scoped to the working
directory — resuming in one project never surfaces another's history.

Each invocation also records the system prompt, tool declarations, model and
config (credentials redacted) it ran with, so every request can be rebuilt
from the file. Format, trimming of interrupted turns, rebuilding trajectories,
and the reserved `compaction` entry: [sessions.md](sessions.md).

## Project instructions

A `HARNESS.md` or `AGENTS.md` in the working directory is appended to the
system prompt, so a project's own conventions do not have to be rediscovered
from the code every run. Interactive mode says which file it used:

```
Project instructions: AGENTS.md
```

The first name that exists wins, the file is read from `--cwd`, and it is
capped at `project.max_bytes`. Full behavior:
[configuration.md](configuration.md#project-instructions).

## Choosing a model

```bash
harness --model claude-opus-5          # Anthropic, Messages API
harness --model gpt-5                  # OpenAI, Responses API
harness --model deepseek-flash         # DeepSeek, Chat Completions
harness --model glm-5.3                # GLM, Chat Completions
harness --model their-model --provider groq
```

The provider is inferred from the model id, so `--provider` is only needed for
an id no profile claims. OpenAI, DeepSeek and GLM are configured out of the
box; any other endpoint speaking either protocol is a few lines of config —
[configuration.md](configuration.md#provider-profiles).

Reasoning models differ in what they will show you and what has to be handed
back; [providers.md](providers.md#how-thinking-is-handled-across-models) has
the table.

## Diagnostics

```bash
harness --log-level debug -p "…"     # trace this run
HARNESS_LOG_LEVEL=debug harness      # trace this shell
```

Logs go to **stderr**, always, so they never mix into the answer on stdout or
the JSONL stream. Nothing is printed by default beyond genuine problems: a
model id missing from the catalog, project instructions truncated, `max_turns`
exhausted with the model still working, a failed provider request, and — in
`-p` mode, which has no terminal to show them in — a resumed session's
warnings.

`debug` adds the trace you need when a setting appears to do nothing — every
config file read or skipped, in precedence order:

```
DEBUG harness.config: No config file at /home/you/.harness/config.toml
DEBUG harness.config: Applying config from /repo/.harness/config.toml
DEBUG harness.config: No config file at /repo/.harness/config.local.toml
```

Levels, what each one adds, and why a level set in a config file cannot report
that same file being read:
[configuration.md](configuration.md#logging).

## Web tools

`web_search` and `web_fetch` are enabled by default in the interactive
terminal. Search needs a `TAVILY_API_KEY` (or `backend = "ddgs"` in config);
fetch needs nothing.

Neither is approval-gated. They are read-only with respect to your machine, and
prompting before every search would make research unusable — but note that a
search query does leave the machine.

A missing key is not a crash: the tool returns an error naming both places to
set one, and the model carries on without it.

## Tool approval

Mutating tools (`bash`, `write`, `edit`) prompt before executing:

```
● bash(rm -rf build/)
Allow bash? [y]es / [n]o / [a]lways
```

This is a `before_tool_call` hook, not loop logic — see
[tools.md](tools.md#approval) for the full contract.

**It fails closed.** Only `y`/`yes` and `a`/`always` proceed; a bare Enter, an
unrecognized answer, Ctrl-C, and Ctrl-D all deny. `always` is remembered for the
session, never persisted.

Approval is **skipped in `-p` mode**, which means one-shot runs execute mutating
tools without asking — there is nobody at the keyboard to answer. Use `-p`
deliberately, and `--yes` only when you mean it.
