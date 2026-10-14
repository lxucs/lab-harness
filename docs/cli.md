# CLI

## Modes

```bash
lab-harness                      # interactive
lab-harness -p "fix the lints"   # one-shot, plain text
lab-harness -p "..." --json      # one-shot, JSONL event stream
lab-harness --resume             # continue the most recent session
```

All of these are **subscribers to the same event stream**. The interactive
terminal and the JSONL writer differ only in what they do with events, not in
what the runtime does.

## Flags

| Flag | Purpose |
|---|---|
| `-p`, `--prompt` | Run one prompt and exit. Omit it to start the interactive terminal. |
| `--json` | Emit the raw event stream as JSONL instead of rendered text. |
| `--model` | Model id. Beats `$LAB_HARNESS_MODEL`, which beats `agent.model` in config. |
| `--provider` | `anthropic` or a configured `providers.openai` profile. Normally inferred from the model id. |
| `--cwd` | Working directory for tools, config discovery, *and* project instructions. Defaults to the current directory. |
| `--log-level` | `debug`, `info`, `warning` or `error`. Beats `$LAB_HARNESS_LOG_LEVEL`, which beats `logging.level` in config. |
| `--yes` | Skip tool approval prompts in interactive mode. |
| `--resume [ID]` | Continue a session. With no id, the most recent one here. |
| `--sessions` | List saved sessions for this directory and exit. |
| `--no-save` | Do not write a transcript for this run. |

**Exit status is meaningful.** A run whose final turn ended in `error` or
`aborted` exits `1`; a clean run exits `0`; `Ctrl-C` exits `130`. A script
piping this needs to tell a refusal from an outage.

In plain-text mode, assistant text goes to **stdout** while tool-call notices
and errors go to **stderr**, so `lab-harness -p "…" > answer.txt` captures the
answer alone. Diagnostics follow the same rule — see
[Diagnostics](#diagnostics) below.

## Interactive

prompt_toolkit handles input: history, multi-line entry, and key bindings. rich
handles output: streaming text, markdown with highlighted code, and diffs.

Output is **inline** — it lands in real terminal scrollback rather than being
trapped inside a full-screen app, so scrolling back through a long session works
the way it does with any other command.

```
› why does the retry test fail?

⏺ I'll read the test first.

⏺ read(tests/test_http.py)
  ⎿  1  """Retries against a fake transport."""
     2
     3  import httpx
     … 38 more line(s)

✻ Thought for 4s · Ctrl-O to expand

⏺ The backoff cap is applied after the jitter. Moving it.

⏺ edit(lab_harness/llm/providers/_http.py)
  ⎿  Replaced 1 occurrence(s) in lab_harness/llm/providers/_http.py  +2 -1
    41       for attempt in range(retries):
    42 -         delay = jitter(base * 2**attempt)
    42 +         delay = min(base * 2**attempt, cap)
    43 +         delay = jitter(delay)
    44           await asyncio.sleep(delay)

✶ Working… (12s · Ctrl-C to cancel)       ← only while the run is in flight
```

The layout follows Claude Code's. A `⏺` opens each message and each tool call;
a call's bullet turns green when it succeeds and red when it fails, and its
output hangs under `⎿`, collapsed past `terminal.max_result_lines`. Colors
come from [`[terminal.theme]`](configuration.md#the-terminal).

**Edits show their diff**, as Claude Code's do: a line number, `-` or `+`, and
the line, removed lines on a red bar and added lines on a green one that runs
to the edge of the terminal, `⋮` between separate hunks. Collapsed past
`terminal.max_diff_lines` (40). `edit` and `write` compute the diff and hand
it over in their result's `details` — a standard unified diff, never sent to
the model — with `tools.diff.context_lines` (3) unchanged lines around each
change. A write that creates a file shows it as all additions.

**Two kinds of output.** What is finished is printed once and stays in
scrollback. What is still in flight — the message streaming in, running tools
with their latest line of output, the spinner — is drawn in one region below
it, redrawn in place and erased when the run ends. Erased, because text
already written to a scrollback terminal cannot be taken back: a region that
stayed behind would leave a second copy of every answer. Approval questions
take the region down while they wait, so the question is never drawn over.

### Thinking

A model's reasoning streams as it arrives, in dim italics under
`✻ Thinking…`. When the message ends, `terminal.thinking` decides what stays:

| Mode | In scrollback | While streaming |
|---|---|---|
| `collapse` (default) | One line: `✻ Thought for 4s · Ctrl-O to expand` | The last `thinking_preview_lines` lines |
| `full` | The whole thought, above its answer | All of it that fits |
| `hidden` | Nothing | Only the spinner |

**Ctrl-O** switches between `full` and the configured mode for the rest of the
session. Switching to `full` also prints what the last run collapsed, once.
Collapse is the default because raw reasoning — DeepSeek's and GLM's — often
runs to pages and buries the answer; summaries are short enough that `full`
suits them.

What there is to show depends on the provider: DeepSeek and GLM stream their
real chain of thought; OpenAI's Responses API and Claude send a summary; OpenAI
via Chat Completions sends nothing. Claude Opus 5 and Sonnet 5 return empty
thinking unless asked for a summary, which the adapter does by default —
`providers.anthropic.thinking_display`, see
[providers.md](providers.md#the-thinking-request).

Ctrl-O is a prompt key binding. Pressed mid-run it waits in the input queue and
takes effect when the prompt returns; nothing reads keys during a run, which is
also what keeps text typed ahead safe. (A cooked-mode terminal on macOS and the
BSDs would otherwise treat Ctrl-O as DISCARD and silently drop the run's
output, so the terminal turns that one character off for the run's length.)

| Key | Action |
|---|---|
| `Ctrl-C` | Cancel the in-flight turn; the session survives |
| `Ctrl-O` | Toggle thinking between shown in full and the configured mode |
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
$ lab-harness -p "what files are here?" --json
{"type": "agent_start"}
{"type": "turn_start"}
{"message": {"content": [], "model": "...", "stop_reason": "stop", ...}, "type": "message_start"}
{"tool_call_id": "c1", "tool_name": "ls", "arguments": {}, "type": "tool_execution_start"}
{"tool_call_id": "c1", "tool_name": "ls", "content": [...], "is_error": false, "duration_ms": 1, "details": {"path": ".", "entries": 4, "total_entries": 4}, "type": "tool_execution_end"}
{"type": "turn_end", ...}
{"type": "agent_end", "messages": [...]}
```

Pipe it into `jq`, or use it as the assertion surface for end-to-end tests —
checking a structured event stream beats scraping rendered terminal output.

Every line round-trips back into real event objects through
`lab_harness.llm.serde`, which is what makes a consumer in another language
possible. See [events.md](events.md).

## Sessions

Every run is saved, per project, under `~/.lab-harness/sessions/`:

```bash
lab-harness --sessions            # what is saved for this directory
lab-harness --resume              # continue the most recent one
lab-harness --resume 20261010T11  # a specific one; a unique prefix works
lab-harness -p "…" --no-save      # run without persisting
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
lab-harness --model claude-opus-5          # Anthropic, Messages API
lab-harness --model gpt-5                  # OpenAI, Responses API
lab-harness --model deepseek-flash         # DeepSeek, Chat Completions
lab-harness --model glm-5.3                # GLM, Chat Completions
lab-harness --model their-model --provider groq
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
lab-harness --log-level debug -p "…"              # trace this run
LAB_HARNESS_LOG_LEVEL=debug lab-harness           # trace this shell
lab-harness --log-level debug 2> lab-harness.log  # interactive, trace kept apart
```

Logs go to **stderr**, so they never mix into the answer on stdout or the JSONL
stream. Nothing is printed by default beyond genuine problems: a model id
missing from the catalog, project instructions truncated, `max_turns`
exhausted with the model still working, a failed provider request, and — in
`-p` mode, which has no terminal to show them in — a resumed session's
warnings.

In the interactive terminal they print in scrollback, dimmed, above the live
region — straight to the terminal, a record would land inside the region and
break its next redraw. To keep them out of the conversation, redirect stderr
and `tail -f lab-harness.log` in another pane.

`info` adds the decisions — which model, from where, which provider and why —
and one line per run. `debug` adds the trace: every request, response and tool
call, with the ids that find each one in the transcript and in `--json`:

```
DEBUG lab_harness.agent.loop: Turn 1/50: 1 message(s) to claude-sonnet-5
DEBUG lab_harness.llm.providers.anthropic: Request to claude-sonnet-5: max_tokens=64000 messages=1 system=429 tools=9 thinking={"type":"adaptive","display":"summarized"}
DEBUG lab_harness.llm.providers.anthropic: Response msg_01ExampleId from claude-sonnet-5
DEBUG lab_harness.llm.providers.anthropic: Response stop_reason=tool_use
DEBUG lab_harness.agent.loop: Response stop=tool_use thinking=212 text=31 tool_calls=1 in=1520 cache_read=0 cache_write=0 out=96 time=2.1s first_token=0.8s
DEBUG lab_harness.agent.loop: Tool call toolu_01ExampleId: bash(command)
DEBUG lab_harness.cli.approval: Tool call toolu_01ExampleId: approval answered 'yes'
DEBUG lab_harness.agent.loop: Tool call toolu_01ExampleId: ok in 1515ms exit_code=0 truncated=False
DEBUG lab_harness.agent.loop: Turn 2/50: 3 message(s) to claude-sonnet-5
…
INFO lab_harness.agent.loop: Run finished: requests=2 tool_calls=1 failed=0 stop=stop in=3170 out=180 time=4.7s
```

The stop reason appears twice on purpose: the adapter's is what the vendor
sent, the loop's is what it was mapped to — and a `refusal` or a
`content_filter` maps to a plain `stop`. Content never appears: requests are
logged by size, tool calls by their argument *names*.

`debug` also traces config resolution, which is what you want when a setting
appears to do nothing — every file read or skipped, in precedence order:

```
DEBUG lab_harness.config: No config file at /home/you/.lab-harness/config.toml
DEBUG lab_harness.config: Applying config from /repo/.lab-harness/config.toml
DEBUG lab_harness.config: No config file at /repo/.lab-harness/config.local.toml
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
⏺ bash(rm -rf build/)
Allow bash? [y]es / [n]o / [a]lways
```

The call line is cut to one short line, so a longer command is printed in
full beneath it — approving a command shown cut off is approving something
unseen. For the same reason an `edit` shows the change it will make, and a
`write` the content it will write:

```
⏺ edit(lab_harness/config.py)
          max_turns: int = 50
    -     parallel_tools: bool = False
    +     parallel_tools: bool = True
Allow edit? [y]es / [n]o / [a]lways
```

The edit preview has no line numbers: it is built from the call's arguments,
and where the text sits in the file is not known until the tool runs. The
result that follows has them.

This is a `before_tool_call` hook, not loop logic — see
[tools.md](tools.md#approval) for the full contract.

**It fails closed.** Only `y`/`yes` and `a`/`always` proceed; a bare Enter, an
unrecognized answer, Ctrl-C, and Ctrl-D all deny. `always` is remembered for the
session, never persisted.

Approval is **skipped in `-p` mode**, which means one-shot runs execute mutating
tools without asking — there is nobody at the keyboard to answer. Use `-p`
deliberately, and `--yes` only when you mean it.
