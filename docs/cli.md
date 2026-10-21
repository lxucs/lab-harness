# CLI

## Modes

```bash
lab-harness                      # interactive
lab-harness -p "fix the lints"   # one-shot, plain text
lab-harness -p "..." --json      # one-shot, JSONL event stream
lab-harness --resume ID          # continue a session; --resume alone lists them
```

All of these are **subscribers to the same event stream**. The interactive
terminal and the JSONL writer differ only in what they do with events, not in
what the runtime does.

## Flags

| Flag | Purpose |
|---|---|
| `-p`, `--prompt` | Run one prompt and exit. Omit it to start the interactive terminal. |
| `--json` | Emit the raw event stream as JSONL instead of rendered text. One-shot mode only — an interactive session renders instead |
| `--model` | Model id. Beats `$LAB_HARNESS_MODEL`, which beats `agent.model` in config. |
| `--provider` | `anthropic` or a configured `providers.openai` profile. Normally inferred from the model id. |
| `--reasoning-effort` | How hard a reasoning model thinks, in its provider's terms — `low`, `high`, …. Beats the model's `reasoning_effort` and `agent.reasoning_effort`; `""` sends none. See [Reasoning effort](#reasoning-effort). |
| `--cwd` | Working directory for tools, config discovery, *and* project instructions. Defaults to the current directory; `~` is expanded, and a path that is not a directory is a startup error |
| `--tools` | Comma-separated tools to enable, e.g. `read,ls,grep,glob`. Beats `tools.enabled` in config; `""` enables none. |
| `--skills` | Comma-separated skills to make available, e.g. `commit,release`. Beats `skills.selected` in config; `""` makes none available. |
| `--system-prompt` | Replaces `prompts.system` for this run. The other sections, project instructions included, are still sent. A template like any other section, so `{agent_model}` in it is filled |
| `--system-prompt-file` | The same, read from a file. Exclusive with `--system-prompt`. |
| `--log-level` | `debug`, `info`, `warning` or `error`. Beats `$LAB_HARNESS_LOG_LEVEL`, which beats `logging.level` in config. |
| `--yes` | Skip tool approval prompts in interactive mode. |
| `--resume [ID]` | Continue the session with this id; a unique prefix works. Interactive mode prints the saved conversation on resuming. With no id, list this directory's sessions and exit. |
| `--export [ID]` | Write the session as one HTML page beside its transcript, print its path, and exit. With no id, this directory's most recent session. |
| `--no-save` | Do not write a transcript for this run. |

**Exit status is meaningful.** A run whose final turn ended in `error` or
`aborted` exits `1`; a clean run exits `0`; `Ctrl-C` exits `130`. A script
piping this needs to tell a refusal from an outage — and the words a cancelled
run had reached are on stdout either way.

In plain-text mode, assistant text goes to **stdout** while tool-call notices
and errors go to **stderr**, so `lab-harness -p "…" > answer.txt` captures the
answer alone. Diagnostics follow the same rule — see
[Diagnostics](#diagnostics) below.

## Interactive

prompt_toolkit handles input: history, multi-line entry, and key bindings. rich
handles output: streaming text, markdown with highlighted code, and diffs.

A session opens with a box saying what is running, where, and how hard it is
asked to think, before anything else it has to say:

```
╭───────────────────────────────────────────────────────────╮
│ lab-harness · gpt-6-luna via openai · effort medium       │
│ ~/projects/my-project                                     │
│ / for commands · Ctrl-C cancels · Ctrl-O toggles thinking │
╰───────────────────────────────────────────────────────────╯
```

Output is **inline** — it lands in real terminal scrollback rather than being
trapped inside a full-screen app, so scrolling back through a long session works
the way it does with any other command.

```
❯ why does the retry test fail?

⏺ I'll read the test first.

⏺ Read(tests/test_http.py)
  ⎿  1  """Retries against a fake transport."""
     2
     3  import httpx
     … +38 lines

✻ Thought for 4s · Ctrl-O to expand

⏺ The backoff cap is applied after the jitter. Moving it.

⏺ Edit(lab_harness/llm/providers/_http.py)
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
output hangs under `⎿`, collapsed past `terminal.max_result_lines` into
`… +N lines`. Colors come from
[`[terminal.theme]`](configuration.md#the-terminal).

**A tool is named the way Claude Code names it** — `Read(path)`,
`WebSearch(query)` — while the name the model calls, and the one the transcript
records, stays `read` and `web_search`. Only the label changes, so a live call
and the same call replayed after `--resume` read identically.

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
via Chat Completions sends nothing. Claude Fable 5, Opus 5 and Sonnet 5 return
empty thinking unless asked for a summary, which the adapter does by default —
`providers.anthropic.thinking_display`, see
[providers.md](providers.md#the-thinking-request). Some gateways also drop the
last few words of reasoning when streaming — the same request sent unstreamed
ends cleanly — so a thought can stop mid-sentence where the answer begins.
Expanding shows everything that arrived, and nothing is cut on this side; the
transcript holds the same text, and it is what goes back on the next turn.

Ctrl-O is a prompt key binding. Pressed mid-run it waits in the input queue and
takes effect when the prompt returns; nothing reads keys during a run, which is
also what keeps text typed ahead safe. For the run's length the terminal
neither echoes keys nor treats Ctrl-O as DISCARD. An echoed key would land
inside the live region and leave stale lines behind its redraws, and DISCARD,
on macOS and the BSDs, would silently drop the run's output.

| Key | Action |
|---|---|
| `Ctrl-C` | Cancel the in-flight turn; the session survives. What it had reached stays on screen and in the transcript |
| `Ctrl-O` | Toggle thinking between shown in full and the configured mode |
| `Ctrl-D` | Exit |
| `Enter` | Submit |
| `Alt-Enter` | Newline without submitting |

The agent runs one turn sequence at a time. Typing while it works is buffered by
the terminal, unechoed, and appears at the next prompt — submitted at once if
it ended with Enter. The runtime itself refuses a second concurrent `prompt()`
rather than coordinating with the one in flight.

**What you type is kept**, in `~/.lab-harness/history` by default, so `Up` in a
later session recalls an earlier one's prompts. Set `session.enabled = false`
and nothing is persisted, history included.

**The model and how full its window is sit at the right of the input**, dim,
and turn to `terminal.theme.warning` past `terminal.context_warning` (85%):

```
❯ what next?                                   gpt-6-luna · context 12%
```

It is a float over the input's first row, so it takes no row of its own and
steps aside for long input rather than squeezing it — which is why it is there
and not on a status bar under the prompt. A question asked through the same
session, an approval for instance, draws none.

### Commands

A line that is a `/` and a name is a command, which the terminal runs itself
rather than sending:

| Command | Action |
|---|---|
| `/status` | Token, cache and cost totals for the session |
| `/effort [LEVEL]` | Show the reasoning effort, or set it for the runs after; `/effort default` sends none |
| `/model [ID]` | Show the model, or switch to another one this session's provider serves |
| `/tools [NAMES]` | Show the tools this session has, or choose them — `read,skill`; `none` is none |
| `/skills [NAMES]` | List the skills that apply, or choose them — `commit,release`; `none` is none |
| `/clear` | Forget the conversation and start over; the transcript keeps what came before |
| `/help` | List the commands |
| `/exit`, `/quit`, `exit` | Exit |

**Typing `/` lists them**, as Claude Code does: above the prompt, every command
the input could be the start of, by name or alias, each beside its description,
the first highlighted. **So does every skill:** each is a command of its own,
added when the session starts, so `/commit` — or Tab and Enter on it in the
list — sends that skill's instructions as the next turn. `/skills` prints what
is there:

```
commit — Write a commit message for staged changes   allows: bash
release — Cut a release
Invoke one with /name, or ask for it by name.
```

A skill whose name is already a command's is left out rather than shadowing it.

```
  /effort [level]  Show or set the reasoning effort; "default" sends none
  /exit (quit)     Exit the session
❯ /e
```

The input moves down to make room and back up when the list closes. On the
terminal's bottom row the room comes from scrolling earlier output up, which
cannot be undone, so there the input ends up a few rows higher until the next
output fills the gap.

| Key | In the list |
|---|---|
| `Up`, `Down` | Move the highlight; the input is left as typed |
| `Tab` | Put the highlighted command in the input, ready for arguments |
| `Enter` | Run the highlighted command — `/st` and Enter is `/status` |
| `Esc` | Close the list; typing reopens it |

Typed ahead during a run, `/st` and Enter still runs `/status` when the prompt
returns.

A path is not a command: `/tmp/notes.md what is this?` fails at its second
slash and goes to the model. A name that is no command, such as `/stauts`, is
reported rather than sent.

Every command is an entry in one registry, `Commands` in
[`cli/commands.py`](../lab_harness/cli/commands.py), which both runs and lists
it. A new one is a name, a one-line description and an async handler given the
rest of the line, added with `terminal.commands.add(Command(...))` — it is
listed, completed and run with nothing else to change.

`/model` stays inside the provider the session started on: which provider
serves a request is decided once, when the agent is built, so a model another
one serves is refused with the suggestion to start a new session with `--model`
rather than sent somewhere it does not belong. `/clear` empties the
conversation without ending the session — the transcript on disk keeps what
came before, and the next run opens a new segment in it.

### Usage and cost

Each run ends with one dim line covering every request it made:

```
45.2k in · 1.3k out · 82% cached · effort medium · context 15.1k of 1M (2%) · $0.0421
```

| Part | Meaning |
|---|---|
| `in` | Prompt tokens read across the run's requests, cached or not |
| `out` | Tokens generated |
| `cached` | Share of `in` served from the provider's prompt cache |
| `effort` | The [reasoning effort](#reasoning-effort) the requests went out with; `default` when none is sent, and left out for a model that does not reason |
| `context` | Size of the latest request — prompt plus output — against the model's window |
| cost | From the model's [`price`](configuration.md#prices); left out for an unpriced model |

`context` is the latest request, not a sum: every request resends the history,
so adding them up would count it again on each turn. `in` *is* a sum, because
that is what is billed.

`/status` shows the same figures for the whole session, with the prompt split
into uncached, cache-read and cache-write tokens:

```
model      gpt-6-luna via openai
effort     medium
requests   12
input      450.4k (58.4k uncached · 392k cache read · 0 cache write)
output     5.2k
cache hit  87%
context    48k of 1.1M (5%)
cost       $0.0124
```

Both are computed from the transcript rather than counted as events arrive —
every assistant message carries its request's `Usage` — so after `--resume`
the totals include the restored history.

Past `terminal.context_warning` (85% of the window) the run's line is followed
by a warning, and the status beside the input turns from dim to
`terminal.theme.warning`. Nothing is compacted: the harness does not rewrite
history behind your back, so the warning says what to do instead — `/clear`, or
a new session.

Set `terminal.show_usage = false` to drop the per-run line; `/status` still
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
{"system_prompt": "...", "tools": [...], "model": {"id": "gpt-6-luna", ...}, "history": 0, "type": "agent_start"}
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
lab-harness --resume              # what is saved for this directory
lab-harness --resume 20261010T11  # continue one; a unique prefix works
lab-harness -p "…" --no-save      # run without persisting
```

Resume continues writing to the *same* file, so a conversation stays one
transcript across many invocations. Sessions are scoped to the working
directory — resuming in one project never surfaces another's history.

The interactive terminal prints the saved conversation before the first
prompt, drawn exactly as new turns are and expandable with Ctrl-O, headed by a
rule that says it is old: `── Resumed 20261010T11 — 12 message(s) ──`. With
`--no-save` it is a read-only view — nothing is appended to the file.

`--export [ID]` writes the session as one self-contained HTML page beside its
transcript and prints its path — the conversation, every segment's context,
and tokens and cost per request, opening from disk with nothing fetched.

Each invocation also records the system prompt, tool declarations, model and
config (credentials redacted) it ran with, and records them again whenever one
changes mid-session, so every request can be rebuilt from the file. Format,
trimming of interrupted turns, rebuilding trajectories, and the reserved
`compaction` entry: [sessions.md](sessions.md).

## The system prompt

What every request opens with: the sections of `[prompts]` in config — who the
model is, how the harness works, the environment it runs in, context
management, the skills found — and last the project's instructions. It names no
tools: what the model can call comes from the tool declarations alone, so
narrowing `--tools` never leaves the prompt describing a tool that is not
there. The text ships in
[`lab_harness/prompts/prompts.toml`](../lab_harness/prompts/prompts.toml).

```bash
lab-harness --system-prompt "Answer in one sentence."
lab-harness --system-prompt-file prompts/terse.md
```

Either replaces `prompts.system` for one run; the other sections and the
project's instructions are still sent. The file is read relative to the shell's
directory, not `--cwd`, and its surrounding whitespace dropped. To change a
section for good, or leave one out, set it in config:
[configuration.md](configuration.md#prompts).

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

## Skills

Instructions for a kind of task, loaded when the model asks for one by name:
the model sees their names and what each is for, and the instructions arrive
only when it calls `skill`. The format, `allowed-tools` and the config:
[tools.md](tools.md#skills).

**Each is also a command** — `/commit [request]` sends one's instructions as
the next turn, and `/skills` lists them with what each allows — see
[Commands](#commands) above.

**Which apply is yours to choose**, at any of the three levels:
`skills.selected` in config, `--skills` for one run, and `/skills` mid-session;
`--skills ""` and `/skills none` make none available. The same for tools:
`tools.enabled`, `--tools` and `/tools`. A change mid-session rebuilds the
system prompt, so what the model is offered and what it can ask for never
disagree — and the next run's `agent_start` carries the new declarations, so a
recorded session opens a new segment there.

## Choosing a model

```bash
lab-harness --model claude-opus-5          # Anthropic, Messages API
lab-harness --model gpt-6-luna             # OpenAI, Responses API
lab-harness --model deepseek-flash         # DeepSeek, Chat Completions
lab-harness --model glm-5.3                # GLM, Chat Completions
lab-harness --model their-model --provider groq
```

The provider is inferred from the model id, so `--provider` is only needed for
an id the catalog does not list. OpenAI, DeepSeek and GLM are configured out of
the box, in [`lab_harness/llm/catalog.toml`](../lab_harness/llm/catalog.toml);
any other endpoint speaking either protocol is a few lines of config —
[configuration.md](configuration.md#provider-profiles).

Reasoning models differ in what they will show you and what has to be handed
back; [providers.md](providers.md#how-thinking-is-handled-across-models) has
the table.

### Reasoning effort

How hard a reasoning model thinks is one setting, resolved like the model
itself — highest first:

1. `/effort LEVEL` in the terminal, for the runs after it
2. `--reasoning-effort LEVEL`
3. the model's `reasoning_effort` in `[models]`
4. `agent.reasoning_effort`

Empty everywhere — the default — sends nothing, and the model thinks at its
provider's default. `--reasoning-effort ""` and `/effort default` mean the same
for one run or the rest of a session, whatever config says.

```bash
lab-harness --reasoning-effort low            # quick answers
lab-harness --model glm-5.3 --reasoning-effort high
```

```toml
[agent]
reasoning_effort = "medium"     # any model that sets none

[models."qwen3.8-flash"]
reasoning_effort = "low"        # this one, wherever it is used
```

**A level is the provider's word, sent as is.** Chat Completions gets
`reasoning_effort`, the Responses API `reasoning.effort`, and Claude
`output_config.effort`; a Claude model older than 4.6 has no effort, so any
level turns on its budgeted thinking instead
([providers.md](providers.md#the-thinking-request)). Vendors disagree on the
words — some take `none` to switch thinking off, some levels beyond `high` —
so nothing here checks them: a server that rejects one usually says which it
takes, in the error. A model whose entry sets `supports_reasoning = false` is
sent no level at all.

A level is part of the model a request goes to, so a session records it with
the model in each `context` entry: a changed level opens a new segment, the
export page marks it, and a rebuilt request body carries it. The current one is
on screen throughout — in the box a session opens with, in the line that ends
each run, and in `/status`. A model that does not reason gets no level, and no
place claims one.

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

`info` adds the decisions — which model, from where, which provider and why,
which tools, which system prompt — and one line per run. `debug` adds the trace: every request, response and tool
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
DEBUG lab_harness.config: No config file at /home/you/.lab-harness/config.local.toml
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
⏺ Bash(rm -rf build/)
Allow bash? [y]es / [n]o / [a]lways
```

The call line is cut to one short line, so a longer command is printed in
full beneath it — approving a command shown cut off is approving something
unseen. For the same reason an `edit` shows the change it will make, and a
`write` the content it will write:

```
⏺ Edit(lab_harness/config.py)
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
unrecognized answer, Ctrl-C, and Ctrl-D all deny. Ctrl-C denies *and* ends the
turn, the way it does anywhere else in a run: refusing one call and carrying on
would spend another request on a run the user walked out of, and ask again for
the next call in the same batch. `always` is remembered for the
session, never persisted. A skill's `allowed-tools` grants the same way, for the
rest of the session, and the terminal names what it granted — see
[tools.md](tools.md#skills). Setting `tools.approval.allow_remember` to false
suppresses that grant too.

Approval is **skipped in `-p` mode**, which means one-shot runs execute mutating
tools without asking — there is nobody at the keyboard to answer. Use `-p`
deliberately, and `--yes` only when you mean it. To keep a tool out of a run
altogether, leave it out of the tool set: `--tools read,ls,grep,glob` makes a
read-only run, in either mode, because a tool not enabled is never declared to
the model.
