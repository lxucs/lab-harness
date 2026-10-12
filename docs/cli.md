# CLI

> **Status:** All modes are implemented — one-shot, JSONL, interactive, and
> resume.

## Modes

```bash
harness                      # interactive                    ✅ implemented
harness -p "fix the lints"   # one-shot, plain text           ✅ implemented
harness -p "..." --json      # one-shot, JSONL event stream   ✅ implemented
harness --resume             # continue the most recent session   ✅ implemented
```

All of these are **subscribers to the same event stream**. The interactive
terminal and the JSONL writer differ only in what they do with events, not in
what the runtime does.

## Flags

| Flag | Purpose |
|---|---|
| `-p`, `--prompt` | Run one prompt and exit. Omit it to start the interactive terminal. |
| `--json` | Emit the raw event stream as JSONL instead of rendered text. |
| `--model` | Model id. Defaults to config, then `HARNESS_MODEL`. |
| `--cwd` | Working directory for tools *and* config discovery. Defaults to the current directory. |
| `--yes` | Skip tool approval prompts in interactive mode. |
| `--resume [ID]` | Continue a session. With no id, the most recent one here. |
| `--sessions` | List saved sessions for this directory and exit. |
| `--no-save` | Do not write a transcript for this run. |

**Exit status is meaningful.** A run whose final turn ended in `error` or
`aborted` exits `1`; a clean run exits `0`; `Ctrl-C` exits `130`. A script
piping this needs to tell a refusal from an outage.

In plain-text mode, assistant text goes to **stdout** while tool-call notices
and errors go to **stderr**, so `harness -p "…" > answer.txt` captures the
answer alone.

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

The agent runs one turn sequence at a time. Typing while it works is buffered by
the terminal and submitted when the run finishes — the runtime itself refuses a
second concurrent `prompt()` rather than coordinating with the one in flight.

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

Format, trimming of interrupted turns, and the reserved `compaction` entry:
[sessions.md](sessions.md).

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
