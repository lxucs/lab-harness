# CLI

> **Status:** Not yet implemented. This page describes the designed interface
> and is updated as `cli/` lands.

## Modes

```bash
harness                      # interactive
harness -p "fix the lints"   # one-shot, plain text
harness -p "..." --json      # one-shot, JSONL event stream
harness --resume             # continue the most recent session for this cwd
```

All three are **subscribers to the same event stream**. The interactive terminal
and the JSONL writer differ only in what they do with events, not in what the
runtime does.

## Interactive

prompt_toolkit handles input: history, multi-line entry, and key bindings. rich
handles output: streaming text, markdown, and syntax-highlighted diffs.

Output is **inline** — it lands in real terminal scrollback rather than being
trapped inside a full-screen app, so scrolling back through a long session works
the way it does with any other command.

```
> read the config and summarize

● read(harness/config.py)
  └ 42 lines

The config defines three…

> ▊
```

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

```bash
$ harness -p "list the python files" --json
{"type": "agent_start"}
{"type": "turn_start"}
{"type": "message_update", "block_index": 0, "delta": "I'll"}
{"type": "tool_execution_start", "tool_name": "glob", "args": {"pattern": "**/*.py"}}
...
```

Pipe it into `jq`, or use it as the assertion surface for end-to-end tests —
checking a structured event stream beats scraping rendered terminal output.

## Tool approval

Mutating tools (`bash`, `write`, `edit`) prompt before executing:

```
● bash(rm -rf build/)
  Allow? [y]es / [n]o / [a]lways for bash
```

This is a `before_tool_call` hook, not loop logic. `always` is remembered for
the session, not persisted.

Approval is skipped in `-p` mode, which means one-shot runs execute mutating
tools without asking — use it deliberately.
