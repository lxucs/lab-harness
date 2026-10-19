# The terminal

> **TODO:** a short recording of an interactive session.

The workflows people use most, from the repository root. Every flag, key and
config setting is in [docs/cli.md](../docs/cli.md).

## Interactive

```bash
export ANTHROPIC_API_KEY=sk-...
uv run lab-harness
```

| Key or command | Does |
|---|---|
| `Ctrl-C` | Cancel the turn in flight; the session carries on |
| `Ctrl-O` | Expand or collapse the model's thinking |
| `:status` | Tokens, cache hits and cost so far |
| `Ctrl-D` | Exit |

Mutating tools — `bash`, `write`, `edit` — ask before they run, and anything
but an explicit yes denies. `--yes` turns the asking off.

A `HARNESS.md` or `AGENTS.md` in the working directory is appended to the
system prompt, so a project's conventions do not have to be rediscovered every
run.

## One-shot

```bash
uv run lab-harness -p "summarize the config module"
uv run lab-harness -p "summarize the config module" > summary.txt   # the answer alone
```

**There is no approval gate in `-p` mode** — nobody is at the keyboard to
answer, so `bash`, `write` and `edit` run unprompted. To rule them out, give
the run fewer tools; a tool left out is never offered to the model:

```bash
uv run lab-harness -p "review the config module" --tools read,ls,grep,glob
```

Tool notices go to stderr and the answer to stdout. The exit status is `1`
when the final turn errored or was aborted, `0` otherwise.

## Events as JSONL, for scripts

```bash
uv run lab-harness -p "what files are here?" --json \
  | jq -c 'select(.type == "tool_execution_end") | {tool_name, is_error, duration_ms}'
```

One event per line, flushed as it happens. The event types and their fields:
[docs/events.md](../docs/events.md).

## Sessions

```bash
uv run lab-harness --resume                 # what is saved for this directory
uv run lab-harness --resume <session-id>    # continue one; a unique prefix works
uv run lab-harness --no-save                # write nothing
```

Continuing one prints the saved conversation — questions, answers, tool calls —
before the first prompt, so the session picks up where the eye left off.

### Viewing a session

```bash
uv run lab-harness --resume <session-id> --no-save   # read it in the terminal, then Ctrl-D
uv run lab-harness --export <session-id>             # or as a web page; prints its path
open "$(uv run lab-harness --export)"                # the latest session, in a browser (macOS)
```

`--resume` with `--no-save` is a read-only view: the whole conversation, Ctrl-O
to expand every thought, and nothing appended to the file. `--export` writes
one self-contained HTML page beside the transcript, with more than the terminal
shows: each invocation's system prompt, tools and config, and tokens and cost
per request. The page searches the conversation (`/`), folds thinking (`t`) and
tool calls (`o`), lists the turns down the side, and follows the system's light
or dark theme. It loads nothing from the network, so it can be mailed or
attached as it is.

Transcripts live under `~/.lab-harness/sessions/`, one directory per project.
Each records what every request was sent with;
[trajectories.py](trajectories.py) exports them as JSONL.

## Other models

```bash
uv run lab-harness --model claude-opus-5
DEEPSEEK_API_KEY=sk-... uv run lab-harness --model deepseek-flash   # routed by config
ZHIPUAI_API_KEY=... uv run lab-harness --model glm-5.3
OPENAI_API_KEY=sk-... uv run lab-harness --model gpt-6-luna
```

Adding a vendor or a model is a few lines of config:
[docs/configuration.md](../docs/configuration.md#provider-profiles).

## The system prompt

```bash
uv run lab-harness --system-prompt "Answer in one sentence."
uv run lab-harness -p "summarize the config module" --system-prompt-file prompts/terse.md
```

Either replaces the opening section of the default, which ships in
[`lab_harness/prompts/prompts.toml`](../lab_harness/prompts/prompts.toml), for
one run. The rest is still sent — how the harness works, the working
directory, platform and model, and a project's `AGENTS.md`. To change any
section for good, or leave one out, set it under `[prompts]` in config:
[docs/configuration.md](../docs/configuration.md#prompts).

## From another directory

Install the command once, from the repository root, and move the settings and
keys every project should get to `~/.lab-harness/`, which is read wherever the
harness runs:

```bash
uv tool install --editable .        # lab-harness on PATH; edits here apply at once
mkdir -p ~/.lab-harness
cp .lab-harness/config.local.toml ~/.lab-harness/config.local.toml
```

Then start it in the directory to work on — no `--cwd`, no `uv run`:

```bash
cd /path/to/project
lab-harness                         # its tools, AGENTS.md and sessions; your config
```

A `.lab-harness/` there still applies on top of yours. Without installing,
`uv run lab-harness --cwd /path/to/project` from here does the same.

## Debugging

```bash
uv run lab-harness --log-level debug -p "..."             # trace a run on stderr
uv run lab-harness --log-level debug 2> lab-harness.log   # interactive, trace kept apart
```

The debug trace shows each request, response and tool call by size and id,
never by content.
