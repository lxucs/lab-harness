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

Either replaces the default, which ships in
[`lab_harness/prompts.toml`](../lab_harness/prompts.toml), for one run; a
project's `AGENTS.md` is still appended. To change it for good, set
`[prompts] system` in config: [docs/configuration.md](../docs/configuration.md#prompts).

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
