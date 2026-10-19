# Sessions

## Format

Append-only JSONL, one file per session:

```
~/.lab-harness/sessions/<project-slug>/<session-id>.jsonl
```

Line 1 is the header; every subsequent line is one entry.

```jsonc
{"type": "session", "version": 4, "id": "...", "cwd": "/path",
 "timestamp": "...", "model": "..."}
{"type": "context", "timestamp": "...", "restored": 0, "system_prompt": "...",
 "tools": [ /* declarations */ ], "model": { /* Model */ }, "config": { /* ... */ }}
{"type": "message", "timestamp": "...", "message": { /* ... */ }}
{"type": "usage", "usage": { /* ... */ }, "model": "..."}
{"type": "rewrite", "timestamp": "...", "messages": [ /* ... */ ], "model": null}
```

Every invocation — the first run, and each `--resume` after it — writes one
`context` entry before its messages, and so does every run that starts with a
different system prompt, tool set, model or reasoning effort than the last
entry records. That entry and the messages after it are a **segment**.

**Append-only, flushed per line.** A crashed or killed run leaves a readable
transcript up to its last completed message. That is the whole durability
story — the cheap, boring format survives what a clever one would not.

### The project slug

```
lab-harness-3f9a2c01
└─┬───┘ └─┬─────┘
  │       └ first 8 hex of sha256(absolute path)
  └ directory name, sanitized
```

The directory name alone is not enough: `~/work/api` and `~/fork/api` are
different projects and must not share a history. The hash disambiguates them;
the readable prefix keeps the directory browsable.

### Session ids

```
20261010T112109-8e2d2d
└─────┬───────┘ └─┬──┘
      │           └ random, so two sessions in one second cannot collide
      └ UTC timestamp, so sorting by name sorts by time
```

Sorting by name is the chronological order, so "most recent" never has to stat
every candidate.

## Entry types

| Type | Purpose |
|---|---|
| `session` | Header. One per file, first line. |
| `context` | What a segment's requests were sent with, and the config they ran under — recorded, never sent. Opens the segment. |
| `message` | One transcript message, stamped with when it was written |
| `usage` | Token usage for a request, written beside its assistant message. Repeats the message's own `usage`. |
| `rewrite` | What a `prepare_request` hook sent instead, for the request after it. See below. |
| `compaction` | **Reserved.** Not emitted yet; see below. |

**Unknown types are skipped, not fatal.** `compaction` is reserved so adding
compaction later needs no schema bump, and a reader that choked on an
unrecognized `type` would make every future addition a breaking change. The
count of skipped entries is surfaced so a resume can mention it.

`rewrite` is the exception that bumped the version to 3: a reader that skipped
it would rebuild those requests wrong rather than merely miss something.
Version 4 added `reasoning_effort` to the recorded model, for the same reason:
a reader that cannot place it cannot rebuild the request it changed.

## Trajectories

A transcript holds everything needed to rebuild every request the session made,
exactly as the provider received it, alongside the response it got:

```python
from lab_harness.session import read_transcript, recorded_requests

for step in recorded_requests(read_transcript(path)):
    step.request  # Context: system_prompt, messages, tools — what was sent
    step.response  # the AssistantMessage that came back, usage included
    step.timestamp  # when the response finished
    step.model  # the Model it went to, limits and all
    step.config  # the config snapshot the invocation ran with
```

That holds across resumes, an interrupted turn, a prompt, tool set, model,
reasoning effort or config that changed between invocations, one that changed between runs of the
same invocation, and a request a hook rewrote. Tests rebuild sessions with
each of those and compare against what the provider was actually handed.

### As the provider received it

A step is provider-neutral. The adapter that sent it turns it back into the
JSON body it posted — for a Chat Completions endpoint, OpenAI-format
`messages` and `tools`, the input the server's chat template was applied to
and the format training frameworks read:

```python
from lab_harness.config import from_snapshot
from lab_harness.llm.providers import build_provider

adapter = build_provider(step.model.provider, from_snapshot(step.config))
adapter.request_body(step.model, step.request)  # the body, as posted
adapter.reply_entries(step.response)  # the reply, as that dialect writes it
```

`from_snapshot` reads back the config the invocation recorded, so the body is
rebuilt under the provider settings it was sent with, not today's. A reply's
entries are what the next request carries for it, with one difference: they
always include the reasoning, even under a profile that sends none back.
[trajectories.py](../examples/trajectories.py) `--wire` writes both, one line
per request. Tests check the rebuilt body against what each adapter posted.

The token sequence itself — after the chat template, with the model's special
tokens — is not recoverable from a session. Only the inference server has it,
and only one serving open weights can return it.

### Changes mid-session

The system prompt, tools, model and reasoning effort can change between runs
without starting a new session — from Python by assigning to the agent's
state, and the effort from the terminal too, with `:effort`:

```python
agent.state.system_prompt = "Answer in French."
agent.state.tools = default_registry().select(["read", "grep"])
agent.state.model = resolve_model("claude-opus-5")  # same provider
agent.state.model = replace(agent.state.model, reasoning_effort="high")
await agent.prompt("...")  # recorded under the new context
```

Each run's `agent_start` event carries what it is sent with, and the recorder
writes a new `context` entry when that differs from the last one, before the
run's first message. A session that changes nothing writes nothing extra.

### Rewritten requests

A `prepare_request` hook may send something other than the history — a
compacted view, a redacted one — or send it to another model. The history
stays as it was: that is what resume continues from. So the file holds both.
The loop announces the rewrite as a `request_rewritten` event, the recorder
writes it as a `rewrite` entry, and `recorded_requests` rebuilds that request
from it: the hook's messages and model, the segment's prompt and tools.

A `rewrite` holds the whole view it sent, so a hook that rewrites every
request grows the file by a view per request. The CLI installs no hook.
Compaction, when it lands, will add its own entry to write a summary once
rather than with every request.

Every assistant message yields one step, errored and aborted ones included, so
filter on `step.response.stop_reason` for the ones worth keeping. An abort
*between* turns produces an aborted message that never reached the model; its
step is the request that would have been sent.

What a `context` entry holds:

| Field | Holds |
|---|---|
| `system_prompt` | The prompt in full, project instructions included |
| `tools` | Every tool declaration — name, description, JSON schema — in the order sent |
| `model` | The resolved `Model`: id, the provider that served it, limits, capability flags, reasoning effort |
| `config` | The whole resolved config as plain data, credentials replaced with `[redacted]` |
| `restored` | How many earlier messages this segment carried forward |
| `timestamp` | When the invocation started, or the run that changed the context |

**What is not recorded:**

- **The wire translation.** A step is the provider-neutral request; the bytes
  sent are what the adapter makes of it under the provider settings in
  `config` — which it does again on demand, [above](#as-the-provider-received-it).
- **Values read from the environment** — credentials and `*_BASE_URL`
  overrides. The snapshot holds the variables' names, not their values.
- **Streaming deltas.** The assembled message is kept instead.

**Credentials never reach the file.** Keys read from the environment never
enter `Config` at all, and `snapshot` replaces any a `config.local.toml`
supplied. A test fails if a new field named like a credential is left off the
redaction list.

**The transcript is as sensitive as the conversation.** It holds the full
system prompt, project instructions included, and tool results already put
whole files in it — share one as you would the session itself. Earlier
versions stored only a hash of the prompt; that hid little, given the rest of
the file, and made the trajectory impossible to rebuild.

### `restored`, and why the history is folded

The file is append-only, so when resume trims an interrupted turn the trimmed
messages stay on disk. `restored` records how many messages the next
invocation actually carried forward; reading a segment drops anything past
that count before appending the segment's own messages. Without it, the
second `--resume` after an interrupted turn would read the dropped turn back
into the middle of the history — an unanswered tool call that every provider
rejects.

`Transcript.history` is that folded history: the conversation as the last
invocation left it, each message with its timestamp. `Transcript.messages` is
the same without the timestamps, and is what resume continues from.

### Timestamps

Every `message` entry carries the UTC time it was written. For an assistant
message that is when its response finished streaming; for the user message,
when its run began. The results of one tool batch are written together once
the whole batch is done, so they share a moment — each one's `duration_ms`
says how long it ran.

### Older files

Version 3 files are version 4 files whose models ask for no reasoning effort —
none was sent then — and version 2 files are those without `rewrite` entries;
both read as such. Files written before the `context` entry (version 1) still read and
resume. Their messages count as history for anything appended later, but yield
no steps of their own: nothing recorded what they were sent with. Their
timestamps read as empty.

## Resume

```bash
lab-harness --resume              # list what is saved for this directory
lab-harness --resume <session-id> # continue one; a unique prefix works
lab-harness --no-save             # run without persisting anything
```

`--resume` alone lists and exits rather than continuing the most recent
session, which would be the wrong conversation whenever the last run was not
the one meant. In Python, `resume()` without a `session_id` does take the most
recent.

Resume reads the file, rebuilds the message list, and continues writing to the
**same** file, opening a new segment. An ambiguous id prefix is an error rather
than a guess: resuming the wrong conversation is worse than being asked again.

Resume warns when the system prompt differs from the one the last segment
recorded — usually because the project's `AGENTS.md` changed, or because the
run uses another model, which the prompt names. The session still
continues, under the new prompt; its segment records which one.

Interactive mode prints resume warnings under the "Resumed" line, and then
the conversation itself, the way each turn looked when it ran — so the session
continues where the eye left off rather than from a summary line. `-p` mode has
no terminal to print them in, so its warnings go to stderr as `warning` log
records — clear of the answer on stdout and of the `--json` stream.

### Interrupted turns are trimmed

A session killed mid-turn can end with an assistant message requesting tool
calls whose results were never written. Providers require every tool call in a
turn to have a matching result, so replaying that transcript fails on the first
request — the one case where resume would reliably break.

Resume drops the incomplete trailing turn and says so. A partially answered
batch is dropped whole, not truncated to its answered part: an assistant message
with *some* of its results attached is exactly as unsendable as one with none.

Trimming costs the user one re-ask. Not trimming costs them the session.

## Relationship to the event stream

Session entries reuse the serialization from the event protocol — the same
`to_dict()` that makes events JSON-safe makes messages persistable. The recorder
is an ordinary subscriber; the runtime does not know it exists, which is what
lets `--no-save` be a wiring decision rather than a branch in the loop.

It is *not* the same stream, though: events describe a run as it happens,
including streaming deltas and tool execution progress. The session records only
the durable result. **Deltas are never persisted.**

The agent emits the user's message on the stream for exactly this reason — a
transcript with no user messages in it cannot be resumed from. `restore()` is
deliberately silent in the other direction: replaying a resumed history through
the event stream would make the session re-record its own past. The interactive
terminal shows the history anyway, by drawing it directly through the renderer
— off the event stream, so the recorder never sees it.

## Viewing a session

| Command | Shows |
|---|---|
| `lab-harness --resume ID --no-save` | The conversation in the terminal, read-only |
| `lab-harness --export [ID]` | One self-contained HTML page beside the transcript: the conversation, every segment's context, usage and cost per request |
| `examples/trajectories.py [ID]` | Every request exactly as sent, one JSON line each; `--wire` as the endpoint received it |

The page is a view, not a record: it is written next to the `.jsonl`, never
read back, and safe to delete. See [cli.md](cli.md#sessions).

## Configuration

| Key | Default | Purpose |
|---|---|---|
| `session.home` | `~/.lab-harness` | Root for all harness state |
| `session.home_env` | `LAB_HARNESS_HOME` | Env var that overrides it |
| `session.directory_name` | `sessions` | Subdirectory holding projects |
| `session.enabled` | `true` | Write transcripts at all |
| `session.version` | `4` | Schema version written to each header |

Tests redirect `LAB_HARNESS_HOME` through an autouse fixture, so the suite can never
write to a real `~/.lab-harness`.
