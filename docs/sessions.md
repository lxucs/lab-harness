# Sessions

## Format

Append-only JSONL, one file per session:

```
~/.harness/sessions/<project-slug>/<session-id>.jsonl
```

Line 1 is the header; every subsequent line is one entry.

```jsonc
{"type": "session", "version": 2, "id": "...", "cwd": "/path",
 "timestamp": "...", "model": "..."}
{"type": "context", "timestamp": "...", "restored": 0, "system_prompt": "...",
 "tools": [ /* declarations */ ], "model": { /* Model */ }, "config": { /* ... */ }}
{"type": "message", "timestamp": "...", "message": { /* ... */ }}
{"type": "usage", "usage": { /* ... */ }, "model": "..."}
```

Every invocation — the first run, and each `--resume` after it — writes one
`context` entry before its messages. That entry and the messages after it are a
**segment**.

**Append-only, flushed per line.** A crashed or killed run leaves a readable
transcript up to its last completed message. That is the whole durability
story — the cheap, boring format survives what a clever one would not.

### The project slug

```
harness-3f9a2c01
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
| `context` | What one invocation's requests were sent with. Opens its segment. |
| `message` | One transcript message, stamped with when it was written |
| `usage` | Token usage for a request, written beside its assistant message. Repeats the message's own `usage`. |
| `compaction` | **Reserved.** Not emitted yet; see below. |

**Unknown types are skipped, not fatal.** `compaction` is reserved so adding
compaction later needs no schema bump, and a reader that choked on an
unrecognized `type` would make every future addition a breaking change. The
count of skipped entries is surfaced so a resume can mention it.

## Trajectories

A transcript holds everything needed to rebuild every request the session made,
exactly as the provider received it, alongside the response it got:

```python
from harness.session import read_transcript, recorded_requests

for step in recorded_requests(read_transcript(path)):
    step.request  # Context: system_prompt, messages, tools — what was sent
    step.response  # the AssistantMessage that came back, usage included
    step.timestamp  # when the response finished
    step.model  # the Model, limits and all
    step.config  # the config snapshot the invocation ran with
```

That holds across resumes, an interrupted turn, and a prompt, tool set, model
or config that changed between invocations — each segment records its own. A
test rebuilds a session with all of those and compares against what the
provider was actually handed.

Every assistant message yields one step, errored and aborted ones included, so
filter on `step.response.stop_reason` for the ones worth keeping. An abort
*between* turns produces an aborted message that never reached the model; its
step is the request that would have been sent.

What a `context` entry holds:

| Field | Holds |
|---|---|
| `system_prompt` | The prompt in full, project instructions included |
| `tools` | Every tool declaration — name, description, JSON schema — in the order sent |
| `model` | The resolved `Model`: id, the provider that served it, limits, capability flags |
| `config` | The whole resolved config as plain data, credentials replaced with `[redacted]` |
| `restored` | How many earlier messages this invocation carried forward |
| `timestamp` | When the invocation started |

**What is not recorded:**

- **The wire translation.** A step is the provider-neutral request; the bytes
  sent are what the adapter makes of it under the provider settings in
  `config`.
- **Anything a `prepare_request` hook does.** It rewrites the request without
  touching the transcript. The CLI installs none, but a step would not reflect
  one that did.
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

### Version 1 files

Files written before the `context` entry still read and resume. Their messages
count as history for anything appended later, but yield no steps of their own:
nothing recorded what they were sent with. Their timestamps read as empty.

## Resume

```bash
harness --resume              # most recent session for this cwd
harness --resume <session-id> # a specific session; a unique prefix works
harness --sessions            # list what is saved for this directory
harness --no-save             # run without persisting anything
```

Resume reads the file, rebuilds the message list, and continues writing to the
**same** file, opening a new segment. An ambiguous id prefix is an error rather
than a guess: resuming the wrong conversation is worse than being asked again.

Resume warns when the system prompt differs from the one the last segment
recorded — usually because the project's `AGENTS.md` changed. The session still
continues, under the new prompt; its segment records which one.

Interactive mode prints resume warnings under the "Resumed" line. `-p` mode has
no terminal to print them in, so they go to stderr as `warning` log records —
clear of the answer on stdout and of the `--json` stream.

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
the event stream would make the session re-record its own past.

## Configuration

| Key | Default | Purpose |
|---|---|---|
| `session.home` | `~/.harness` | Root for all harness state |
| `session.home_env` | `HARNESS_HOME` | Env var that overrides it |
| `session.directory_name` | `sessions` | Subdirectory holding projects |
| `session.enabled` | `true` | Write transcripts at all |
| `session.version` | `2` | Schema version written to each header |

Tests redirect `HARNESS_HOME` through an autouse fixture, so the suite can never
write to a real `~/.harness`.
