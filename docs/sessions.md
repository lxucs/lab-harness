# Sessions

## Format

Append-only JSONL, one file per session:

```
~/.harness/sessions/<project-slug>/<session-id>.jsonl
```

Line 1 is the header; every subsequent line is one entry.

```jsonc
{"type": "session", "version": 1, "id": "...", "cwd": "/path",
 "timestamp": "...", "model": "...", "system_prompt_hash": "..."}
{"type": "message", "message": { /* ... */ }}
{"type": "usage", "usage": { /* ... */ }, "model": "..."}
```

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
| `message` | One transcript message |
| `usage` | Token usage for a request, written beside its assistant message |
| `compaction` | **Reserved.** Not emitted in v1; see below. |

**Unknown types are skipped, not fatal.** `compaction` is reserved so adding
compaction later needs no schema bump, and a reader that choked on an
unrecognized `type` would make every future addition a breaking change. The
count of skipped entries is surfaced so a resume can mention it.

## The system prompt is hashed, not stored

The header records `system_prompt_hash`, not the prompt. A system prompt can
carry project context that does not belong in a file someone might share or
paste into a bug report. The hash is still enough to warn on resume that the
prompt has changed since the session was recorded.

## Resume

```bash
harness --resume              # most recent session for this cwd
harness --resume <session-id> # a specific session; a unique prefix works
harness --sessions            # list what is saved for this directory
harness --no-save             # run without persisting anything
```

Resume reads the file, rebuilds the message list, and continues writing to the
**same** file. An ambiguous id prefix is an error rather than a guess: resuming
the wrong conversation is worse than being asked again.

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
| `session.version` | `1` | Schema version written to each header |

Tests redirect `HARNESS_HOME` through an autouse fixture, so the suite can never
write to a real `~/.harness`.
