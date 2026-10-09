# Sessions

> **Status:** Not yet implemented. This page describes the designed format and
> is updated as `session/` lands.

## Format

Append-only JSONL, one file per session:

```
~/.harness/sessions/<project-slug>/<session-id>.jsonl
```

Line 1 is the header; every subsequent line is one entry.

```jsonc
{"type": "session", "version": 1, "id": "...", "cwd": "/path",
 "timestamp": "...", "model": "...", "system_prompt_hash": "..."}
{"type": "message", "id": "...", "message": { /* ... */ }}
{"type": "usage", "id": "...", "usage": { /* ... */ }}
```

Append-only matters: a crashed run leaves a readable transcript, and writes are
cheap enough to do synchronously after each message.

## Entry types

| Type | Purpose |
|---|---|
| `session` | Header. One per file, first line. |
| `message` | One transcript message |
| `usage` | Token usage and cost for a request |
| `compaction` | **Reserved.** Not emitted in v1; see below. |

`compaction` is reserved now so that adding compaction later does not require a
schema version bump. A v1 reader encountering one should skip it rather than
fail — forward-compatible skipping of unknown `type` values is part of the
format.

## Resume

```bash
harness --resume              # most recent session for this cwd
harness --resume <session-id> # a specific session
```

Resume replays the file to rebuild the transcript, then continues. Entries with
unrecognized types are skipped.

## Relationship to the event stream

Session entries reuse the serialization from the event protocol — the same
`to_dict()` that makes events JSON-safe makes messages persistable. This is why
the event types are built first in the build order.

It is *not* the same stream, though: events describe a run as it happens
(including streaming deltas and tool execution progress); the session records
the durable result (messages and usage). Deltas are never persisted.
