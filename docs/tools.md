# Tools

## The contract

A tool returns `ToolResult(is_error=True)` or raises; the loop converts a raise
into an error result. **A tool can never kill the loop.**

Error messages are written *for the model*, not for a log. The model's next
action is chosen from that text, so a good message says what went wrong and
what to do instead:

```
`old` text appears 3 times in config.py. Add surrounding context to make it
unique, or set replace_all.
```

A `ToolResult` carries two things. `content` is what the model reads.
`details` is structured data it never sees, for whoever renders the call —
`edit`'s diff, `bash`'s exit code — and reaches renderers on the
`tool_execution_end` event. Keep it to JSON primitives, like every event, and
paths in it relative to `cwd`, like output: it reaches `--json` logs too. See
[events.md](events.md#tool_execution_end-carries-the-results-details).

The contract lives in `lab_harness/tools/base.py` rather than in `agent/`, because
dependencies point downward: `agent/` consumes tools, so `tools/` must never
import from it. `agent/types.py` re-exports the names for callers.

## Defining a tool

Parameters are a Pydantic model — one object gives JSON Schema, validation, and
type hints. The decorator reads the model off the first argument's annotation,
so the schema and the implementation cannot drift apart.

```python
from pydantic import BaseModel, Field
from lab_harness.tools.base import ToolContext, ToolResult, fail, ok, tool


class ReadParams(BaseModel):
    path: str = Field(description="Path to the file, relative or absolute")
    offset: int | None = Field(None, ge=1, description="1-indexed start line")
    limit: int | None = Field(None, ge=1, description="Maximum lines to read")


@tool(name="read", description="Read the contents of a file.")
async def read(p: ReadParams, ctx: ToolContext) -> ToolResult:
    ...
    return ok(rendered, path=str(path), lines=42)
```

Adding a tool is one file plus one registry entry.

**The description is the only place to tell the model how to use a tool.** The
system prompt names no tools, so it stays true whichever ones `--tools`
enables; advice such as "prefer this to running grep through a shell" goes in
the description, which is only sent when the tool is.

**Invalid arguments never reach your function.** `FunctionTool.execute`
validates first and returns a `fail()` naming the offending fields, so the model
can correct itself. A bad signature — a first argument that is not a Pydantic
model, or the wrong arity — raises at *definition* time, not when the model
first calls it.

### `ToolContext`

| Field | Purpose |
|---|---|
| `cwd` | Working directory for this call — **always resolve paths against this** |
| `call_id` | The tool call's id, for correlating updates |
| `config` | Resolved `Config`; read limits from here, never hardcode them |
| `signal` | Cancellation token; long-running tools should check it |
| `on_update` | Emit a partial result (e.g. bash output as it arrives) |

Everything a tool needs arrives through this. Tools never reach for globals or
the process working directory, which is what makes them testable against
`tmp_path` and safe when two agents have different roots.

### Execution mode

Tools run in parallel by default. Declare `execution_mode="sequential"` to opt
out — `write` and `edit` do, since concurrent mutation of the same file is not
something to leave to chance. One sequential tool in a batch serializes that
whole batch.

## The registry

```python
from lab_harness.tools import ToolRegistry, default_registry

registry = default_registry()  # every built-in
tools = registry.select(["read", "ls"])  # narrow, in order
```

There is deliberately **no module-level default instance**. A global registry is
what makes two tests with different tool sets interfere. Construct one per run
and pass it down.

Duplicate names raise on registration, and `select()` raises on an unknown name
or one given twice — a typo in configured tool names should fail at wiring
time, not silently disable a tool and leave the model unable to do its job.

The CLI selects with `tools.enabled` from config, or `--tools read,ls` for one
run, and the same list applies in interactive and one-shot mode. **Interactive
mode can change it mid-session**: `/tools read,skill`, or `/tools none`. A tool
that is not selected is never declared to the model, which makes selection the
way to keep `bash` out of a `-p` run — approval cannot, since `-p` has none. An
unknown name is a startup error that names where it came from, and one given
mid-session is reported without changing anything.

## bash

Three things make it more than "run a subprocess":

**No stdin.** The command gets `/dev/null`, never the terminal's — a command
that reads (`read`, `cat`, an editor, `ssh` asking for a passphrase) would
otherwise take the keys meant for the next prompt, or wait on a keyboard nobody
is watching. Piping *within* the command is unaffected.

**Streaming.** Output is forwarded through `ctx.on_update` as it arrives, so a
long build is visible while it runs.

**Cancellation.** `ctx.signal` stops the process *group*, not just the shell.
Killing only the shell orphans whatever it started — `sh -c "sleep 100"` would
leave the sleep running. On cancellation the group gets SIGTERM, then SIGKILL
after a grace period.

**Bounded output.** A command printing a gigabyte must not reach the context
window. The head and tail are kept and the middle elided, because both ends
usually matter and the middle rarely does. The tail is a ring buffer, so memory
stays bounded no matter the volume.

Exit status is reported to the model, and a non-zero code makes the result an
error so the model knows the command failed.

## grep and glob

Pure Python, not a shell-out to ripgrep or find. The dependency-free version is
fast enough at this scale, behaves identically on every platform, and — the
deciding reason — runs under the offline suite with no binary installed.

Both skip `tools.search.ignored_directories`, pruned *during* the walk rather
than filtered after, so a large `node_modules` is never descended into.

`glob` sorts **newest first**: in a repo someone is actively working in, recency
is a better relevance signal than alphabetical order.

## Approval

Mutating tools are gated by a `before_tool_call` hook, not by loop logic — which
is why one-shot and interactive modes differ without the loop branching.

| Mode | Gate |
|---|---|
| `lab-harness -p "…"` | **None.** Nobody is at the keyboard to answer — narrow `--tools` instead. |
| `lab-harness` (interactive) | Prompts for `tools.approval.required` |
| `lab-harness --yes` | Disabled explicitly |

**It fails closed.** Only `y`/`yes`, `a`/`always` proceed. A bare Enter, an
unrecognized answer, Ctrl-C, and Ctrl-D all deny — pressing Enter is a reflex,
not consent. A hook that *crashes* also denies, because the loop converts a
raising hook into a blocked call.

`always` is remembered for the session and never persisted: a decision made in
one run must not silently authorize the next.

## Built-in tools

| Tool | Mode | Notes |
|---|---|---|
| `read` | parallel | 1-indexed numbered lines; offset/limit; clips long lines; refuses oversized and binary files |
| `write` | **sequential** | Creates parent directories; overwrites; reports a diff against what was there. **Approval-gated.** |
| `edit` | **sequential** | Exact string replacement; refuses ambiguous matches unless `replace_all`; reports a diff. **Approval-gated.** |
| `ls` | parallel | Directories first, marked with `/`; hides dotfiles unless `all` |
| `bash` | **sequential** | Combined stdout+stderr, streamed; timeout; output capped. **Approval-gated.** |
| `grep` | parallel | Regex over file contents; skips binary files and ignored directories |
| `glob` | parallel | Files by path pattern, newest first |
| `web_search` | parallel | Titles, URLs and snippets via a pluggable backend |
| `web_fetch` | parallel | One page as markdown-ish text; follows redirects |
| `skill` | parallel | Loads a skill's instructions by name; see [Skills](#skills) |

### Limits come from config

Every threshold is a config key, never a literal:

| Key | Default | Controls |
|---|---|---|
| `tools.read.default_limit` | 2000 | Lines returned when none requested |
| `tools.read.max_line_length` | 2000 | Where a long line is clipped |
| `tools.read.max_bytes` | 256000 | Largest file `read` will open |
| `tools.list.max_entries` | 1000 | Entries per `ls` |
| `tools.diff.context_lines` | 3 | Unchanged lines around each change in an edit's or write's diff |
| `tools.bash.timeout_seconds` | 120 | Default command timeout |
| `tools.bash.max_timeout_seconds` | 600 | Ceiling on a requested timeout |
| `tools.bash.max_output_bytes` | 30000 | Output kept before eliding the middle |
| `tools.search.max_results` | 200 | Matches per `grep` |
| `tools.search.max_files` | 500 | Paths per `glob` |
| `tools.search.ignored_directories` | `.git`, `node_modules`, … | Never descended into |
| `tools.web.backend` | `tavily` | Which search backend runs |
| `tools.web.max_results` | 5 | Results per search |
| `tools.web.timeout_seconds` | 30.0 | Wall-clock limit per search or fetch |
| `tools.web.max_bytes` | 2000000 | Largest response `web_fetch` downloads |
| `tools.web.max_content_chars` | 50000 | Extracted text handed to the model |
| `tools.web.user_agent` | `lab-harness/0.1 …` | Sent on every outbound request |
| `skills.enabled` | `true` | Whether skills are looked for at all |
| `skills.selected` | (empty) | Which of those found apply; empty is every one |
| `skills.directory_name` | `skills` | Subdirectory holding them, in `.lab-harness` and in home |
| `skills.max_bytes` | 32000 | Where a long `SKILL.md` is cut |

`read`'s truncation note names the offset to resume from, so a follow-up call is
mechanical rather than guesswork.

### Paths in output

Tool output renders paths **relative to `cwd`** when they are inside it, and
so do the paths in `details` and in a diff's header. Absolute paths leak the
machine's layout into a transcript that may well be committed, and into every
`--json` log — see the hygiene rules in [contributing.md](contributing.md).

### Diffs

`edit` and `write` report their change as a unified diff in
`details["diff"]` — the standard format rather than an invented one, so a
consumer in any language can parse it. A write that creates a file diffs
against `/dev/null`; one that overwrites a file that was not UTF-8 text writes
it anyway and reports no diff. The model is told only what it always was
("Replaced 1 occurrence(s) in a.py"): a diff it already knows the content of
would cost tokens and tell it nothing.

## Skills

A skill is instructions for one kind of task, kept in a directory rather than
in the prompt:

```
.lab-harness/skills/commit/SKILL.md      this project
~/.lab-harness/skills/commit/SKILL.md    every project
```

```markdown
---
name: commit
description: Write a commit message for staged changes
allowed-tools: Bash(git:*), Read
---
Steps to follow, and whatever examples the task needs.
```

The frontmatter is YAML. `description` is required — it is the whole basis on
which one is picked — and `name` defaults to the directory's. Unknown keys are
ignored, so a file written for another harness still loads.

**Only the name and the description ride in every request**, as a list in the
system prompt; the instructions come back when the model calls `skill` with the
name. That is the point of the split: a directory of skills costs a list, not
the text of all of them. The other files a skill's directory holds are named in
the result, so the model can read the one it needs instead of having a long
reference inlined.

The list is left out entirely when `skill` is not among the tools a run gets —
`tools.enabled`, or `--tools` — since it would offer instructions the model has
no way to ask for.

`allowed-tools` names tools the skill may use without asking. An entry may
carry an argument pattern, as in `Bash(git:*)`: what is before the parenthesis
is the tool. Names are matched without case or underscores, so `WebFetch` finds
`web_fetch`. They are granted for the rest of the session when the skill is
loaded, the way answering "always" is, and the terminal says which. One naming
a tool this run does not have is dropped with a warning — a skill written for
another harness names tools that are not here — and turning
`tools.approval.allow_remember` off suppresses the grant as well.

A `SKILL.md` that exists but cannot be used — no frontmatter, invalid YAML, no
description — is a startup error rather than a silent skip, for the reason a
malformed config file is.

Each is also a command in the terminal: `/commit` sends these instructions as
the next turn, and `/skills` lists them — see [cli.md](cli.md#skills).

**Which apply is chosen at three levels**, all reading the same setting:
`skills.selected` in config, `--skills` for one run, and `/skills` in the
terminal mid-session. Nothing is filtered anywhere else — the `skill` tool, the
list in the system prompt, and the terminal's menu all ask the same question,
which is why a change in one place is a change in all of them.

## Web tools

`web_search` finds pages; `web_fetch` reads one.

**They are two tools because search returns snippets, not pages.** A result is a
title, a URL and a sentence or two — enough to learn *where* an answer is, not
enough to answer from. The model is expected to search, pick a result, then
fetch it. Treating fetch as an afterthought would make search half a feature.

### Backends

`web_search` runs through a `SearchBackend`, chosen by `tools.web.backend`:

| Backend | Key | Returns |
|---|---|---|
| `tavily` *(default)* | `TAVILY_API_KEY` or `tools.web.tavily.api_key` | Title, URL, query-relevant snippet, relevance score |
| `ddgs` | none | Title, URL, snippet |

Adding a third is a class with a `search()` method plus one line in `BACKENDS`.

Tavily's snippets are extracted against the query rather than being the page's
meta description, which is the practical difference from a keyless backend. Set
`include_raw_content` to get full page text per result — off by default, since
raw content for five results is usually far more than the answer needs and
`web_fetch` can get it for the one result that matters.

> **The keyless `ddgs` backend is unproven.** It is implemented and
> unit-tested, and a failure surfaces as a readable tool error rather than an
> exception — but it scrapes search front-ends that are not reachable from
> every network, so it has not been confirmed against a live engine. Tavily is
> the default partly for this reason.

### Keys

A key is only needed for Tavily, and only for search — `web_fetch` never needs
one. Resolution is environment first, then `config.local.toml`, so a key left
in a file can be overridden rather than being the one that wins. A missing key
produces an error naming both places to set one *and* pointing at the keyless
backend, because the useful next step when you have no key is to stop needing
one.

The `ddgs` package is an optional dependency — most users should not have to
install a scraping stack they never call:

```bash
uv sync --extra ddgs
```

Selecting the backend without the package installed fails with that command in
the message.

### Fetching

`web_fetch` follows redirects, caps the download at `max_bytes`, and converts
HTML to markdown-ish text with a stdlib parser (`tools/html_text.py` — no
converter dependency, for the same reason `grep` and `glob` are hand-rolled).
Non-HTML content types pass through unparsed, so fetching a JSON API returns
JSON rather than JSON run through an HTML parser.

Extraction drops scripts, styles, and navigation chrome. Dropping navigation
matters more than it looks: output is capped at `max_content_chars`, so a
sidebar of two hundred links does not merely waste tokens, it pushes the
article off the end.

Only absolute `http(s)` URLs are accepted. `read` is the tool for local files,
and `web_fetch` must not quietly become a second one.

### Not planned: the provider-native path

Anthropic can run `web_search` server-side, doing the whole search-read-cite
loop inside one API request with no key of ours. **We are not building that.**
`tools.web.backend` selects a *local* backend, and that is the whole story: a
tool is executed locally, full stop, with no tool-kind enum and no second
execution path to reason about. A provider-specific capability that only one
of three adapters could offer is not worth that split.

Two pieces of groundwork survive because they are honest descriptions of
things rather than scaffolding for this: `Model.supports_server_tools` records
a real model capability, and `ProviderBlock` round-trips provider-native
content the core does not model — which it does today for any block an adapter
does not recognize.

## Testing

Tools are tested against `tmp_path`. Web tools are tested through
`httpx.MockTransport`, which runs the real request-building and
response-parsing code against canned bytes; backend *selection* is tested by
registering a fake in the `BACKENDS` dict under a config name, so the real
lookup path runs. The suite is offline — see [`../AGENTS.md`](../AGENTS.md).
