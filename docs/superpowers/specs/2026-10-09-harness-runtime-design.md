# Harness Runtime — Design

**Date:** 2026-10-09
**Status:** Approved, pre-implementation

## Goal

A lightweight agent harness whose runtime is Python. It runs a basic local
agentic loop with file read/write, shell, search, and web tools. It is modeled
on [pi](https://github.com/earendil-works/pi) — the same layering, trimmed — and
optimized for being read and extended rather than for feature coverage.

Non-goals for v1: a provider catalog, MCP, sub-agents, plan mode, RPC control
channels, extensions-as-plugins, sandboxing, and context compaction (deferred;
see §7).

## 1. Architecture

Three layers, mirroring pi's package split, as subpackages of one installable
`harness` package. Dependencies point strictly downward.

```
harness/
  llm/        # ≈ pi-ai          — provider-neutral LLM access
  agent/      # ≈ pi-agent-core  — the loop + stateful Agent
  tools/      #                  — built-in tools + registry
  session/    #                  — JSONL transcript store
  cli/        #                  — terminal + json sink (subscribers)
  config.py
  prompts/
```

- `llm/` knows messages, models, and tool *declarations*. Not tool execution,
  not the filesystem, not the terminal.
- `agent/` knows the loop, tools-as-executables, and events. Not the filesystem
  layout, not any UI.
- `cli/` subscribes to the agent's event stream. The runtime never imports from
  `cli/`; deleting `cli/` leaves the runtime working.

`context/` is deferred along with compaction (§7), but the `prepare_request`
hook that would host it exists from day one.

### Deviation from pi: prompt and tools are not in the transcript

pi stores the system prompt and tool declarations *inside* the transcript as
system messages, replaying them to derive current state — `declareToolChanges`,
`getCurrentTools`, section diffing, several hundred lines. We instead pass
`Context(system_prompt, messages, tools)` to the provider directly.

We lose mid-session tool-change provenance. We gain a much smaller core. For a
harness whose tool set is fixed at startup, this is the right trade.

## 2. `llm/` — provider layer

```
llm/
  types.py            # Message union, content blocks, Usage, StopReason, Model
  events.py           # StreamEvent protocol
  provider.py         # Provider protocol + StreamOptions
  providers/
    anthropic.py
    openai.py
  registry.py         # model resolution from config/env
  fake.py             # scripted provider for tests (≈ pi's faux.ts)
```

Frozen dataclasses of primitives throughout.

**Content blocks:** `TextBlock`, `ThinkingBlock`, `ImageBlock`, `ToolCallBlock`,
plus `ProviderBlock` — an opaque passthrough that round-trips provider-native
content. `ProviderBlock` is how Anthropic's server-side `web_search_tool_use`
and `web_search_tool_result` blocks survive a turn without polluting core types.

**Messages:** `SystemMessage`, `UserMessage`, `AssistantMessage`,
`ToolResultMessage`. `AssistantMessage` carries `usage`, `stop_reason`,
`error_message`, and the model/provider that produced it.

**Stream protocol**, mirroring pi:

```
start → {text,thinking,toolcall}_{start,delta,end}* → done | error
```

**The provider contract**, lifted from pi verbatim because it is why the loop
has so few try/excepts:

> A provider **must not raise** for request, model, or runtime failures.
> Failures terminate the stream with an `error` event carrying an
> `AssistantMessage` with `stop_reason="error"` and `error_message`.

```python
class Provider(Protocol):
    async def stream(
        self, model: Model, ctx: Context, opts: StreamOptions
    ) -> AsyncIterator[StreamEvent]: ...
```

Two adapters ship: **Anthropic Messages** (native thinking, `cache_control`,
server tools) and **OpenAI Chat Completions** (also reaches DeepSeek, Groq,
OpenRouter, vLLM, and Ollama via `base_url`). Two is the minimum that keeps
provider-specific shapes out of the core types.

## 3. `agent/` — the loop

```
agent/
  types.py   # AgentEvent, AgentTool, AgentContext, hook protocols
  loop.py    # run_agent_loop() — pure async function, no state
  agent.py   # Agent — owns transcript, subscribers, abort
```

Two tiers, as in pi: `loop.py` is a function you hand a context and a config;
`agent.py` is the stateful object a frontend talks to.

**Events:** `agent_start/end`, `turn_start/end`, `message_start/update/end`,
`tool_execution_start/update/end`.

`message_update` carries `delta: str` and `block_index: int` rather than a full
partial snapshot — small and cheap to serialize. Consumers needing the whole
in-flight message read `agent.state.streaming_message`.

**Hooks**, all optional async callables, at the seams:

| Hook | Signature | Used by |
|---|---|---|
| `before_tool_call` | `(ctx) -> BeforeResult \| None` | approval gate |
| `after_tool_call` | `(ctx) -> AfterResult \| None` | truncation, redaction |
| `prepare_request` | `(req) -> RequestUpdate \| None` | compaction (later) |

`finish_turn` was in this design but was not built — nothing needed it. It
belongs here when an auto-continue feature does.

**Simplification over pi:** no message queue at all. pi has separate steering
and follow-up queues with `all` / `one-at-a-time` drain modes. Here a single run
is active at a time and `prompt()` refuses while one is in flight; a frontend
buffers mid-run typing itself.

*Revised during implementation.* The first cut did have one queue drained at turn
boundaries. Review found it broke transcript fidelity — an injected message
survived a single request and then disappeared from the provider's view while
staying in `state.messages` — and that a message queued on the final turn was
never delivered. Removing it was cheaper than correcting it, and makes the
failure impossible rather than fixed.

**Parallel tool execution:** prepare calls sequentially, so approval prompts
stay ordered, then `asyncio.gather` the allowed ones. `tool_execution_end` fires
in completion order; tool-result *messages* append in assistant source order,
because providers require that ordering. Tools opt out with
`execution_mode = "sequential"` — `write` and `edit` do.

## 4. `tools/`

```
tools/
  base.py      # Tool, ToolResult, ToolContext, ToolKind
  registry.py  # register / get / declarations_for(provider)
  fs.py        # read, write, edit, ls
  shell.py     # bash
  search.py    # grep, glob
  web.py       # web_search, web_fetch (local backends)
  native.py    # provider-native server-tool declarations
```

Pydantic models for parameters — one object yields JSON Schema, validation, and
type hints:

```python
class ReadParams(BaseModel):
    path: str = Field(description="Path to the file, relative or absolute")
    offset: int | None = Field(None, description="1-indexed start line")
    limit: int | None = None


@tool(name="read", description="Read the contents of a file.")
async def read(p: ReadParams, ctx: ToolContext) -> ToolResult: ...
```

Adding a tool is one file plus one `register()` call. `ToolContext` carries
`cwd`, a cancel token, and `on_update` for streaming partial results (bash
output as it arrives), so tools never reach for globals.

**`ToolKind`** is what makes both web-search modes coexist:

- `ToolKind.LOCAL` — the loop executes it.
- `ToolKind.SERVER` — the declaration is forwarded to the provider, which
  executes it; the loop sees results as `ProviderBlock`s and executes nothing.

`web_search` resolves to Anthropic's native server tool when the active model
supports it, and falls back to a local backend otherwise. `web_fetch` likewise,
with httpx plus HTML→markdown locally.

Local search backends (Tavily / Brave / Exa) are selected by env var behind a
`SearchBackend` protocol. **Note:** no search API key is currently configured in
the development environment, so the local path is unverifiable end-to-end until
one exists. The server-native path works today.

## 5. `session/`

Append-only JSONL, one file per session, under
`~/.harness/sessions/<project-slug>/<session-id>.jsonl`.

A header line (id, cwd, timestamp, model, system-prompt hash) followed by one
entry per line: `message`, `usage`. (`compaction` entries are reserved for when
§7 lands.) Resume replays the file; `--resume` with no argument takes the most
recent session for the cwd.

This reuses the serialization from §2, which is the argument for building that
first.

## 6. `cli/`

```
cli/
  main.py       # arg parsing, wiring
  terminal.py   # prompt_toolkit input + rich renderer
  renderers.py  # per-tool display (diffs, truncated bash output)
  approval.py   # y/n/always prompt → before_tool_call
  json_sink.py  # JSONL event writer
```

Two subscribers over one event stream:

- `harness` — interactive. prompt_toolkit for input (history, multi-line, Ctrl-C
  to interrupt mid-turn); rich for streaming output, markdown, and
  syntax-highlighted diffs. Inline scrollback, so output stays in real terminal
  history.
- `harness -p "…"` — one-shot text.
- `harness -p "…" --json` — the raw event stream as JSONL.

Ctrl-C cancels the in-flight turn without killing the session. Ctrl-D exits.

The JSONL sink is ~30 lines and exists for three reasons: it makes the runtime
scriptable, it lets end-to-end tests assert on an event stream instead of
scraping rendered terminal output, and — most importantly — a second consumer is
what keeps the "events are plain serializable data" rule from decaying.

## 7. Deferred: context compaction

Cut from v1 deliberately. It is the one item on the original list that is purely
additive, because `prepare_request` is a stable seam: compaction reads the
context, decides whether to summarize, and returns a replacement. Nothing in the
loop changes when it arrives.

To keep it cheap to add later, v1 commits to:

- `prepare_request` exists and is called before **every** provider request,
  including the first.
- `llm/types.py` defines `Model.context_window`, and `Usage` is recorded per
  assistant message, so token accounting has a home.
- `session/` reserves a `compaction` entry type in the JSONL schema.
- Token usage is surfaced in the CLI footer, so the need becomes visible before
  it becomes urgent.

When built, it becomes `context/` with `tokens.py` and `compaction.py`: a
threshold trigger, summarize-oldest with a cheap model, splice in a
`CompactionSummary` message, keep the most recent K turns, and never cut across
a tool-call/tool-result pair. Target ~200 lines; pi's 1100 covers branch
summarization and structured artifact indexes we are not building.

## 8. Error handling

Four rules, each enforced at one boundary:

1. **Providers** encode failures as `error` events — they never raise (§2).
2. **Tools** return `ToolResult(is_error=True)` or raise; the loop converts a
   raise into an error result. A tool can never kill the loop.
3. **Cancellation** propagates as `asyncio.CancelledError`. The turn ends with
   `stop_reason="aborted"`, in-flight tool calls get error results, and the
   transcript stays valid for the next request.
4. **Truncated responses** (`stop_reason="length"`) fail *all* of that message's
   tool calls rather than executing possibly-truncated arguments. pi learned
   this one the hard way: a best-effort JSON salvage parser can produce
   arguments that parse and validate while being silently incomplete.

## 9. Testing

pytest + pytest-asyncio, with `FakeProvider` fed scripted event sequences.

**The suite is offline by construction.** `tests/conftest.py` installs an
autouse fixture blocking outbound socket connections, so a test reaching for a
real API fails loudly instead of going flaky or costing tokens. Real end-to-end
verification against live APIs is manual, done by the user.

- **Serialization round-trip over every event type.** This is the test that
  keeps the frontend seam honest; without it the "plain data" rule decays.
- Loop: tool calls, parallel ordering, blocked calls, aborts, truncated
  responses.
- Tools: against `tmp_path`; web tools against injected fake backends.
- End-to-end: scripted provider drives a real tool, assert on the event stream.

Tooling: `uv`, ruff, pyright.

## 10. Build order

Each step is independently verifiable:

1. `llm/types.py` + `events.py` + serialization tests
2. `llm/fake.py` + `provider.py`
3. `agent/loop.py` against `FakeProvider`
4. `agent/agent.py` — state, subscribers, abort
5. `tools/base.py` + `registry.py` + `fs.py`
6. `cli/json_sink.py` → first end-to-end run
7. `llm/providers/anthropic.py` → first real run
8. `cli/terminal.py` — interactive
9. `tools/shell.py`, `search.py`, approval gate
10. `session/`
11. `tools/web.py` + `native.py`
12. `llm/providers/openai.py`

Documentation under `docs/` is written alongside each step, not after.
