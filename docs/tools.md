# Tools

> **Status:** Not yet implemented. This page describes the designed contract and
> is updated as `tools/` lands.

## The contract

A tool returns `ToolResult(is_error=True)` or raises; the loop converts a raise
into an error result. **A tool can never kill the loop.**

## Defining a tool

Parameters are a Pydantic model — one object gives JSON Schema, validation, and
type hints:

```python
from pydantic import BaseModel, Field
from harness.tools.base import ToolContext, ToolResult, tool


class ReadParams(BaseModel):
    path: str = Field(description="Path to the file, relative or absolute")
    offset: int | None = Field(None, description="1-indexed start line")
    limit: int | None = Field(None, description="Maximum lines to read")


@tool(name="read", description="Read the contents of a file.")
async def read(p: ReadParams, ctx: ToolContext) -> ToolResult: ...
```

Adding a tool is one file plus one `register()` call.

### `ToolContext`

| Field | Purpose |
|---|---|
| `cwd` | Working directory for this call |
| `cancel` | Cancellation token; long-running tools should check it |
| `on_update` | Emit a partial result (e.g. bash output as it arrives) |
| `model` | Active model, for tools that adapt to its capabilities |

Everything a tool needs arrives through this. Tools never reach for globals,
which is what makes them testable against `tmp_path`.

### Execution mode

Tools run in parallel by default. Declare `execution_mode = "sequential"` to opt
out — `write` and `edit` do, since concurrent mutation of the same file is not
something to leave to chance.

## `ToolKind`

| Kind | Who executes it |
|---|---|
| `LOCAL` | The loop |
| `SERVER` | The provider — the declaration is forwarded, results arrive as `ProviderBlock`s |

This distinction is what lets `web_search` be either a provider-native server
tool or a locally-executed one depending on the active model, without the loop
needing to care.

## Built-in tools

| Tool | Kind | Notes |
|---|---|---|
| `read` | local | Text and images; offset/limit; truncates with a continuation hint |
| `write` | local | Sequential |
| `edit` | local | Exact string replacement; sequential |
| `ls` | local | |
| `bash` | local | Streams output via `on_update`; timeout support |
| `grep` | local | |
| `glob` | local | |
| `web_search` | server or local | Native when the model supports it, else a search backend |
| `web_fetch` | server or local | URL → markdown |

## Web tools

`web_search` and `web_fetch` resolve at registration time based on the active
model:

- **Anthropic models with server-tool support** → the declaration is forwarded
  and the provider runs the search. No extra API key.
- **Everything else** → a local implementation behind a `SearchBackend`
  protocol, with Tavily / Brave / Exa selected by environment variable.
  `web_fetch` uses httpx plus HTML→markdown.

```
HARNESS_SEARCH_PROVIDER=tavily   # tavily | brave | exa
TAVILY_API_KEY=...
```

**Known gap:** no search API key is configured in the development environment,
so the local path is not verifiable end-to-end until one exists. The
server-native path works today. Local backends are tested against injected
fakes.

## Testing

Tools are tested against `tmp_path`; web tools against fake backends injected
through `ToolContext`. The suite is offline — see [`../AGENTS.md`](../AGENTS.md).
