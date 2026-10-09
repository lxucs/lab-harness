# Providers

> **Status:** Not yet implemented. This page describes the designed contract and
> is updated as `llm/` lands.

## The contract

> A provider **must not raise** for request, model, or runtime failures.
> Failures terminate the stream with an `error` event carrying an
> `AssistantMessage` with `stop_reason="error"` and `error_message`.

This is lifted from pi verbatim, and it is why the agent loop has very few
try/excepts. Honor it when adding a provider.

```python
class Provider(Protocol):
    async def stream(
        self, model: Model, ctx: Context, opts: StreamOptions
    ) -> AsyncIterator[StreamEvent]: ...
```

## Stream protocol

```
start → {text,thinking,toolcall}_{start,delta,end}* → done | error
```

- `start` always comes first on a successful stream.
- A stream may terminate directly with `error` if request setup fails.
- Updates and `done` never appear before `start`.

## Core types

**Content blocks:** `TextBlock`, `ThinkingBlock`, `ImageBlock`, `ToolCallBlock`,
and `ProviderBlock`.

`ProviderBlock` is an opaque passthrough that round-trips provider-native
content the core does not model. It is how Anthropic's server-side
`web_search_tool_use` and `web_search_tool_result` blocks survive a turn without
leaking into the core types. If you are tempted to add a provider-specific block
type to `llm/types.py`, use `ProviderBlock` instead.

**Messages:** `SystemMessage`, `UserMessage`, `AssistantMessage`,
`ToolResultMessage`. `AssistantMessage` carries `usage`, `stop_reason`,
`error_message`, and the model and provider that produced it.

**`Model`** carries `context_window`, `max_tokens`, cost rates, and capability
flags (reasoning, image input, server tools). `context_window` exists in v1 even
though compaction does not — it is where token accounting will hang.

## Shipped providers

| Provider | API | Also reaches |
|---|---|---|
| `anthropic` | Messages | — |
| `openai` | Chat Completions | DeepSeek, Groq, OpenRouter, vLLM, Ollama via `base_url` |

Two adapters is the minimum that keeps provider-specific assumptions out of the
core types. A single adapter would let Anthropic's shapes — thinking blocks,
`cache_control`, server tools — quietly become the core's shapes.

## Adding a provider

1. Add `llm/providers/<name>.py` implementing the `Provider` protocol.
2. Translate core `Message` objects to the provider's wire format, and the
   provider's stream back to `StreamEvent`s.
3. Map failures to an `error` event. Never raise.
4. Register it in `llm/registry.py`.
5. Update the table above.

## Testing

`llm/fake.py` provides `FakeProvider`, which replays a scripted list of
`StreamEvent`s. Every loop and agent test uses it. The suite never touches the
network — see [`../AGENTS.md`](../AGENTS.md).
