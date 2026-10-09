# Providers

> **Status:** The `Provider` protocol, `StreamOptions`, and `FakeProvider` are
> implemented. Real adapters (Anthropic, OpenAI-compatible) arrive in a later
> milestone — see the build order in [architecture.md](architecture.md).

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

Build scripts with the helpers rather than by hand — they emit protocol-correct
sequences, so a test that passes is testing the loop and not a malformed script:

| Helper | Produces |
|---|---|
| `text_response(value, chunk_size=0)` | A complete text turn; `chunk_size` splits it into multiple deltas |
| `tool_call_response(calls, preamble=None)` | A turn requesting tool calls, `stop_reason="tool_use"` |
| `error_response(error, reason="error")` | A failed turn — **a lone `StreamError`, with no leading `StreamStart`** |
| `truncated_tool_call_response(calls)` | Tool calls cut off by the token limit, `stop_reason="length"` |

That `error_response` detail is a real part of the protocol, not a shortcut in
the fake: a stream may terminate with `error` as its *first* event when request
setup fails before generation starts. Consumers must not assume a `StreamStart`
preceded it.

`FakeProvider` raises `AssertionError` when its script is exhausted. That does
not violate the never-raise contract — a run making more requests than its
script provides is a bug in the test, which is a programming error rather than a
request failure.
