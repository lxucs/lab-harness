# Providers

> **Status:** The `Provider` protocol, `StreamOptions`, `FakeProvider`, the
> model registry, and the **Anthropic** adapter are implemented. The
> OpenAI-compatible adapter arrives in step 12 — see the build order in
> [architecture.md](architecture.md).

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

| Provider | API | Status | Also reaches |
|---|---|---|---|
| `anthropic` | Messages | implemented | — |
| `openai` | Chat Completions | step 12 | DeepSeek, Groq, OpenRouter, vLLM, Ollama via `base_url` |

Two adapters is the minimum that keeps provider-specific assumptions out of the
core types. A single adapter would let Anthropic's shapes — thinking blocks,
`cache_control`, server tools — quietly become the core's shapes.

## The Anthropic adapter

`harness/llm/providers/anthropic.py`. Credentials come from the environment,
never from a config file or a constructor argument; config holds only the env
var *names*.

| Variable | Purpose |
|---|---|
| `ANTHROPIC_AUTH_TOKEN` | Bearer token. Takes precedence over the key. |
| `ANTHROPIC_API_KEY` | API key |
| `ANTHROPIC_BASE_URL` | Override the endpoint |

**Missing credentials do not raise.** The client is built in the constructor,
but a failure there is recorded and reported on the first `stream()` call as a
`StreamError` — because a missing key is a request failure like any other, and
the contract says the stream reports it. `_failure()` is the single place in the
module that builds one.

Four translation details that are not obvious:

- **Tool results merge.** The API requires every `tool_result` for a turn to
  arrive in one user message; the loop emits them as separate messages, so the
  adapter merges consecutive ones.
- **Empty content is dropped.** The API rejects a message with no content. An
  errored or aborted assistant turn has none, so it is not replayed.
- **Signature-less thinking blocks are dropped.** The API rejects a replayed
  thinking block without its signature.
- **Partial tool JSON is never salvaged.** A truncated argument object can parse
  into something that validates while silently missing fields. Malformed JSON
  yields `{}`, and the loop already refuses to execute calls from a
  length-stopped turn.

## Model registry

`harness/llm/registry.py` holds the catalog and `resolve_model()`. Precedence:
explicit argument, then `HARNESS_MODEL`, then the configured default.

An **unknown model id is not an error**. Model names ship faster than this
catalog does, so an unlisted id is assumed to be a current Anthropic model and
given conservative limits. Being unable to use a model released last week is
worse than a slightly wrong `context_window`.

Capability flags live with the model because the runtime acts on them:
`supports_reasoning` gates the thinking request, `supports_server_tools` will
decide whether `web_search` is forwarded or executed locally.

## Adding a provider

1. Add `llm/providers/<name>.py` implementing the `Provider` protocol.
2. Translate core `Message` objects to the provider's wire format, and the
   provider's stream back to `StreamEvent`s.
3. Map failures to an `error` event. Never raise.
4. Add its models to `llm/registry.py`.
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
