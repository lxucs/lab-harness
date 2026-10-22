# Providers

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
- A 2xx body that never sends a frame ends in `error`, not an empty `done`.
  Gateways refuse with a 200 and a JSON body — `{"code": 1005, "msg":
  "invalid model"}` from one asked for an id it does not serve — which, read
  as a stream, is a turn that said nothing. Both HTTP adapters read the body
  through `_http.Frames`, which reports it like an error status.

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

**`Model`** carries `context_window`, `max_tokens`, and capability flags
(reasoning, image input, server tools). `context_window` exists in v1 even
though compaction does not — it is where token accounting will hang, and the
terminal already measures context against it. Prices are deliberately *not*
on `Model`: they vary by vendor, platform and contract, so they live in
config — see [`models`](configuration.md#models).

## Shipped providers

Modules are named for the **protocol**, not the vendor, because the mapping
stopped being one-to-one: OpenAI moved to the Responses API while everyone
else kept speaking the dialect OpenAI invented.

| Module | Protocol | Reaches |
|---|---|---|
| `anthropic.py` | Messages | Claude |
| `responses.py` | OpenAI Responses | OpenAI |
| `chat_completions.py` | Chat Completions | DeepSeek, GLM, Groq, OpenRouter, Together, vLLM, Ollama |

Chat Completions is **not** the legacy path here — it is the lingua franca, and
the only one most vendors offer.

Three adapters, not one, is what keeps provider-specific assumptions out of the
core types. A single adapter would let whichever API was written first — its
thinking blocks, its `cache_control`, its item model — quietly become the
core's shapes.

A new vendor is still normally a *profile*, not a module. See
[Profiles](#profiles).

Each shipped adapter can also say what it sends, outside a run:
`request_body(model, ctx)` is the JSON a request posts, and
`reply_entries(message)` a reply as its dialect writes it into the next
request's history. That is how a session's requests are rebuilt
[as the endpoint received them](sessions.md#as-the-provider-received-it).

## The Anthropic adapter

`lab_harness/llm/providers/anthropic.py`. Credentials come from the environment,
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
- **Empty content is dropped.** The API rejects a message with no content, so an
  errored turn — which has none — is not replayed. An aborted one carries what
  the stream had reached, minus any tool call; see [Abort](events.md#abort).
- **Signature-less thinking blocks are dropped.** The API rejects a replayed
  thinking block without its signature.
- **Partial tool JSON is never salvaged.** A truncated argument object can parse
  into something that validates while silently missing fields. Malformed JSON
  yields `{}`, and the loop already refuses to execute calls from a
  length-stopped turn.

**TODO: prompt caching.** The adapter sends no `cache_control` markers, and
Anthropic caches nothing without them, so every request pays full price for
its prompt and the terminal's cache figure reads 0%. The plan: one marker on
the system prompt, which caches the tools rendered ahead of it too, and one on
the last block of each of the last two user messages. The second of those sits
exactly where the previous request wrote its entry, so a turn that adds more
than the API's 20-block lookback — many parallel tool calls — still hits.
Three markers, under the limit of four. `Usage` already reports cache reads
and writes, and the shipped [prices](configuration.md#prices) already include
both rates, so nothing downstream changes when it lands.

### The thinking request

Two request shapes, chosen per model:

| Models | Sent | Thinks |
|---|---|---|
| Fable 5, Opus 5, Sonnet 5, and any Claude id not listed below | `thinking: {type: "adaptive", display: "summarized"}`, on every request | Always — they think by default |
| Everything before 4.6 — `budget_thinking_models`, by id prefix | `thinking: {type: "enabled", budget_tokens: …}`, only when an effort is set | Only when asked |

**Adaptive is sent even though the models think unasked**, because the
request is the only place to ask for `display`. The 5-series models default it
to `"omitted"`: thinking blocks arrive with empty text, which to a terminal
looks like a long silent pause. `"summarized"` gets a readable summary; the
raw chain is never returned either way, and billing is the same.
`providers.anthropic.thinking_display` sets it, and `""` sends no preference.

**A budget never reaches an adaptive model** — the 5-series rejects one with a
400. Effort is the dial there instead: the model's `reasoning_effort` goes out
as `output_config.effort`, and none is sent when it is empty, leaving the
API's default. A budget model has no effort, so any level only switches its
thinking on, at `thinking_budget_tokens`. Where the level comes from:
[cli.md](cli.md#reasoning-effort).

**An unlisted id is assumed current**, the catalog's rule everywhere else, so
it gets adaptive. That is why the budget list names the old models rather
than the new ones: a model released next month should show its thinking
without a config change, and an old one sent adaptive would fail. Either
kind drops `temperature`, which the API rejects alongside thinking.

Thinking blocks come back signed, and are replayed with their signature
exactly as received — empty text included.

## How thinking is handled across models

The hardest thing to make provider-neutral, because vendors disagree about
whether reasoning is *visible* and whether it is *replayable* — and those are
independent questions. So the core type asks both:

```python
class ThinkingBlock:
    thinking: str  # can you read it?
    signature: str | None = None  # must you hand it back?
```

That gives four regimes, and all four are reachable:

| Provider | `thinking` | `signature` | Replayed? |
|---|---|---|---|
| Anthropic | a **summary** on the 5-series ([asked for](#the-thinking-request)); the text before | the signed token | **Yes** — the API rejects it without the signature |
| OpenAI via `responses` | a model-written **summary** | packed `{id, encrypted_content}` | **Yes** — verbatim, or the model loses its place |
| DeepSeek, GLM | the real chain, in plaintext | `None` | **Yes** — as plain `reasoning_content`; DeepSeek returns a 400 for a tool loop without it |
| OpenAI via `chat_completions` | *no block at all* | — | nothing to replay |

**Why OpenAI needs the replay.** A reasoning model thinks, calls a tool, and
then has to resume from where it left off. OpenAI never returns the raw chain;
it returns a summary you can show a user and an opaque `encrypted_content`
blob that is the actual continuation. Drop the blob and nothing errors — the
model just restarts its reasoning after every tool call, silently and worse.
That is the whole reason `responses.py` exists.

**Why the signature holds JSON there.** Replaying needs the item's `id` as
well as the ciphertext. Rather than add an OpenAI-shaped field to
`llm/types.py`, the adapter packs both into the one field the core already
documents as *an opaque provider replay token* — and it is the only thing that
ever looks inside.

**The last row is not a bug.** On Chat Completions, OpenAI's reasoning models
stream no reasoning content whatsoever, so no block is created. Absence is the
honest representation; there is genuinely nothing to show. (The token count is
still right: `output` includes reasoning tokens. What is lost is the ability to
*break out* how many — the core `Usage` has no field for it.)

**Why DeepSeek and GLM need it too.** No token to hand back, but the text
itself. DeepSeek's rule depends on the request: with `tools`, every earlier
turn's `reasoning_content` must come back — including turns that called no
tool — or the API returns a 400; without `tools` it is ignored. GLM asks for it
complete and unmodified, and says dropping it costs quality and cache hits —
but its standard endpoint discards it anyway unless the request opts into
*Preserved Thinking*, so the shipped `glm` profile also sets
`preserve_thinking`. Both cache by prefix, so the text goes back byte for byte
as it streamed.

This used to be the other way round: DeepSeek once rejected reasoning sent
back. Replay is therefore a per-profile setting, `replay_reasoning_field`, on
for the shipped `deepseek` and `glm` profiles and off for any profile you add —
a server that never asked for the field may reject it. See
[Profiles](#profiles).

**Mixing providers mostly degrades.** Each adapter drops reasoning it cannot
legitimately replay: a signature-less block never reaches Anthropic or
Responses, and a signed one never reaches Chat Completions. So `--resume`
against a different provider usually costs the model some context and nothing
else; both directions are tested. The exception is untested: DeepSeek's rule
covers *all* earlier turns, so a session started elsewhere and resumed on
DeepSeek with tools may be refused for the turns that carry no DeepSeek
reasoning.

## The Chat Completions adapter

`lab_harness/llm/providers/chat_completions.py`. Raw `httpx` rather than the
`openai` SDK: httpx is already a dependency for the web tools, the SSE framing
is forty lines, and `reasoning_content` sits outside the SDK's types anyway.

Five translation details, each forced by the wire format rather than chosen:

- **The system prompt is a message**, not a top-level field. The other two
  adapters do the opposite.
- **Tool results stay separate** `{"role": "tool"}` messages. Anthropic
  requires every result for a turn to be merged into one user message; this
  dialect requires exactly the opposite, and getting it backwards is a 400.
- **Tool arguments go back as a JSON string**, which is how they arrived.
- **Reasoning goes back as a sibling field**, not a content block — on an
  assistant message beside `content` and `tool_calls`, and only for a profile
  that sets `replay_reasoning_field`. See the table above.
- **Block indices are assigned here.** The stream reports no block boundaries —
  no start, no stop, and no index for text — so the adapter assigns a stable
  `block_index` on first sight and closes every open block when the stream
  ends. The event protocol holds even though the wire format does not supply it.

Reasoning deltas are read from `reasoning_content` (DeepSeek's name) or
`reasoning` (what several gateways relabel it to).

## The Responses adapter

`lab_harness/llm/providers/responses.py`. Opt in with `api = "responses"`.

Not a tweak of Chat Completions — a different shape:

- **The conversation is a list of items** (`message`, `function_call`,
  `function_call_output`, `reasoning`) rather than messages with roles. A tool
  call and its result are siblings. One of our `AssistantMessage`es can become
  three items, and reasoning must come first.
- **The system prompt is top-level `instructions`**, as in the Messages API.
- **Tool declarations are flat** — `{"type": "function", "name": …}` — where
  Chat Completions nests everything under a `function` key.
- **`max_output_tokens`**, not `max_tokens`.
- **Reasoning is a first-class item** carrying the encrypted continuation.

We send `store: false` and ask for `reasoning.encrypted_content`. The
alternative is letting OpenAI keep the thread and referring to it by
`previous_response_id` — which would put the authoritative history on someone
else's server, where `--resume` and the session files cannot see it. The two
are alternatives, not complements: `store = true` disables the encrypted
replay.

Errors can arrive *inside* a 200 stream (`error`, `response.failed`), so those
events end the turn rather than being dropped.

**Partial tool JSON is never salvaged** in any adapter — a truncated object can
parse into arguments that validate while silently missing fields.

## Profiles

A vendor is config, not code. `providers.openai` is a table of named profiles,
and one of only two places in the schema where an unknown key is *not* an
error — a new key is a new vendor. The other is `models`, where a model names
the profile that serves it.

```toml
[providers.openai.groq]
api = "chat_completions"      # the default
base_url = "https://api.groq.com/openai/v1"
api_key_env = "GROQ_API_KEY"

[models."llama-3.3-70b-versatile"]
provider = "groq"
```

`api` picks the protocol: `chat_completions` (default, since that is what a new
vendor almost certainly speaks) or `responses`. The shipped `openai` profile —
like every shipped endpoint, in
[`lab_harness/llm/catalog.toml`](../lab_harness/llm/catalog.toml) — sets
`responses`; `deepseek` and `glm` use the default. An unusable value fails at
startup naming the profile.

`replay_reasoning_field` names the assistant-message field earlier reasoning
goes back in. The shipped `deepseek` and `glm` profiles set
`"reasoning_content"`; every other profile starts at `""`, sending none. Set it
for a reasoning vendor only when its docs ask for reasoning back, and to the
field they name — a vendor that wants it and does not get it may refuse the
request, and one that does not want it may refuse the unknown field.

`preserve_thinking` adds `"thinking": {"type": "enabled", "clear_thinking":
false}` — GLM's Preserved Thinking, exactly as Z.ai documents it. Only the
`glm` profile sets it: the shape is GLM's, and `type: "enabled"` also switches
thinking on for GLM-4.6, the one model in the lineup that otherwise decides for
itself. Z.ai documents it for GLM-4.7 and recommends it for GLM-5.3-Flash; it
does not say whether GLM-4.6 accepts it.

Keys inside a profile are checked as strictly as anywhere else, so a typo there
still fails at startup.

**Which provider serves a model**, highest precedence first:

1. `--provider`
2. the `provider` of the id's [`models`](configuration.md#models) entry — so
   `--model deepseek-flash` reaches DeepSeek with no flag
3. `agent.provider` (default `openai`)

Rung 2 is config like any other, so re-pointing a catalogued model at a
gateway, or at a local server running the same weights, is one line:
`[models.claude-sonnet-5] provider = "my-gateway"`.

`select_provider`, `build_provider` and `route` live in
`lab_harness/llm/providers/__init__.py`. `route` is the one the CLI calls; it
resolves the id, picks the provider, and writes it onto `Model.provider` — an
uncatalogued id resolves naming none, and `--provider` can override the one a
catalogued id names.

A profile's credential resolves from the environment first, then
`config.local.toml` — never a committed file.

## The model catalog

Every LLM setting lab-harness ships is data rather than code:
[`lab_harness/llm/catalog.toml`](../lab_harness/llm/catalog.toml) holds
Claude's adapter settings, the endpoints, every model, and in `[defaults]` what
an endpoint or a model gets for a key it leaves out. It is the default of the
`providers` and `models` config sections and uses their format exactly, so a
config file adds a model or a vendor, or changes one, with the same tables. One
`[models.<id>]` table holds everything about a model — who serves it, its
limits and flags, its price; the keys are in
[configuration.md](configuration.md#models).

`resolve_model()` in `lab_harness/llm/registry.py` turns an id into the `Model`
the adapters read. Precedence: explicit argument, then `LAB_HARNESS_MODEL`,
then the configured default.

An **unknown model id is not an error**. Model names ship faster than this
catalog does, so an unlisted id is given conservative limits rather than
refused. Being unable to use a model released last week is worse than a
slightly wrong `context_window`. Routing still works for it: it goes to
`agent.provider`, OpenAI by default, unless a `[models]` entry or `--provider`
names another.

A model's `provider` names the *profile* that serves it, not the
dialect it speaks — `deepseek-flash` says `deepseek`, not `openai`.

Capability flags live with the model, not at the call site.
`supports_reasoning` gates the thinking request; for Claude it decides whether
one is sent at all ([The thinking request](#the-thinking-request)). The level
that request asks for lives there too, as `Model.reasoning_effort` — from the
model's entry, `agent.reasoning_effort`, or `--reasoning-effort`
([cli.md](cli.md#reasoning-effort)) — because a level one model takes, another
rejects. Each adapter sends it in its own dialect; `StreamOptions.reasoning`
overrides it for a single request, `""` included.
`supports_server_tools` only records a capability; provider-native
`web_search` is not planned (see [tools.md](tools.md)).

## Adding a vendor

**If it speaks either protocol — which almost all do — write no code.** Add a
profile, and a `[models]` entry for each model it serves:

```toml
[providers.openai.myvendor]
base_url = "https://api.myvendor.example/v1"
api_key_env = "MYVENDOR_API_KEY"

[models.their-model-v1]
provider = "myvendor"
```

`lab-harness --model their-model-v1` now routes there. Add its real limits,
capability flags and price to the same `[models]` table if you have them —
[configuration.md](configuration.md#models) lists the keys. For a
reasoning model, check the vendor's docs on sending reasoning back before
setting `replay_reasoning_field` — see [Profiles](#profiles).

**A genuinely different API** needs an adapter:

1. Add `llm/providers/<protocol>.py` implementing the `Provider` protocol.
   Name it for the protocol, not the vendor — the mapping is not one-to-one.
2. Translate core `Message` objects to the wire format, and the stream back to
   `StreamEvent`s. Reuse `_http.py` for credentials, the client, SSE framing
   and failures; read the body through `_http.Frames`, which turns a 200 that
   never sends a frame — a gateway refusing a model id, say — into an error
   rather than an empty answer. Log the request with `_trace.request_shape`,
   and the response's id and stop reason as the vendor sent them, at `debug`.
3. Map failures to an `error` event through `_http.failure`. Never raise.
4. Decide the reasoning regime from the table above, and say so in the module
   docstring. Replaying something the server cannot decrypt is a 400; dropping
   something it needed is a silent loss of context.
5. Expose `request_body`, the JSON `stream` posts, and `reply_entries`, a reply
   as the next request carries it, and add the class to `Adapter` — they are
   how a session's requests are rebuilt
   [as the endpoint received them](sessions.md#as-the-provider-received-it).
   Test the first against what `stream` actually posts.
6. Register it in `PROTOCOLS` so a profile can name it.
7. Add its models to `llm/catalog.toml` and update the table above.

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
