# Harness Runtime Core — Implementation Plan (Milestone 1a)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully tested agentic loop driven by scripted model responses — core types, event protocol, provider protocol, the loop, and the stateful `Agent`.

**Architecture:** Three source packages. `harness/llm/` holds provider-neutral message/event types and the `Provider` protocol. `harness/agent/` holds a pure `run_agent_loop()` function and an `Agent` class that wraps it with transcript state, subscribers, abort, and a message queue. Everything is exercised through `FakeProvider`, which replays scripted stream events — no network, no tools, no CLI in this milestone.

**Tech Stack:** Python 3.12+, asyncio, frozen dataclasses (protocol types), pytest 9 + pytest-asyncio 1.4 (`asyncio_mode = "auto"`), ruff, pyright, uv.

**Scope:** Build-order steps 1–4 from [the design spec](../specs/2026-10-09-harness-runtime-design.md). Tools, providers, session persistence, and the terminal are later milestones. At the end of this plan nothing is user-facing; the deliverable is a tested loop.

## Global Constraints

- **Python 3.12+.** Use `X | None`, not `Optional[X]`. Use `list[X]`, not `List[X]`.
- **Offline tests.** `tests/conftest.py` blocks outbound sockets. Never call a real API in a test.
- **Events are frozen dataclasses of JSON primitives.** No live objects, file handles, SDK types, or callables on any event or message. Enforced by `tests/test_serialization.py`.
- **Providers never raise** for request/model/runtime failures — they terminate the stream with an `error` event.
- **No hardcoded tunables.** Anything configurable lives in `harness/config.py`. Protocol constants (schema versions, API version headers) are exempt.
- **No secrets in committed files.** No keys, absolute paths, private URLs.
- **Mutable containers on frozen dataclasses are `tuple`, not `list`** — hashability and accidental-mutation safety.
- **Do not commit unless the user explicitly asks.** Each task ends with a *proposed* commit command; run it only on instruction. See [AGENTS.md](../../../AGENTS.md).

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, deps, ruff/pyright/pytest config |
| `harness/config.py` | Config schema + defaults — the single list of tunables |
| `harness/llm/types.py` | Content blocks, messages, `Usage`, `StopReason`, `Model`, `Context` |
| `harness/llm/events.py` | `StreamEvent` union — the provider wire protocol |
| `harness/llm/serde.py` | `to_dict` / `from_dict` for the tagged unions |
| `harness/llm/provider.py` | `Provider` protocol, `StreamOptions` |
| `harness/llm/fake.py` | `FakeProvider` — replays scripted events |
| `harness/agent/types.py` | `AgentEvent` union, `AgentTool`, `AgentContext`, hook protocols |
| `harness/agent/loop.py` | `run_agent_loop()` — pure async, no state |
| `harness/agent/agent.py` | `Agent` — transcript, subscribers, abort, queue |
| `tests/conftest.py` | Autouse socket blocker + shared fixtures |

Task 1 creates the first four rows, Task 2 the next two, Task 3 the loop, Task 4 the `Agent`.

---

## Task 1: Project scaffold, core types, and the serialization contract

Sets up the package and defines the types every later task consumes. Ends with the round-trip test that enforces the "plain data" invariant.

**Files:**
- Create: `pyproject.toml`, `.python-version`
- Create: `harness/__init__.py`, `harness/config.py`
- Create: `harness/llm/__init__.py`, `harness/llm/types.py`, `harness/llm/events.py`, `harness/llm/serde.py`
- Create: `tests/conftest.py`, `tests/test_serialization.py`

**Interfaces:**
- Consumes: nothing — this is the root task.
- Produces:
  - `TextBlock(text, type="text")`, `ThinkingBlock(thinking, signature, type="thinking")`, `ToolCallBlock(id, name, arguments, type="tool_call")`, `ProviderBlock(provider_type, data, type="provider")` — all frozen, all with a `type` literal discriminator
  - `ContentBlock` = union of the four
  - `SystemMessage(content, role="system")`, `UserMessage(content, role="user")`, `AssistantMessage(content, model, provider, usage, stop_reason, error_message, role="assistant")`, `ToolResultMessage(tool_call_id, tool_name, content, is_error, duration_ms, role="tool_result")`
  - `Message` = union of the four
  - `Usage(input, output, cache_read, cache_write, total)`
  - `StopReason` = `Literal["stop", "length", "tool_use", "error", "aborted"]`
  - `Model(id, name, provider, context_window, max_tokens, supports_reasoning, supports_images, supports_server_tools)`
  - `ToolDeclaration(name, description, parameters)`
  - `Context(system_prompt, messages, tools)`
  - `StreamEvent` union (see Step 5)
  - `to_dict(obj) -> dict`, `message_from_dict(d) -> Message`, `block_from_dict(d) -> ContentBlock`, `event_from_dict(d) -> StreamEvent`
  - `Config`, `load_config() -> Config`

- [ ] **Step 1: Create the project scaffold**

```bash
cd /path/to/lab-harness
uv init --name harness --python 3.12 --no-readme --bare
uv add --dev pytest pytest-asyncio ruff pyright
mkdir -p harness/llm harness/agent tests
touch harness/__init__.py harness/llm/__init__.py harness/agent/__init__.py
```

Then replace `pyproject.toml` entirely with:

```toml
[project]
name = "harness"
version = "0.1.0"
description = "A lightweight agent harness"
requires-python = ">=3.12"
dependencies = []

[project.scripts]
harness = "harness.cli.main:main"

[dependency-groups]
dev = ["pytest>=9.1", "pytest-asyncio>=1.4", "ruff>=0.14", "pyright>=1.1.400"]

[build-system]
requires = ["uv_build>=0.9,<0.10"]
build-backend = "uv_build"

[tool.uv.build-backend]
module-name = "harness"
module-root = ""

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "RUF"]

[tool.pyright]
include = ["harness", "tests"]
pythonVersion = "3.12"
typeCheckingMode = "standard"
```

The `[project.scripts]` entry points at a module that does not exist until a
later milestone. That is fine — it is metadata, not an import. Do not create a
stub for it.

- [ ] **Step 2: Write the offline-enforcement conftest**

Create `tests/conftest.py`:

```python
"""Shared fixtures. The autouse network blocker is the offline guarantee."""

import socket

import pytest


class NetworkBlockedError(AssertionError):
    """Raised when a test attempts a network connection."""


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly on any outbound connection.

    Tests must never reach a real API: it costs tokens, goes flaky, and hides
    missing fakes. Real end-to-end verification is manual.
    """

    def deny(self: socket.socket, *args: object, **kwargs: object) -> None:
        raise NetworkBlockedError(
            "Network access is blocked in tests. Use FakeProvider or an injected fake."
        )

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
```

- [ ] **Step 3: Verify the blocker works**

Create `tests/test_offline.py`:

```python
import socket

import pytest

from tests.conftest import NetworkBlockedError


def test_network_is_blocked() -> None:
    with pytest.raises(NetworkBlockedError):
        socket.create_connection(("example.com", 80), timeout=1)


def test_loopback_is_blocked_too() -> None:
    with pytest.raises(NetworkBlockedError):
        socket.socket().connect(("127.0.0.1", 9))
```

Run: `uv run pytest tests/test_offline.py -v`
Expected: 2 passed. If either fails, stop — the offline guarantee is broken and nothing else in this plan is trustworthy.

- [ ] **Step 4: Write the core types**

Create `harness/llm/types.py`:

```python
"""Provider-neutral message and model types.

Every type here is a frozen dataclass of JSON primitives. Tuples rather than
lists, so instances stay hashable and cannot be mutated by accident. The `type`
and `role` literals are discriminators for serialization — see serde.py.
"""

from dataclasses import dataclass, field
from typing import Literal

# --- content blocks -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True, slots=True)
class ThinkingBlock:
    thinking: str
    #: Opaque provider replay token. Required by some providers on the next turn.
    signature: str | None = None
    type: Literal["thinking"] = "thinking"


@dataclass(frozen=True, slots=True)
class ImageBlock:
    #: base64-encoded image data
    data: str
    mime_type: str
    type: Literal["image"] = "image"


@dataclass(frozen=True, slots=True)
class ToolCallBlock:
    id: str
    name: str
    arguments: dict[str, object]
    type: Literal["tool_call"] = "tool_call"


@dataclass(frozen=True, slots=True)
class ProviderBlock:
    """Opaque passthrough for provider-native content the core does not model.

    Anthropic's server-side `web_search_tool_use` / `web_search_tool_result`
    blocks ride through here. Prefer this over adding a provider-specific block
    type to the core.
    """

    provider_type: str
    data: dict[str, object]
    type: Literal["provider"] = "provider"


ContentBlock = TextBlock | ThinkingBlock | ImageBlock | ToolCallBlock | ProviderBlock

# --- usage and stop reasons ----------------------------------------------

StopReason = Literal["stop", "length", "tool_use", "error", "aborted"]


@dataclass(frozen=True, slots=True)
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0

    @property
    def total(self) -> int:
        return self.input + self.output + self.cache_read + self.cache_write


# --- messages -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SystemMessage:
    content: str
    role: Literal["system"] = "system"


@dataclass(frozen=True, slots=True)
class UserMessage:
    content: tuple[ContentBlock, ...]
    role: Literal["user"] = "user"


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    content: tuple[ContentBlock, ...]
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason = "stop"
    error_message: str | None = None
    role: Literal["assistant"] = "assistant"

    @property
    def tool_calls(self) -> tuple[ToolCallBlock, ...]:
        return tuple(b for b in self.content if isinstance(b, ToolCallBlock))


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    tool_call_id: str
    tool_name: str
    content: tuple[ContentBlock, ...]
    is_error: bool = False
    duration_ms: int | None = None
    role: Literal["tool_result"] = "tool_result"


Message = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage

# --- models and request context -------------------------------------------


@dataclass(frozen=True, slots=True)
class Model:
    id: str
    name: str
    provider: str
    context_window: int
    max_tokens: int
    supports_reasoning: bool = False
    supports_images: bool = False
    #: Provider executes some tools itself (e.g. Anthropic web_search).
    supports_server_tools: bool = False


@dataclass(frozen=True, slots=True)
class ToolDeclaration:
    """What the model is told a tool looks like. Not the executable tool."""

    name: str
    description: str
    #: JSON Schema object
    parameters: dict[str, object]


@dataclass(frozen=True, slots=True)
class Context:
    """One request's input.

    Unlike pi, the prompt and tool declarations are passed directly rather than
    replayed from system messages in the transcript.
    """

    system_prompt: str
    messages: tuple[Message, ...]
    tools: tuple[ToolDeclaration, ...] = ()


def text(value: str) -> tuple[TextBlock]:
    """Shorthand for the common single-text-block content."""
    return (TextBlock(text=value),)
```

- [ ] **Step 5: Write the stream event protocol**

Create `harness/llm/events.py`:

```python
"""Provider stream protocol.

    start -> {text,thinking,toolcall}_{start,delta,end}* -> done | error

`start` always precedes updates. A stream may terminate directly with `error`
when request setup fails. Updates and `done` never appear before `start`.
"""

from dataclasses import dataclass
from typing import Literal

from harness.llm.types import AssistantMessage, StopReason, ToolCallBlock


@dataclass(frozen=True, slots=True)
class StreamStart:
    type: Literal["start"] = "start"


@dataclass(frozen=True, slots=True)
class TextStart:
    block_index: int
    type: Literal["text_start"] = "text_start"


@dataclass(frozen=True, slots=True)
class TextDelta:
    block_index: int
    delta: str
    type: Literal["text_delta"] = "text_delta"


@dataclass(frozen=True, slots=True)
class TextEnd:
    block_index: int
    text: str
    type: Literal["text_end"] = "text_end"


@dataclass(frozen=True, slots=True)
class ThinkingStart:
    block_index: int
    type: Literal["thinking_start"] = "thinking_start"


@dataclass(frozen=True, slots=True)
class ThinkingDelta:
    block_index: int
    delta: str
    type: Literal["thinking_delta"] = "thinking_delta"


@dataclass(frozen=True, slots=True)
class ThinkingEnd:
    block_index: int
    thinking: str
    type: Literal["thinking_end"] = "thinking_end"


@dataclass(frozen=True, slots=True)
class ToolCallStart:
    block_index: int
    type: Literal["toolcall_start"] = "toolcall_start"


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    block_index: int
    delta: str
    type: Literal["toolcall_delta"] = "toolcall_delta"


@dataclass(frozen=True, slots=True)
class ToolCallEnd:
    block_index: int
    tool_call: ToolCallBlock
    type: Literal["toolcall_end"] = "toolcall_end"


@dataclass(frozen=True, slots=True)
class StreamDone:
    message: AssistantMessage
    reason: StopReason
    type: Literal["done"] = "done"


@dataclass(frozen=True, slots=True)
class StreamError:
    message: AssistantMessage
    reason: Literal["error", "aborted"]
    type: Literal["error"] = "error"


StreamEvent = (
    StreamStart
    | TextStart
    | TextDelta
    | TextEnd
    | ThinkingStart
    | ThinkingDelta
    | ThinkingEnd
    | ToolCallStart
    | ToolCallDelta
    | ToolCallEnd
    | StreamDone
    | StreamError
)
```

- [ ] **Step 6: Write the failing serialization test**

Create `tests/test_serialization.py`:

```python
"""The round-trip test that keeps the 'events are plain data' rule from decaying.

If this fails because someone attached a live object to an event, the fix is to
remove the object — not to special-case it here.
"""

import json

import pytest

from harness.llm.events import (
    StreamDone,
    StreamError,
    StreamStart,
    TextDelta,
    TextEnd,
    TextStart,
    ThinkingDelta,
    ThinkingEnd,
    ThinkingStart,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)
from harness.llm.serde import block_from_dict, event_from_dict, message_from_dict, to_dict
from harness.llm.types import (
    AssistantMessage,
    ImageBlock,
    ProviderBlock,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
    text,
)

BLOCKS = [
    TextBlock(text="hello"),
    ThinkingBlock(thinking="hmm", signature="sig-abc"),
    ImageBlock(data="aGk=", mime_type="image/png"),
    ToolCallBlock(id="call_1", name="read", arguments={"path": "a.py", "limit": 10}),
    ProviderBlock(provider_type="web_search_tool_result", data={"results": []}),
]

MESSAGES = [
    SystemMessage(content="You are a harness."),
    UserMessage(content=text("hi")),
    AssistantMessage(
        content=(TextBlock(text="sure"), ToolCallBlock(id="c1", name="ls", arguments={})),
        model="m",
        provider="p",
        usage=Usage(input=10, output=5),
        stop_reason="tool_use",
    ),
    AssistantMessage(
        content=(),
        model="m",
        provider="p",
        stop_reason="error",
        error_message="boom",
    ),
    ToolResultMessage(tool_call_id="c1", tool_name="ls", content=text("a.py"), duration_ms=12),
]

_ASSISTANT = AssistantMessage(content=text("done"), model="m", provider="p")

EVENTS = [
    StreamStart(),
    TextStart(block_index=0),
    TextDelta(block_index=0, delta="he"),
    TextEnd(block_index=0, text="hello"),
    ThinkingStart(block_index=1),
    ThinkingDelta(block_index=1, delta="..."),
    ThinkingEnd(block_index=1, thinking="..."),
    ToolCallStart(block_index=2),
    ToolCallDelta(block_index=2, delta='{"p":'),
    ToolCallEnd(block_index=2, tool_call=ToolCallBlock(id="c1", name="read", arguments={"p": "a"})),
    StreamDone(message=_ASSISTANT, reason="stop"),
    StreamError(message=_ASSISTANT, reason="error"),
]


@pytest.mark.parametrize("block", BLOCKS, ids=lambda b: b.type)
def test_block_roundtrip(block: object) -> None:
    assert block_from_dict(json.loads(json.dumps(to_dict(block)))) == block


@pytest.mark.parametrize(
    "message", MESSAGES, ids=lambda m: f"{m.role}-{m.stop_reason if m.role == 'assistant' else ''}"
)
def test_message_roundtrip(message: object) -> None:
    assert message_from_dict(json.loads(json.dumps(to_dict(message)))) == message


@pytest.mark.parametrize("event", EVENTS, ids=lambda e: e.type)
def test_event_roundtrip(event: object) -> None:
    assert event_from_dict(json.loads(json.dumps(to_dict(event)))) == event


def test_every_event_type_is_covered() -> None:
    """Guard against adding an event type without adding it to EVENTS above.

    Without this, a new event type silently escapes the round-trip check and
    the 'plain data' rule decays one event at a time.
    """
    from typing import get_args

    from harness.llm import events as events_module

    union_members = set(get_args(events_module.StreamEvent))
    covered = {type(e) for e in EVENTS}
    assert covered == union_members
```

- [ ] **Step 7: Run the test to verify it fails**

Run: `uv run pytest tests/test_serialization.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'harness.llm.serde'`

- [ ] **Step 8: Implement serde**

Create `harness/llm/serde.py`:

```python
"""Tagged-union serialization.

`dataclasses.asdict` handles nesting and gives plain dicts. Deserialization
dispatches on the `type` / `role` discriminator each type carries.
"""

from dataclasses import asdict, is_dataclass
from typing import Any

from harness.llm import events as _events
from harness.llm import types as _types

_BLOCKS: dict[str, type] = {
    "text": _types.TextBlock,
    "thinking": _types.ThinkingBlock,
    "image": _types.ImageBlock,
    "tool_call": _types.ToolCallBlock,
    "provider": _types.ProviderBlock,
}

_MESSAGES: dict[str, type] = {
    "system": _types.SystemMessage,
    "user": _types.UserMessage,
    "assistant": _types.AssistantMessage,
    "tool_result": _types.ToolResultMessage,
}

_EVENTS: dict[str, type] = {
    "start": _events.StreamStart,
    "text_start": _events.TextStart,
    "text_delta": _events.TextDelta,
    "text_end": _events.TextEnd,
    "thinking_start": _events.ThinkingStart,
    "thinking_delta": _events.ThinkingDelta,
    "thinking_end": _events.ThinkingEnd,
    "toolcall_start": _events.ToolCallStart,
    "toolcall_delta": _events.ToolCallDelta,
    "toolcall_end": _events.ToolCallEnd,
    "done": _events.StreamDone,
    "error": _events.StreamError,
}


def to_dict(obj: Any) -> dict[str, Any]:
    """Convert a protocol dataclass to a JSON-safe dict.

    Raises TypeError for non-dataclasses, which is how the round-trip test
    catches a live object sneaking onto an event.
    """
    if not is_dataclass(obj) or isinstance(obj, type):
        raise TypeError(f"Not a protocol dataclass: {type(obj).__name__}")
    return asdict(obj)


def block_from_dict(data: dict[str, Any]) -> _types.ContentBlock:
    payload = dict(data)
    tag = payload.pop("type")
    cls = _BLOCKS.get(tag)
    if cls is None:
        raise ValueError(f"Unknown content block type: {tag!r}")
    return cls(**payload)


def message_from_dict(data: dict[str, Any]) -> _types.Message:
    payload = dict(data)
    tag = payload.pop("role")
    cls = _MESSAGES.get(tag)
    if cls is None:
        raise ValueError(f"Unknown message role: {tag!r}")
    if isinstance(payload.get("content"), list):
        payload["content"] = tuple(block_from_dict(b) for b in payload["content"])
    if "usage" in payload and isinstance(payload["usage"], dict):
        payload["usage"] = _types.Usage(**payload["usage"])
    return cls(**payload)


def event_from_dict(data: dict[str, Any]) -> _events.StreamEvent:
    payload = dict(data)
    tag = payload.pop("type")
    cls = _EVENTS.get(tag)
    if cls is None:
        raise ValueError(f"Unknown stream event type: {tag!r}")
    if isinstance(payload.get("message"), dict):
        payload["message"] = message_from_dict(payload["message"])
    if isinstance(payload.get("tool_call"), dict):
        payload["tool_call"] = block_from_dict(payload["tool_call"])
    return cls(**payload)
```

Note: `Usage.total` is a property, not a field, so `asdict` omits it and the
round-trip reconstructs it. That is intended — derived values are never on the
wire.

- [ ] **Step 9: Run the test to verify it passes**

Run: `uv run pytest tests/test_serialization.py -v`
Expected: PASS — 22 parametrized cases plus the coverage guard.

If `test_message_roundtrip` fails on the `UserMessage` case with a tuple/list
mismatch, that is the expected shape of the bug: JSON has no tuples, so
`message_from_dict` must rebuild them. The implementation above does.

- [ ] **Step 10: Write the config schema**

Create `harness/config.py`:

```python
"""Configuration schema and defaults.

This module is the single list of every tunable in the system. Code reads
values from a resolved `Config`; it never hardcodes them. See
docs/contributing.md.

Config file loading (TOML, layered, with config.local.toml) arrives with the
CLI milestone. For now `load_config()` returns defaults, so the loop has a real
config object to depend on from the start.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class AgentConfig:
    #: Hard ceiling on provider requests in one run. Guards against a model
    #: that calls tools forever.
    max_turns: int = 50
    #: Run independent tool calls from one message concurrently.
    parallel_tools: bool = True


@dataclass(frozen=True, slots=True)
class Config:
    agent: AgentConfig = field(default_factory=AgentConfig)


def load_config() -> Config:
    """Resolve configuration. Currently defaults only."""
    return Config()
```

- [ ] **Step 11: Run all checks**

```bash
uv run ruff check . && uv run ruff format .
uv run pyright
uv run pytest -v
```

Expected: ruff clean, pyright clean, all tests pass.

If pyright flags the `cls(**payload)` calls in `serde.py` as untyped, that is
expected and acceptable — dynamic construction from a tag map cannot be fully
typed without a large amount of overload boilerplate. Leave it.

- [ ] **Step 12: Update docs and propose a commit**

Update `docs/events.md`: replace the "Not yet implemented" banner with a note
that the stream protocol in `harness/llm/events.py` is implemented, and that
agent-level events arrive in Task 3.

Update the build-order table in `docs/architecture.md` and the checklist in
`README.md`: mark step 1 complete.

Proposed commit — **run only if the user asks**:

```bash
git add pyproject.toml uv.lock .python-version harness/ tests/ docs/ README.md
git commit -m "feat: core LLM types, stream protocol, and serialization"
```

---

## Task 2: Provider protocol and FakeProvider

The seam every later test depends on. `FakeProvider` replays scripted events so the loop can be tested without a network.

**Files:**
- Create: `harness/llm/provider.py`, `harness/llm/fake.py`
- Create: `tests/test_fake_provider.py`

**Interfaces:**
- Consumes: `Model`, `Context`, `AssistantMessage`, `Usage`, `StreamEvent` and all event classes from Task 1.
- Produces:
  - `StreamOptions(max_tokens, temperature, reasoning, signal)`
  - `Provider` protocol with `async def stream(model, ctx, opts) -> AsyncIterator[StreamEvent]`
  - `FakeProvider(scripts)` where `scripts: list[list[StreamEvent]]` — one list per expected request, consumed in order
  - `FakeProvider.requests: list[Context]` — what the loop actually sent, for assertions
  - `text_response(text, *, model, provider) -> list[StreamEvent]` — builds a complete scripted text turn
  - `tool_call_response(calls, *, model, provider) -> list[StreamEvent]` — builds a scripted tool-calling turn
  - `error_response(message, *, model, provider) -> list[StreamEvent]`
  - `FAKE_MODEL: Model`

- [ ] **Step 1: Write the provider protocol**

Create `harness/llm/provider.py`:

```python
"""The provider contract.

A provider MUST NOT raise for request, model, or runtime failures. It
terminates the stream with a StreamError carrying an AssistantMessage whose
stop_reason is "error" or "aborted". This is why the agent loop has almost no
try/except around streaming.

Programming errors (a bad argument type, a bug in the adapter) may still raise
— the rule covers failures that are part of normal operation.
"""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from harness.llm.events import StreamEvent
from harness.llm.types import Context, Model


@dataclass(frozen=True, slots=True)
class StreamOptions:
    max_tokens: int | None = None
    temperature: float | None = None
    #: Reasoning effort, for models that support it.
    reasoning: str | None = None
    #: Cooperative cancellation. Providers should stop and emit a
    #: StreamError(reason="aborted") when this is set.
    signal: asyncio.Event | None = None


class Provider(Protocol):
    """Streams assistant responses. See the module docstring for the contract."""

    name: str

    def stream(
        self,
        model: Model,
        ctx: Context,
        opts: StreamOptions | None = None,
    ) -> AsyncIterator[StreamEvent]: ...
```

- [ ] **Step 2: Write the failing FakeProvider test**

Create `tests/test_fake_provider.py`:

```python
import pytest

from harness.llm.events import StreamDone, StreamStart, TextDelta
from harness.llm.fake import (
    FAKE_MODEL,
    FakeProvider,
    error_response,
    text_response,
    tool_call_response,
)
from harness.llm.types import Context, ToolCallBlock, text


async def _drain(provider: FakeProvider, ctx: Context) -> list[object]:
    return [event async for event in provider.stream(FAKE_MODEL, ctx)]


def _ctx(prompt: str = "sys") -> Context:
    return Context(system_prompt=prompt, messages=())


async def test_replays_scripted_text_response() -> None:
    provider = FakeProvider([text_response("hello")])
    events = await _drain(provider, _ctx())

    assert isinstance(events[0], StreamStart)
    assert isinstance(events[-1], StreamDone)
    assert events[-1].message.content[0].text == "hello"
    assert events[-1].reason == "stop"


async def test_streams_text_in_deltas() -> None:
    provider = FakeProvider([text_response("abc", chunk_size=1)])
    events = await _drain(provider, _ctx())

    deltas = [e.delta for e in events if isinstance(e, TextDelta)]
    assert deltas == ["a", "b", "c"]


async def test_records_requests_for_assertions() -> None:
    provider = FakeProvider([text_response("one"), text_response("two")])
    await _drain(provider, _ctx("first"))
    await _drain(provider, _ctx("second"))

    assert [c.system_prompt for c in provider.requests] == ["first", "second"]


async def test_tool_call_response_sets_tool_use_stop_reason() -> None:
    call = ToolCallBlock(id="c1", name="read", arguments={"path": "a.py"})
    provider = FakeProvider([tool_call_response([call])])
    events = await _drain(provider, _ctx())

    done = events[-1]
    assert done.reason == "tool_use"
    assert done.message.tool_calls == (call,)


async def test_error_response_terminates_with_stream_error() -> None:
    provider = FakeProvider([error_response("rate limited")])
    events = await _drain(provider, _ctx())

    assert events[-1].type == "error"
    assert events[-1].message.stop_reason == "error"
    assert events[-1].message.error_message == "rate limited"


async def test_raises_when_script_is_exhausted() -> None:
    """A loop making more requests than expected is a bug worth failing on."""
    provider = FakeProvider([text_response("only one")])
    await _drain(provider, _ctx())

    with pytest.raises(AssertionError, match="exhausted"):
        await _drain(provider, _ctx())
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_fake_provider.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'harness.llm.fake'`

- [ ] **Step 4: Implement FakeProvider**

Create `harness/llm/fake.py`:

```python
"""A provider that replays scripted events. The backbone of the offline suite.

Build scripts with the `*_response` helpers rather than by hand — they produce
protocol-correct event sequences, so a test that passes against the fake is
testing the loop, not a malformed script.
"""

from collections.abc import AsyncIterator, Sequence
from itertools import count

from harness.llm.events import (
    StreamDone,
    StreamError,
    StreamEvent,
    StreamStart,
    TextDelta,
    TextEnd,
    TextStart,
    ToolCallEnd,
    ToolCallStart,
)
from harness.llm.provider import StreamOptions
from harness.llm.types import (
    AssistantMessage,
    Context,
    Model,
    TextBlock,
    ToolCallBlock,
    Usage,
)

FAKE_MODEL = Model(
    id="fake-model",
    name="Fake Model",
    provider="fake",
    context_window=200_000,
    max_tokens=8_192,
)

_DEFAULT_USAGE = Usage(input=10, output=5)


def text_response(
    value: str,
    *,
    model: str = FAKE_MODEL.id,
    provider: str = "fake",
    chunk_size: int = 0,
) -> list[StreamEvent]:
    """A complete turn that returns `value` as text.

    chunk_size=0 emits one delta; larger values split the text, which is how
    streaming-render tests get multiple updates.
    """
    chunks = (
        [value]
        if chunk_size <= 0
        else [value[i : i + chunk_size] for i in range(0, len(value), chunk_size)]
    )
    message = AssistantMessage(
        content=(TextBlock(text=value),),
        model=model,
        provider=provider,
        usage=_DEFAULT_USAGE,
        stop_reason="stop",
    )
    return [
        StreamStart(),
        TextStart(block_index=0),
        *(TextDelta(block_index=0, delta=c) for c in chunks),
        TextEnd(block_index=0, text=value),
        StreamDone(message=message, reason="stop"),
    ]


def tool_call_response(
    calls: Sequence[ToolCallBlock],
    *,
    preamble: str | None = None,
    model: str = FAKE_MODEL.id,
    provider: str = "fake",
) -> list[StreamEvent]:
    """A turn that requests one or more tool calls, optionally with leading text."""
    events: list[StreamEvent] = [StreamStart()]
    content: list[TextBlock | ToolCallBlock] = []
    index = count()

    if preamble is not None:
        i = next(index)
        events += [
            TextStart(block_index=i),
            TextDelta(block_index=i, delta=preamble),
            TextEnd(block_index=i, text=preamble),
        ]
        content.append(TextBlock(text=preamble))

    for call in calls:
        i = next(index)
        events += [ToolCallStart(block_index=i), ToolCallEnd(block_index=i, tool_call=call)]
        content.append(call)

    message = AssistantMessage(
        content=tuple(content),
        model=model,
        provider=provider,
        usage=_DEFAULT_USAGE,
        stop_reason="tool_use",
    )
    events.append(StreamDone(message=message, reason="tool_use"))
    return events


def error_response(
    error: str,
    *,
    model: str = FAKE_MODEL.id,
    provider: str = "fake",
    reason: str = "error",
) -> list[StreamEvent]:
    """A turn that fails. Note it terminates with StreamError, never an exception."""
    message = AssistantMessage(
        content=(),
        model=model,
        provider=provider,
        stop_reason="error" if reason == "error" else "aborted",
        error_message=error,
    )
    return [StreamError(message=message, reason=reason)]  # type: ignore[arg-type]


def truncated_tool_call_response(
    calls: Sequence[ToolCallBlock],
    *,
    model: str = FAKE_MODEL.id,
    provider: str = "fake",
) -> list[StreamEvent]:
    """Tool calls cut off by the output token limit.

    Arguments may parse and validate while being silently incomplete, so the
    loop must refuse to execute any of them.
    """
    events = tool_call_response(calls, model=model, provider=provider)
    done = events[-1]
    assert isinstance(done, StreamDone)
    message = AssistantMessage(
        content=done.message.content,
        model=model,
        provider=provider,
        usage=done.message.usage,
        stop_reason="length",
    )
    events[-1] = StreamDone(message=message, reason="length")
    return events


class FakeProvider:
    """Replays one script per request, in order."""

    name = "fake"

    def __init__(self, scripts: Sequence[Sequence[StreamEvent]]) -> None:
        self._scripts = [list(s) for s in scripts]
        self._index = 0
        #: Contexts the caller sent, in order. Assert against this.
        self.requests: list[Context] = []

    async def stream(
        self,
        model: Model,
        ctx: Context,
        opts: StreamOptions | None = None,
    ) -> AsyncIterator[StreamEvent]:
        if self._index >= len(self._scripts):
            raise AssertionError(
                f"FakeProvider script exhausted: request {self._index + 1} was made "
                f"but only {len(self._scripts)} script(s) were provided."
            )
        script = self._scripts[self._index]
        self._index += 1
        self.requests.append(ctx)

        for event in script:
            if opts is not None and opts.signal is not None and opts.signal.is_set():
                yield StreamError(
                    message=AssistantMessage(
                        content=(),
                        model=model.id,
                        provider=self.name,
                        stop_reason="aborted",
                        error_message="Operation aborted",
                    ),
                    reason="aborted",
                )
                return
            yield event
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_fake_provider.py -v`
Expected: 6 passed.

- [ ] **Step 6: Run all checks and propose a commit**

```bash
uv run ruff check . && uv run ruff format .
uv run pyright
uv run pytest -v
```

Update `docs/providers.md`: note that the `Provider` protocol and `FakeProvider`
exist, and that real adapters arrive in a later milestone.

Proposed commit — **run only if the user asks**:

```bash
git add harness/llm/ tests/ docs/
git commit -m "feat: provider protocol and FakeProvider"
```

---

## Task 3: The agent loop

A pure async function: stream a response, execute tool calls, repeat until the model stops. No state, no persistence, no UI.

**Files:**
- Create: `harness/agent/types.py`, `harness/agent/loop.py`
- Create: `tests/test_agent_loop.py`

**Interfaces:**
- Consumes: everything from Tasks 1 and 2.
- Produces:
  - `AgentEvent` union: `AgentStart`, `AgentEnd`, `TurnStart`, `TurnEnd`, `MessageStart`, `MessageUpdate`, `MessageEnd`, `ToolExecutionStart`, `ToolExecutionUpdate`, `ToolExecutionEnd`
  - `ToolResult(content, is_error, details)`
  - `AgentTool` protocol: `name`, `description`, `parameters`, `execution_mode`, `async def execute(call_id, args, ctx) -> ToolResult`
  - `AgentContext(system_prompt, messages, tools)`
  - `BeforeToolCallResult(block, reason)`, `AfterToolCallResult(content, is_error)`
  - `LoopConfig(model, provider, config, before_tool_call, after_tool_call, prepare_request, signal)`
  - `async def run_agent_loop(ctx, cfg, emit) -> list[Message]`

- [ ] **Step 1: Write the agent types**

Create `harness/agent/types.py`:

```python
"""Agent-level events, tool contracts, and hook signatures.

Every AgentEvent is a frozen dataclass of JSON primitives — the same rule as
the stream protocol, for the same reason: cli/ consumes these, and a second
consumer in another language must be possible.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from harness.config import Config
from harness.llm.provider import Provider
from harness.llm.types import (
    AssistantMessage,
    ContentBlock,
    Message,
    Model,
    ToolCallBlock,
    ToolDeclaration,
    ToolResultMessage,
)

# --- tools ----------------------------------------------------------------

ExecutionMode = Literal["parallel", "sequential"]


@dataclass(frozen=True, slots=True)
class ToolResult:
    content: tuple[ContentBlock, ...]
    is_error: bool = False
    #: Structured data for renderers. Never sent to the model.
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolContext:
    """What a tool receives. Tools never reach for globals."""

    cwd: str
    call_id: str
    signal: asyncio.Event | None = None
    on_update: Callable[[ToolResult], None] | None = None


class AgentTool(Protocol):
    name: str
    description: str
    parameters: dict[str, object]
    execution_mode: ExecutionMode

    async def execute(self, args: dict[str, object], ctx: ToolContext) -> ToolResult: ...


# --- events ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AgentStart:
    type: Literal["agent_start"] = "agent_start"


@dataclass(frozen=True, slots=True)
class AgentEnd:
    messages: tuple[Message, ...]
    type: Literal["agent_end"] = "agent_end"


@dataclass(frozen=True, slots=True)
class TurnStart:
    type: Literal["turn_start"] = "turn_start"


@dataclass(frozen=True, slots=True)
class TurnEnd:
    message: AssistantMessage
    tool_results: tuple[ToolResultMessage, ...]
    type: Literal["turn_end"] = "turn_end"


@dataclass(frozen=True, slots=True)
class MessageStart:
    message: Message
    type: Literal["message_start"] = "message_start"


@dataclass(frozen=True, slots=True)
class MessageUpdate:
    """A streaming delta. Carries the delta, not a full snapshot — consumers
    needing the whole in-flight message read Agent.state.streaming_message."""

    block_index: int
    delta: str
    kind: Literal["text", "thinking"] = "text"
    type: Literal["message_update"] = "message_update"


@dataclass(frozen=True, slots=True)
class MessageEnd:
    message: Message
    type: Literal["message_end"] = "message_end"


@dataclass(frozen=True, slots=True)
class ToolExecutionStart:
    tool_call_id: str
    tool_name: str
    arguments: dict[str, object]
    type: Literal["tool_execution_start"] = "tool_execution_start"


@dataclass(frozen=True, slots=True)
class ToolExecutionUpdate:
    tool_call_id: str
    tool_name: str
    content: tuple[ContentBlock, ...]
    type: Literal["tool_execution_update"] = "tool_execution_update"


@dataclass(frozen=True, slots=True)
class ToolExecutionEnd:
    tool_call_id: str
    tool_name: str
    content: tuple[ContentBlock, ...]
    is_error: bool
    duration_ms: int
    type: Literal["tool_execution_end"] = "tool_execution_end"


AgentEvent = (
    AgentStart
    | AgentEnd
    | TurnStart
    | TurnEnd
    | MessageStart
    | MessageUpdate
    | MessageEnd
    | ToolExecutionStart
    | ToolExecutionUpdate
    | ToolExecutionEnd
)

EventSink = Callable[[AgentEvent], Awaitable[None]]

# --- hooks ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BeforeToolCallContext:
    tool_call: ToolCallBlock
    messages: tuple[Message, ...]


@dataclass(frozen=True, slots=True)
class BeforeToolCallResult:
    block: bool = False
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class AfterToolCallContext:
    tool_call: ToolCallBlock
    result: ToolResult


@dataclass(frozen=True, slots=True)
class AfterToolCallResult:
    content: tuple[ContentBlock, ...] | None = None
    is_error: bool | None = None


@dataclass(frozen=True, slots=True)
class PrepareRequestContext:
    messages: tuple[Message, ...]
    model: Model


@dataclass(frozen=True, slots=True)
class RequestUpdate:
    """Replacement state for the request being prepared. This is the seam
    context compaction will use."""

    messages: tuple[Message, ...] | None = None
    model: Model | None = None


# --- loop configuration ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class AgentContext:
    system_prompt: str
    messages: tuple[Message, ...]
    tools: tuple[AgentTool, ...] = ()


@dataclass(frozen=True, slots=True)
class LoopConfig:
    model: Model
    provider: Provider
    config: Config
    cwd: str = "."
    before_tool_call: (
        Callable[[BeforeToolCallContext], Awaitable[BeforeToolCallResult | None]] | None
    ) = None
    after_tool_call: (
        Callable[[AfterToolCallContext], Awaitable[AfterToolCallResult | None]] | None
    ) = None
    prepare_request: Callable[[PrepareRequestContext], Awaitable[RequestUpdate | None]] | None = (
        None
    )
    signal: asyncio.Event | None = None


def declarations_for(tools: tuple[AgentTool, ...]) -> tuple[ToolDeclaration, ...]:
    """Project executable tools into what the model is told about them."""
    return tuple(
        ToolDeclaration(name=t.name, description=t.description, parameters=t.parameters)
        for t in tools
    )
```

- [ ] **Step 2: Write the failing loop tests**

Create `tests/test_agent_loop.py`:

```python
import asyncio

import pytest

from harness.agent.loop import run_agent_loop
from harness.agent.types import (
    AgentContext,
    AgentEnd,
    AgentEvent,
    AgentStart,
    BeforeToolCallResult,
    LoopConfig,
    MessageUpdate,
    ToolContext,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolResult,
    TurnEnd,
    TurnStart,
)
from harness.config import Config
from harness.llm.fake import (
    FAKE_MODEL,
    FakeProvider,
    error_response,
    text_response,
    tool_call_response,
    truncated_tool_call_response,
)
from harness.llm.types import (
    AssistantMessage,
    ToolCallBlock,
    ToolResultMessage,
    UserMessage,
    text,
)


class RecordingTool:
    """A tool that records its calls and returns a fixed result."""

    def __init__(
        self,
        name: str = "echo",
        *,
        result: str = "ok",
        is_error: bool = False,
        delay: float = 0.0,
        execution_mode: str = "parallel",
    ) -> None:
        self.name = name
        self.description = f"{name} tool"
        self.parameters = {"type": "object", "properties": {}}
        self.execution_mode = execution_mode
        self.calls: list[dict[str, object]] = []
        self._result = result
        self._is_error = is_error
        self._delay = delay

    async def execute(self, args: dict[str, object], ctx: ToolContext) -> ToolResult:
        self.calls.append(args)
        if self._delay:
            await asyncio.sleep(self._delay)
        return ToolResult(content=text(self._result), is_error=self._is_error)


class RaisingTool:
    name = "boom"
    description = "always raises"
    parameters: dict[str, object] = {"type": "object", "properties": {}}
    execution_mode = "parallel"

    async def execute(self, args: dict[str, object], ctx: ToolContext) -> ToolResult:
        raise RuntimeError("tool exploded")


def _collector() -> tuple[list[AgentEvent], object]:
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    return events, emit


def _cfg(provider: FakeProvider, **kwargs: object) -> LoopConfig:
    return LoopConfig(model=FAKE_MODEL, provider=provider, config=Config(), **kwargs)


def _ctx(*tools: object) -> AgentContext:
    return AgentContext(
        system_prompt="sys",
        messages=(UserMessage(content=text("hi")),),
        tools=tuple(tools),
    )


async def test_simple_text_turn_emits_lifecycle_events() -> None:
    provider = FakeProvider([text_response("hello")])
    events, emit = _collector()

    messages = await run_agent_loop(_ctx(), _cfg(provider), emit)

    kinds = [e.type for e in events]
    assert kinds[0] == "agent_start"
    assert kinds[-1] == "agent_end"
    assert "turn_start" in kinds and "turn_end" in kinds
    assert len(messages) == 1
    assert isinstance(messages[0], AssistantMessage)
    assert messages[0].content[0].text == "hello"


async def test_streaming_deltas_become_message_updates() -> None:
    provider = FakeProvider([text_response("abc", chunk_size=1)])
    events, emit = _collector()

    await run_agent_loop(_ctx(), _cfg(provider), emit)

    deltas = [e.delta for e in events if isinstance(e, MessageUpdate)]
    assert deltas == ["a", "b", "c"]


async def test_executes_tool_call_then_continues() -> None:
    tool = RecordingTool()
    call = ToolCallBlock(id="c1", name="echo", arguments={"v": 1})
    provider = FakeProvider([tool_call_response([call]), text_response("done")])
    events, emit = _collector()

    messages = await run_agent_loop(_ctx(tool), _cfg(provider), emit)

    assert tool.calls == [{"v": 1}]
    roles = [m.role for m in messages]
    assert roles == ["assistant", "tool_result", "assistant"]
    assert any(isinstance(e, ToolExecutionStart) for e in events)
    assert any(isinstance(e, ToolExecutionEnd) for e in events)


async def test_tool_result_is_sent_back_to_the_model() -> None:
    tool = RecordingTool(result="42")
    call = ToolCallBlock(id="c1", name="echo", arguments={})
    provider = FakeProvider([tool_call_response([call]), text_response("done")])
    _, emit = _collector()

    await run_agent_loop(_ctx(tool), _cfg(provider), emit)

    second_request = provider.requests[1]
    tool_results = [m for m in second_request.messages if isinstance(m, ToolResultMessage)]
    assert len(tool_results) == 1
    assert tool_results[0].content[0].text == "42"


async def test_unknown_tool_yields_error_result_without_crashing() -> None:
    call = ToolCallBlock(id="c1", name="nonexistent", arguments={})
    provider = FakeProvider([tool_call_response([call]), text_response("recovered")])
    _, emit = _collector()

    messages = await run_agent_loop(_ctx(), _cfg(provider), emit)

    result = next(m for m in messages if isinstance(m, ToolResultMessage))
    assert result.is_error
    assert "not found" in result.content[0].text.lower()


async def test_raising_tool_becomes_error_result() -> None:
    call = ToolCallBlock(id="c1", name="boom", arguments={})
    provider = FakeProvider([tool_call_response([call]), text_response("recovered")])
    _, emit = _collector()

    messages = await run_agent_loop(_ctx(RaisingTool()), _cfg(provider), emit)

    result = next(m for m in messages if isinstance(m, ToolResultMessage))
    assert result.is_error
    assert "tool exploded" in result.content[0].text


async def test_provider_error_ends_the_run() -> None:
    provider = FakeProvider([error_response("rate limited")])
    events, emit = _collector()

    messages = await run_agent_loop(_ctx(), _cfg(provider), emit)

    assert messages[-1].stop_reason == "error"
    assert messages[-1].error_message == "rate limited"
    assert isinstance(events[-1], AgentEnd)


async def test_truncated_response_fails_all_tool_calls_unexecuted() -> None:
    """stop_reason='length' means arguments may be silently incomplete."""
    tool = RecordingTool()
    calls = [
        ToolCallBlock(id="c1", name="echo", arguments={"v": 1}),
        ToolCallBlock(id="c2", name="echo", arguments={"v": 2}),
    ]
    provider = FakeProvider([truncated_tool_call_response(calls), text_response("retry")])
    _, emit = _collector()

    messages = await run_agent_loop(_ctx(tool), _cfg(provider), emit)

    assert tool.calls == []
    results = [m for m in messages if isinstance(m, ToolResultMessage)]
    assert len(results) == 2
    assert all(r.is_error for r in results)
    assert all("token limit" in r.content[0].text for r in results)


async def test_before_tool_call_can_block() -> None:
    tool = RecordingTool()
    call = ToolCallBlock(id="c1", name="echo", arguments={})
    provider = FakeProvider([tool_call_response([call]), text_response("ok")])
    _, emit = _collector()

    async def deny(ctx: object) -> BeforeToolCallResult:
        return BeforeToolCallResult(block=True, reason="denied by policy")

    messages = await run_agent_loop(_ctx(tool), _cfg(provider, before_tool_call=deny), emit)

    assert tool.calls == []
    result = next(m for m in messages if isinstance(m, ToolResultMessage))
    assert result.is_error
    assert result.content[0].text == "denied by policy"


async def test_parallel_tools_run_concurrently_but_results_keep_source_order() -> None:
    """tool_execution_end fires in completion order; result messages do not."""
    slow = RecordingTool(name="slow", result="slow-done", delay=0.05)
    fast = RecordingTool(name="fast", result="fast-done", delay=0.0)
    calls = [
        ToolCallBlock(id="c1", name="slow", arguments={}),
        ToolCallBlock(id="c2", name="fast", arguments={}),
    ]
    provider = FakeProvider([tool_call_response(calls), text_response("ok")])
    events, emit = _collector()

    messages = await run_agent_loop(_ctx(slow, fast), _cfg(provider), emit)

    completion = [e.tool_call_id for e in events if isinstance(e, ToolExecutionEnd)]
    assert completion == ["c2", "c1"], "fast tool should finish first"

    results = [m.tool_call_id for m in messages if isinstance(m, ToolResultMessage)]
    assert results == ["c1", "c2"], "result messages must follow assistant source order"


async def test_sequential_tool_forces_serial_execution() -> None:
    order: list[str] = []

    class OrderedTool:
        def __init__(self, name: str, delay: float) -> None:
            self.name = name
            self.description = name
            self.parameters: dict[str, object] = {"type": "object", "properties": {}}
            self.execution_mode = "sequential"
            self._delay = delay

        async def execute(self, args: dict[str, object], ctx: ToolContext) -> ToolResult:
            order.append(f"{self.name}-start")
            await asyncio.sleep(self._delay)
            order.append(f"{self.name}-end")
            return ToolResult(content=text("ok"))

    calls = [
        ToolCallBlock(id="c1", name="first", arguments={}),
        ToolCallBlock(id="c2", name="second", arguments={}),
    ]
    provider = FakeProvider([tool_call_response(calls), text_response("ok")])
    _, emit = _collector()

    await run_agent_loop(
        _ctx(OrderedTool("first", 0.03), OrderedTool("second", 0.0)),
        _cfg(provider),
        emit,
    )

    assert order == ["first-start", "first-end", "second-start", "second-end"]


async def test_prepare_request_can_replace_messages() -> None:
    """The seam compaction will use."""
    provider = FakeProvider([text_response("ok")])
    _, emit = _collector()
    replacement = (UserMessage(content=text("replaced")),)

    async def replace(ctx: object) -> object:
        from harness.agent.types import RequestUpdate

        return RequestUpdate(messages=replacement)

    await run_agent_loop(_ctx(), _cfg(provider, prepare_request=replace), emit)

    assert provider.requests[0].messages == replacement


async def test_max_turns_stops_a_runaway_loop() -> None:
    tool = RecordingTool()
    call = ToolCallBlock(id="c1", name="echo", arguments={})
    # Always asks for another tool call: without a cap this never terminates.
    provider = FakeProvider([tool_call_response([call]) for _ in range(5)])
    _, emit = _collector()

    from dataclasses import replace as dc_replace

    from harness.config import AgentConfig

    cfg = LoopConfig(
        model=FAKE_MODEL,
        provider=provider,
        config=dc_replace(Config(), agent=AgentConfig(max_turns=3)),
    )

    messages = await run_agent_loop(_ctx(tool), cfg, emit)

    assistant_count = sum(1 for m in messages if isinstance(m, AssistantMessage))
    assert assistant_count == 3


async def test_abort_signal_stops_the_run() -> None:
    signal = asyncio.Event()
    signal.set()
    provider = FakeProvider([text_response("never seen")])
    events, emit = _collector()

    messages = await run_agent_loop(_ctx(), _cfg(provider, signal=signal), emit)

    assert messages[-1].stop_reason == "aborted"
    assert isinstance(events[-1], AgentEnd)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_agent_loop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'harness.agent.loop'`

- [ ] **Step 4: Implement the loop**

Create `harness/agent/loop.py` with this module docstring and imports:

```python
"""The agentic loop: stream a response, run its tool calls, repeat.

A pure async function. It owns no state — the caller passes a context and gets
back the messages the run produced. `Agent` wraps this with transcript state.

Error handling is concentrated at two boundaries, so the body stays linear:
providers terminate with an error event rather than raising, and tool failures
become error results rather than propagating.
"""

import asyncio
import time
from collections.abc import Sequence

from harness.agent.types import (
    AfterToolCallContext,
    AgentContext,
    AgentEnd,
    AgentEvent,
    AgentStart,
    AgentTool,
    BeforeToolCallContext,
    EventSink,
    LoopConfig,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    PrepareRequestContext,
    ToolContext,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolExecutionUpdate,
    ToolResult,
    TurnEnd,
    TurnStart,
    declarations_for,
)
from harness.llm.events import (
    StreamDone,
    StreamError,
    TextDelta,
    ThinkingDelta,
)
from harness.llm.provider import StreamOptions
from harness.llm.types import (
    AssistantMessage,
    Context,
    Message,
    TextBlock,
    ToolCallBlock,
    ToolResultMessage,
    text,
)

_TRUNCATED_NOTICE = (
    "Tool call {name!r} was not executed: the response hit the output token "
    "limit, so its arguments may be truncated. Re-issue the call with complete "
    "arguments."
)


async def run_agent_loop(
    ctx: AgentContext,
    cfg: LoopConfig,
    emit: EventSink,
) -> list[Message]:
    """Run until the model stops requesting tools, or a limit is hit.

    Returns the messages produced by this run — not including the ones that
    were already in `ctx.messages`.
    """
    transcript: list[Message] = list(ctx.messages)
    produced: list[Message] = []
    tools_by_name = {t.name: t for t in ctx.tools}
    declarations = declarations_for(ctx.tools)

    await emit(AgentStart())

    for _turn in range(cfg.config.agent.max_turns):
        if _aborted(cfg):
            produced.append(_aborted_message(cfg))
            break

        await emit(TurnStart())

        messages = tuple(transcript)
        model = cfg.model
        if cfg.prepare_request is not None:
            update = await cfg.prepare_request(
                PrepareRequestContext(messages=messages, model=model)
            )
            if update is not None:
                messages = update.messages if update.messages is not None else messages
                model = update.model if update.model is not None else model

        request = Context(system_prompt=ctx.system_prompt, messages=messages, tools=declarations)
        assistant = await _stream_turn(request, model, cfg, emit)

        transcript.append(assistant)
        produced.append(assistant)

        if assistant.stop_reason in ("error", "aborted"):
            await emit(TurnEnd(message=assistant, tool_results=()))
            break

        calls = assistant.tool_calls
        if not calls:
            await emit(TurnEnd(message=assistant, tool_results=()))
            break

        results = await _execute_tool_calls(
            calls,
            tools_by_name,
            cfg,
            emit,
            transcript=tuple(transcript),
            truncated=assistant.stop_reason == "length",
        )
        for result in results:
            await emit(MessageStart(message=result))
            await emit(MessageEnd(message=result))
            transcript.append(result)
            produced.append(result)

        await emit(TurnEnd(message=assistant, tool_results=tuple(results)))

    await emit(AgentEnd(messages=tuple(produced)))
    return produced


async def _stream_turn(
    request: Context,
    model: object,
    cfg: LoopConfig,
    emit: EventSink,
) -> AssistantMessage:
    """Consume one provider stream into a final AssistantMessage.

    The provider contract says failures arrive as a StreamError event, so there
    is no try/except here.
    """
    opts = StreamOptions(signal=cfg.signal)
    final: AssistantMessage | None = None

    # A stream may terminate with StreamError as its FIRST event when request
    # setup fails before generation starts, so nothing here may assume a
    # preceding StreamStart.
    async for event in cfg.provider.stream(model, request, opts):  # type: ignore[arg-type]
        if isinstance(event, TextDelta):
            await emit(MessageUpdate(block_index=event.block_index, delta=event.delta, kind="text"))
        elif isinstance(event, ThinkingDelta):
            await emit(
                MessageUpdate(block_index=event.block_index, delta=event.delta, kind="thinking")
            )
        elif isinstance(event, StreamDone):
            final = event.message
        elif isinstance(event, StreamError):
            final = event.message

    if final is None:
        # A provider that ends without done/error violates the contract; surface
        # it as an error turn rather than hanging the loop.
        final = AssistantMessage(
            content=(),
            model=cfg.model.id,
            provider=cfg.provider.name,
            stop_reason="error",
            error_message="Provider stream ended without a terminal event",
        )

    await emit(MessageStart(message=final))
    await emit(MessageEnd(message=final))
    return final


async def _execute_tool_calls(
    calls: Sequence[ToolCallBlock],
    tools_by_name: dict[str, AgentTool],
    cfg: LoopConfig,
    emit: EventSink,
    *,
    transcript: tuple[Message, ...],
    truncated: bool,
) -> list[ToolResultMessage]:
    """Run a batch of tool calls.

    Two orderings, deliberately different:
      - tool_execution_end fires in completion order, for live feedback
      - the returned result messages follow assistant source order, because
        providers require results to match the calls that requested them
    """
    if truncated:
        truncated_results: list[ToolResultMessage] = []
        for call in calls:
            await emit(
                ToolExecutionStart(
                    tool_call_id=call.id, tool_name=call.name, arguments=call.arguments
                )
            )
            final = await _finish(
                call,
                ToolResult(content=text(_TRUNCATED_NOTICE.format(name=call.name)), is_error=True),
                0,
                cfg,
                emit,
            )
            truncated_results.append(
                ToolResultMessage(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    content=final.content,
                    is_error=final.is_error,
                    duration_ms=0,
                )
            )
        return truncated_results

    prepared: list[tuple[ToolCallBlock, AgentTool | None, ToolResult | None]] = []
    for call in calls:
        await emit(
            ToolExecutionStart(tool_call_id=call.id, tool_name=call.name, arguments=call.arguments)
        )
        tool = tools_by_name.get(call.name)
        if tool is None:
            prepared.append(
                (
                    call,
                    None,
                    ToolResult(content=text(f"Tool {call.name!r} not found"), is_error=True),
                )
            )
            continue
        if cfg.before_tool_call is not None:
            decision = await cfg.before_tool_call(
                BeforeToolCallContext(tool_call=call, messages=transcript)
            )
            if decision is not None and decision.block:
                reason = decision.reason or "Tool execution was blocked"
                prepared.append((call, None, ToolResult(content=text(reason), is_error=True)))
                continue
        prepared.append((call, tool, None))

    run_serially = not cfg.config.agent.parallel_tools or any(
        tool is not None and tool.execution_mode == "sequential" for _, tool, _ in prepared
    )

    async def run_one(call: ToolCallBlock, tool: AgentTool) -> tuple[ToolResult, int]:
        """Execute one tool and emit its end event.

        The emit happens HERE, inside the coroutine, so end events fire in
        completion order. `asyncio.gather` still returns in submission order,
        which is what gives the result messages their source ordering. Moving
        this emit outside would collapse both orderings into one.
        """
        start = time.monotonic()

        def on_update(partial: ToolResult) -> None:
            # Fire-and-forget: tool updates are advisory, and awaiting here would
            # serialize parallel tools.
            asyncio.ensure_future(  # noqa: RUF006
                emit(
                    ToolExecutionUpdate(
                        tool_call_id=call.id, tool_name=call.name, content=partial.content
                    )
                )
            )

        tool_ctx = ToolContext(cwd=cfg.cwd, call_id=call.id, signal=cfg.signal, on_update=on_update)
        try:
            result = await tool.execute(call.arguments, tool_ctx)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — a tool must never kill the loop
            result = ToolResult(content=text(str(exc)), is_error=True)

        duration_ms = int((time.monotonic() - start) * 1000)
        final = await _finish(call, result, duration_ms, cfg, emit)
        return final, duration_ms

    # call id -> (final result, duration). Populated in completion order by the
    # coroutines; read back below in assistant source order.
    outcomes: dict[str, tuple[ToolResult, int]] = {}

    if run_serially:
        for call, tool, _immediate in prepared:
            if tool is None:
                continue
            outcomes[call.id] = await run_one(call, tool)
    else:
        runnable = [(call, tool) for call, tool, _i in prepared if tool is not None]
        if runnable:
            gathered = await asyncio.gather(*(run_one(c, t) for c, t in runnable))
            outcomes = {
                call.id: outcome for (call, _), outcome in zip(runnable, gathered, strict=True)
            }

    results: list[ToolResultMessage] = []
    for call, _tool, immediate in prepared:
        if immediate is not None:
            final, duration = await _finish(call, immediate, 0, cfg, emit), 0
        else:
            final, duration = outcomes[call.id]
        results.append(
            ToolResultMessage(
                tool_call_id=call.id,
                tool_name=call.name,
                content=final.content,
                is_error=final.is_error,
                duration_ms=duration,
            )
        )
    return results


async def _finish(
    call: ToolCallBlock,
    result: ToolResult,
    duration_ms: int,
    cfg: LoopConfig,
    emit: EventSink,
) -> ToolResult:
    """Apply after_tool_call, then emit the end event.

    Returns the final ToolResult; the caller builds the result message so that
    message ordering stays independent of completion ordering.
    """
    if cfg.after_tool_call is not None:
        override = await cfg.after_tool_call(AfterToolCallContext(tool_call=call, result=result))
        if override is not None:
            result = ToolResult(
                content=override.content if override.content is not None else result.content,
                is_error=(override.is_error if override.is_error is not None else result.is_error),
                details=result.details,
            )

    await emit(
        ToolExecutionEnd(
            tool_call_id=call.id,
            tool_name=call.name,
            content=result.content,
            is_error=result.is_error,
            duration_ms=duration_ms,
        )
    )
    return result


def _aborted(cfg: LoopConfig) -> bool:
    return cfg.signal is not None and cfg.signal.is_set()


def _aborted_message(cfg: LoopConfig) -> AssistantMessage:
    return AssistantMessage(
        content=(),
        model=cfg.model.id,
        provider=cfg.provider.name,
        stop_reason="aborted",
        error_message="Operation aborted",
    )
```

**How the two orderings coexist** — the one subtle thing in this module:

`run_one` awaits the tool and then calls `_finish`, which emits
`ToolExecutionEnd`. Because that happens *inside* the coroutine, end events fire
as each tool finishes — completion order. `asyncio.gather` separately returns
its results in *submission* order, and the final loop reads `outcomes` by
walking `prepared`, so result messages follow assistant source order.

Both orderings come free from that arrangement. If you move the emit out of
`run_one` and into the source-order loop, completion order is lost and
`test_parallel_tools_run_concurrently_but_results_keep_source_order` fails.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_agent_loop.py -v`
Expected: 14 passed.

- [ ] **Step 6: Extend the serialization test to cover agent events**

Add to `tests/test_serialization.py`:

```python
from harness.agent.types import (
    AgentEnd,
    AgentStart,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolExecutionUpdate,
    TurnEnd,
    TurnStart,
)

AGENT_EVENTS = [
    AgentStart(),
    AgentEnd(messages=(UserMessage(content=text("hi")),)),
    TurnStart(),
    TurnEnd(message=_ASSISTANT, tool_results=()),
    MessageStart(message=UserMessage(content=text("hi"))),
    MessageUpdate(block_index=0, delta="a", kind="text"),
    MessageEnd(message=_ASSISTANT),
    ToolExecutionStart(tool_call_id="c1", tool_name="read", arguments={"p": "a"}),
    ToolExecutionUpdate(tool_call_id="c1", tool_name="read", content=text("partial")),
    ToolExecutionEnd(
        tool_call_id="c1", tool_name="read", content=text("done"), is_error=False, duration_ms=5
    ),
]


@pytest.mark.parametrize("event", AGENT_EVENTS, ids=lambda e: e.type)
def test_agent_event_is_json_serializable(event: object) -> None:
    """Agent events must be plain data — this is what keeps cli/ swappable."""
    encoded = json.dumps(to_dict(event))
    assert json.loads(encoded)["type"] == event.type


def test_every_agent_event_type_is_covered() -> None:
    from typing import get_args

    from harness.agent import types as agent_types

    union_members = set(get_args(agent_types.AgentEvent))
    assert {type(e) for e in AGENT_EVENTS} == union_members
```

Run: `uv run pytest tests/test_serialization.py -v`
Expected: all pass, including 10 new agent-event cases.

- [ ] **Step 7: Run all checks and propose a commit**

```bash
uv run ruff check . && uv run ruff format .
uv run pyright
uv run pytest -v
```

Update `docs/events.md` with the implemented agent event table, and
`docs/architecture.md` + `README.md` to mark step 3 complete.

Proposed commit — **run only if the user asks**:

```bash
git add harness/agent/ tests/ docs/ README.md
git commit -m "feat: agent loop with tool execution and hooks"
```

---

## Task 4: The stateful Agent

Wraps the loop with a transcript, subscribers, abort, and a message queue — the object a frontend talks to.

**Files:**
- Create: `harness/agent/agent.py`
- Create: `tests/test_agent.py`
- Modify: `harness/agent/__init__.py` (public exports)

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces:
  - `AgentState(system_prompt, model, messages, is_running, streaming_message)`
  - `Agent(model, provider, tools, system_prompt, config, cwd, hooks...)`
  - `Agent.subscribe(listener) -> unsubscribe`
  - `async Agent.prompt(text_or_message) -> list[Message]`
  - `Agent.queue(message)` — inject at the next turn boundary
  - `Agent.abort()`
  - `async Agent.wait_for_idle()`
  - `Agent.state`

- [ ] **Step 1: Write the failing Agent tests**

Create `tests/test_agent.py`:

```python
import asyncio

import pytest

from harness.agent.agent import Agent
from harness.agent.types import AgentEvent, ToolContext, ToolResult
from harness.llm.fake import FAKE_MODEL, FakeProvider, text_response, tool_call_response
from harness.llm.types import AssistantMessage, ToolCallBlock, UserMessage, text


def _agent(provider: FakeProvider, **kwargs: object) -> Agent:
    return Agent(model=FAKE_MODEL, provider=provider, system_prompt="sys", **kwargs)


async def test_prompt_appends_user_and_assistant_messages() -> None:
    agent = _agent(FakeProvider([text_response("hello")]))

    await agent.prompt("hi")

    roles = [m.role for m in agent.state.messages]
    assert roles == ["user", "assistant"]
    assert agent.state.messages[-1].content[0].text == "hello"


async def test_transcript_accumulates_across_prompts() -> None:
    agent = _agent(FakeProvider([text_response("one"), text_response("two")]))

    await agent.prompt("first")
    await agent.prompt("second")

    roles = [m.role for m in agent.state.messages]
    assert roles == ["user", "assistant", "user", "assistant"]


async def test_second_prompt_sees_the_first_exchange() -> None:
    provider = FakeProvider([text_response("one"), text_response("two")])
    agent = _agent(provider)

    await agent.prompt("first")
    await agent.prompt("second")

    assert len(provider.requests[1].messages) == 3


async def test_subscribers_receive_events() -> None:
    received: list[AgentEvent] = []

    async def listener(event: AgentEvent) -> None:
        received.append(event)

    agent = _agent(FakeProvider([text_response("hi")]))
    agent.subscribe(listener)

    await agent.prompt("go")

    assert [e.type for e in received][0] == "agent_start"
    assert [e.type for e in received][-1] == "agent_end"


async def test_unsubscribe_stops_delivery() -> None:
    received: list[AgentEvent] = []

    async def listener(event: AgentEvent) -> None:
        received.append(event)

    agent = _agent(FakeProvider([text_response("a"), text_response("b")]))
    unsubscribe = agent.subscribe(listener)

    await agent.prompt("one")
    count_after_first = len(received)
    unsubscribe()
    await agent.prompt("two")

    assert len(received) == count_after_first


async def test_failing_subscriber_does_not_break_the_run() -> None:
    """A broken renderer must not take down the agent."""

    async def bad_listener(event: AgentEvent) -> None:
        raise RuntimeError("listener bug")

    agent = _agent(FakeProvider([text_response("survived")]))
    agent.subscribe(bad_listener)

    await agent.prompt("go")

    assert agent.state.messages[-1].content[0].text == "survived"


async def test_rejects_concurrent_prompts() -> None:
    agent = _agent(FakeProvider([text_response("a"), text_response("b")]))

    async def slow_listener(event: AgentEvent) -> None:
        await asyncio.sleep(0.01)

    agent.subscribe(slow_listener)
    first = asyncio.create_task(agent.prompt("one"))
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="already running"):
        await agent.prompt("two")

    await first


async def test_queued_message_is_injected_at_the_next_turn() -> None:
    class SlowTool:
        name = "slow"
        description = "slow"
        parameters: dict[str, object] = {"type": "object", "properties": {}}
        execution_mode = "parallel"

        async def execute(self, args: dict[str, object], ctx: ToolContext) -> ToolResult:
            await asyncio.sleep(0.02)
            return ToolResult(content=text("done"))

    call = ToolCallBlock(id="c1", name="slow", arguments={})
    provider = FakeProvider([tool_call_response([call]), text_response("finished")])
    agent = _agent(provider, tools=(SlowTool(),))

    task = asyncio.create_task(agent.prompt("start"))
    await asyncio.sleep(0.01)
    agent.queue("also do this")
    await task

    second = provider.requests[1].messages
    user_texts = [m.content[0].text for m in second if isinstance(m, UserMessage)]
    assert "also do this" in user_texts


async def test_abort_marks_the_run_aborted() -> None:
    class HangingTool:
        name = "hang"
        description = "hangs"
        parameters: dict[str, object] = {"type": "object", "properties": {}}
        execution_mode = "parallel"

        async def execute(self, args: dict[str, object], ctx: ToolContext) -> ToolResult:
            await asyncio.sleep(0.05)
            return ToolResult(content=text("never"))

    call = ToolCallBlock(id="c1", name="hang", arguments={})
    provider = FakeProvider([tool_call_response([call]), text_response("unused")])
    agent = _agent(provider, tools=(HangingTool(),))

    task = asyncio.create_task(agent.prompt("go"))
    await asyncio.sleep(0.01)
    agent.abort()
    await task

    assert not agent.state.is_running
    assert any(
        isinstance(m, AssistantMessage) and m.stop_reason == "aborted" for m in agent.state.messages
    )


async def test_is_running_reflects_lifecycle() -> None:
    agent = _agent(FakeProvider([text_response("hi")]))
    assert not agent.state.is_running

    await agent.prompt("go")

    assert not agent.state.is_running


async def test_wait_for_idle_returns_after_the_run() -> None:
    agent = _agent(FakeProvider([text_response("hi")]))
    task = asyncio.create_task(agent.prompt("go"))
    await agent.wait_for_idle()
    await task

    assert not agent.state.is_running
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'harness.agent.agent'`

- [ ] **Step 3: Implement Agent**

Create `harness/agent/agent.py`:

```python
"""Stateful wrapper around the agent loop.

Owns the transcript, fans events out to subscribers, and exposes abort and
message queueing. This is the object a frontend talks to; `run_agent_loop`
stays pure.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from harness.agent.loop import run_agent_loop
from harness.agent.types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEvent,
    AgentTool,
    BeforeToolCallContext,
    BeforeToolCallResult,
    LoopConfig,
    MessageEnd,
    MessageStart,
    PrepareRequestContext,
    RequestUpdate,
)
from harness.config import Config, load_config
from harness.llm.provider import Provider
from harness.llm.types import AssistantMessage, Message, Model, UserMessage, text

logger = logging.getLogger(__name__)

Listener = Callable[[AgentEvent], Awaitable[None]]


@dataclass
class AgentState:
    system_prompt: str
    model: Model
    messages: list[Message] = field(default_factory=list)
    is_running: bool = False
    #: Text accumulated for the in-flight assistant message, for renderers that
    #: want the whole thing rather than deltas.
    streaming_message: str = ""


class Agent:
    def __init__(
        self,
        *,
        model: Model,
        provider: Provider,
        system_prompt: str = "",
        tools: tuple[AgentTool, ...] = (),
        config: Config | None = None,
        cwd: str = ".",
        before_tool_call: (
            Callable[[BeforeToolCallContext], Awaitable[BeforeToolCallResult | None]] | None
        ) = None,
        after_tool_call: (
            Callable[[AfterToolCallContext], Awaitable[AfterToolCallResult | None]] | None
        ) = None,
        prepare_request: (
            Callable[[PrepareRequestContext], Awaitable[RequestUpdate | None]] | None
        ) = None,
    ) -> None:
        self.state = AgentState(system_prompt=system_prompt, model=model)
        self._provider = provider
        self._tools = tools
        self._config = config or load_config()
        self._cwd = cwd
        self._before_tool_call = before_tool_call
        self._after_tool_call = after_tool_call
        self._prepare_request = prepare_request
        self._listeners: list[Listener] = []
        self._queue: list[Message] = []
        self._signal: asyncio.Event | None = None
        self._idle = asyncio.Event()
        self._idle.set()

    # --- subscriptions ----------------------------------------------------

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Register an event listener. Returns an unsubscribe callable."""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    async def _emit(self, event: AgentEvent) -> None:
        self._reduce(event)
        for listener in list(self._listeners):
            try:
                await listener(event)
            except Exception:
                # A broken renderer must not take down the run.
                logger.exception("Event listener raised; continuing")

    def _reduce(self, event: AgentEvent) -> None:
        """Fold an event into agent state before listeners see it."""
        if isinstance(event, MessageEnd):
            self.state.messages.append(event.message)
            if isinstance(event.message, AssistantMessage):
                self.state.streaming_message = ""
        elif isinstance(event, MessageStart):
            if isinstance(event.message, AssistantMessage):
                self.state.streaming_message = ""
        elif event.type == "message_update":
            self.state.streaming_message += event.delta

    # --- running ----------------------------------------------------------

    async def prompt(self, message: str | Message) -> list[Message]:
        """Send a message and run until the agent stops."""
        if self.state.is_running:
            raise RuntimeError(
                "Agent is already running. Use queue() to inject a message, or "
                "await wait_for_idle()."
            )

        user = UserMessage(content=text(message)) if isinstance(message, str) else message
        self.state.messages.append(user)
        return await self._run()

    async def _run(self) -> list[Message]:
        self.state.is_running = True
        self._idle.clear()
        self._signal = asyncio.Event()

        try:
            ctx = AgentContext(
                system_prompt=self.state.system_prompt,
                messages=tuple(self.state.messages),
                tools=self._tools,
            )
            cfg = LoopConfig(
                model=self.state.model,
                provider=self._provider,
                config=self._config,
                cwd=self._cwd,
                before_tool_call=self._before_tool_call,
                after_tool_call=self._after_tool_call,
                prepare_request=self._prepare_request_with_queue,
                signal=self._signal,
            )
            # The loop appends to its own transcript copy; _reduce keeps
            # self.state.messages in sync via MessageEnd events. Drop the
            # pre-existing messages the loop echoes back.
            produced = await run_agent_loop(ctx, cfg, self._emit)
            return produced
        finally:
            self.state.is_running = False
            self._signal = None
            self._idle.set()

    async def _prepare_request_with_queue(
        self, request: PrepareRequestContext
    ) -> RequestUpdate | None:
        """Inject queued messages at the turn boundary, then delegate.

        This is where 'type while it's working' lands: the queue is drained
        once per request, and the messages are emitted so subscribers render
        them in order alongside everything else.
        """
        messages = request.messages
        changed = False

        if self._queue:
            drained = tuple(self._queue)
            self._queue.clear()
            messages = messages + drained
            changed = True
            for message in drained:
                await self._emit(MessageStart(message=message))
                await self._emit(MessageEnd(message=message))

        if self._prepare_request is not None:
            update = await self._prepare_request(
                PrepareRequestContext(messages=messages, model=request.model)
            )
            if update is not None:
                return RequestUpdate(
                    messages=update.messages if update.messages is not None else messages,
                    model=update.model,
                )

        return RequestUpdate(messages=messages) if changed else None

    # --- control ----------------------------------------------------------

    def queue(self, message: str | Message) -> None:
        """Inject a message at the next turn boundary.

        This is how 'type while it's working' is handled — one queue, drained
        when the loop next prepares a request.
        """
        self._queue.append(
            UserMessage(content=text(message)) if isinstance(message, str) else message
        )

    def abort(self) -> None:
        """Cancel the in-flight run. The transcript stays valid."""
        if self._signal is not None:
            self._signal.set()

    async def wait_for_idle(self) -> None:
        await self._idle.wait()
```

Note `_reduce` appends on `MessageEnd`, so the user message added in `prompt()`
would be double-counted if the loop also emitted it. It does not — the loop
only emits messages it produces. Verify with
`test_prompt_appends_user_and_assistant_messages`: exactly two messages.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_agent.py -v`
Expected: 11 passed.

If `test_prompt_appends_user_and_assistant_messages` shows three messages, the
loop is echoing the input back — fix by having `run_agent_loop` return only
messages it produced, which the Task 3 implementation already does.

- [ ] **Step 5: Export the public API**

Replace `harness/agent/__init__.py`:

```python
"""The agent runtime: a pure loop and a stateful wrapper around it."""

from harness.agent.agent import Agent, AgentState
from harness.agent.loop import run_agent_loop
from harness.agent.types import (
    AgentContext,
    AgentEvent,
    AgentTool,
    LoopConfig,
    ToolContext,
    ToolResult,
)

__all__ = [
    "Agent",
    "AgentContext",
    "AgentEvent",
    "AgentState",
    "AgentTool",
    "LoopConfig",
    "ToolContext",
    "ToolResult",
    "run_agent_loop",
]
```

And `harness/llm/__init__.py`:

```python
"""Provider-neutral LLM access."""

from harness.llm.provider import Provider, StreamOptions
from harness.llm.types import (
    AssistantMessage,
    Context,
    Message,
    Model,
    SystemMessage,
    TextBlock,
    ToolCallBlock,
    ToolDeclaration,
    ToolResultMessage,
    Usage,
    UserMessage,
    text,
)

__all__ = [
    "AssistantMessage",
    "Context",
    "Message",
    "Model",
    "Provider",
    "StreamOptions",
    "SystemMessage",
    "TextBlock",
    "ToolCallBlock",
    "ToolDeclaration",
    "ToolResultMessage",
    "Usage",
    "UserMessage",
    "text",
]
```

- [ ] **Step 6: Write the milestone integration test**

Create `tests/test_integration.py`:

```python
"""End-to-end through the public API: a scripted model drives a real tool."""

from harness.agent import Agent, ToolContext, ToolResult
from harness.agent.types import AgentEvent
from harness.llm.fake import FAKE_MODEL, FakeProvider, text_response, tool_call_response
from harness.llm.types import ToolCallBlock, text


class CounterTool:
    name = "increment"
    description = "Add one to a number"
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }
    execution_mode = "parallel"

    async def execute(self, args: dict[str, object], ctx: ToolContext) -> ToolResult:
        return ToolResult(content=text(str(int(args["value"]) + 1)))


async def test_full_tool_calling_round_trip() -> None:
    call = ToolCallBlock(id="c1", name="increment", arguments={"value": 41})
    provider = FakeProvider(
        [
            tool_call_response([call], preamble="Let me compute that."),
            text_response("The answer is 42."),
        ]
    )
    events: list[AgentEvent] = []

    agent = Agent(
        model=FAKE_MODEL,
        provider=provider,
        system_prompt="You are a harness.",
        tools=(CounterTool(),),
    )
    agent.subscribe(lambda e: _record(events, e))

    await agent.prompt("What is 41 plus one?")

    assert agent.state.messages[-1].content[0].text == "The answer is 42."
    roles = [m.role for m in agent.state.messages]
    assert roles == ["user", "assistant", "tool_result", "assistant"]

    kinds = [e.type for e in events]
    assert kinds.count("turn_start") == 2
    assert "tool_execution_end" in kinds


async def _record(sink: list[AgentEvent], event: AgentEvent) -> None:
    sink.append(event)
```

The `lambda` passed to `subscribe` returns a coroutine, which `_emit` awaits —
that works, but if pyright objects, define a named async closure instead.

Run: `uv run pytest tests/test_integration.py -v`
Expected: 1 passed.

- [ ] **Step 7: Run every check**

```bash
uv run ruff check . && uv run ruff format .
uv run pyright
uv run pytest -v
```

Expected: ruff clean, pyright clean, roughly 55 tests passing.

- [ ] **Step 8: Update docs and propose a commit**

- `docs/events.md` — drop the "Not yet implemented" banner; the agent event
  table is now accurate.
- `docs/architecture.md` — mark build-order steps 1–4 complete.
- `README.md` — tick steps 1–4; move "Milestone 1a complete: tested runtime
  core" into In progress, and note that tools, providers, and the CLI are next.
- `docs/contributing.md` — no change expected; verify the invariants listed
  still match the code.

Proposed commit — **run only if the user asks**:

```bash
git add harness/ tests/ docs/ README.md
git commit -m "feat: stateful Agent with subscribers, queue, and abort"
```

---

## Milestone exit criteria

- [ ] `uv run pytest` passes with no network access
- [ ] `uv run pyright` clean
- [ ] `uv run ruff check .` clean
- [ ] Every `AgentEvent` and `StreamEvent` member round-trips through JSON, enforced by a coverage guard that fails when a new member is added without a test case
- [ ] A scripted model can drive a real tool end-to-end through `Agent.prompt()`
- [ ] Abort, queueing, hook blocking, parallel-vs-sequential ordering, truncated responses, and provider errors all have tests
- [ ] `docs/` and `README.md` reflect what exists

**Next milestone (1b):** `tools/base.py` + registry + filesystem tools, `cli/json_sink.py`, and the Anthropic provider — build-order steps 5–7, ending with a real agent run against a live API.
