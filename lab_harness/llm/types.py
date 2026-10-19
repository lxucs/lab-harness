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
    #: How hard a reasoning model is asked to think, in the provider's own
    #: terms ("low", "high", …), on every request it is sent. Empty asks for
    #: nothing, leaving it to the model.
    reasoning_effort: str = ""


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
