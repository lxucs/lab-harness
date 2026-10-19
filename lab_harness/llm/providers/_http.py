"""Shared plumbing for the HTTP-based adapters.

`chat_completions.py` and `responses.py` speak different protocols but reach
them the same way: a bearer token resolved env-first, an httpx client that may
be injected for tests, SSE frames parsed off the body, and every failure
funnelled into one `StreamError` builder.

Nothing here knows about messages or events — that is each adapter's job.
"""

import json
import logging
import os
from contextlib import nullcontext
from typing import Any

import httpx

from lab_harness.config import OpenAIConfig
from lab_harness.llm.events import StreamError
from lab_harness.llm.provider import StreamOptions
from lab_harness.llm.types import AssistantMessage, Model

ABORTED_NOTICE = "Operation aborted"

#: Terminates a Chat Completions stream. Not JSON, so it is matched before
#: parsing. The Responses API never sends it; harmless to watch for anyway.
_DONE_PAYLOAD = "[DONE]"

_log = logging.getLogger(__name__)


class Done:
    """Sentinel for the terminating frame, so it cannot be mistaken for data."""


DONE = Done()


def credential(config: OpenAIConfig) -> str:
    """The API key: environment first, then config. Empty means none found."""
    return os.environ.get(config.api_key_env) or config.api_key


def log_sources(name: str, config: OpenAIConfig) -> None:
    """Which variable or config key the key and endpoint come from, at debug.

    Names only. A key is a key, and an overridden endpoint is often a private
    gateway — see docs/contributing.md.
    """
    if os.environ.get(config.api_key_env):
        key = config.api_key_env
    else:
        key = "api_key" if config.api_key else "nowhere"
    url = config.base_url_env if os.environ.get(config.base_url_env) else "base_url"
    _log.debug("Profile %s: key from %s, endpoint from %s", name, key, url)


def endpoint(config: OpenAIConfig, path: str) -> str:
    base = os.environ.get(config.base_url_env) or config.base_url
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def client(existing: httpx.AsyncClient | None, config: OpenAIConfig) -> Any:
    """The injected client, or a new one scoped to this request.

    `nullcontext` so an injected client survives the `async with` — the caller
    owns it, and closing someone else's client breaks the next request.
    """
    if existing is not None:
        return nullcontext(existing)
    return httpx.AsyncClient(timeout=config.timeout_seconds)


def frame(line: str) -> dict[str, Any] | Done | None:
    """One SSE line -> a parsed payload, `DONE`, or None to skip.

    Keepalives are blank lines and comments start with ':'. `event:` lines are
    skipped too: both protocols repeat the event type inside the JSON payload,
    so the data line alone is enough and neither adapter has to track state
    across two lines.
    """
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    body = line[len("data:") :].strip()
    if not body:
        return None
    if body == _DONE_PAYLOAD:
        return DONE
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        # A malformed frame is skipped rather than failing the turn: the rest
        # of the stream is usually fine, and losing one delta beats losing the
        # whole answer.
        _log.debug("Skipping unparseable SSE frame")
        return None
    return payload if isinstance(payload, dict) else None


class Frames:
    """Parses a 2xx body line by line, telling a stream from a body that is not.

    A gateway can refuse a request with a 200: one asked for a model it does
    not serve answers `{"code": 1005, "msg": "invalid model"}`. No line of that
    is a frame, so read as a stream it says nothing, and the turn would end as
    a clean `stop` with no answer and no error. So lines are kept until the
    first frame arrives, and a body that never sends one is reported the way an
    error status is.
    """

    def __init__(self) -> None:
        self._framed = False
        self._unframed: list[str] = []

    def parse(self, line: str) -> dict[str, Any] | Done | None:
        """`frame(line)`, noting whether the body has shown itself a stream."""
        payload = frame(line)
        if payload is not None:
            self._framed = True
            self._unframed.clear()
        elif not self._framed:
            self._unframed.append(line)
        return payload

    def not_a_stream(self, response: httpx.Response) -> str | None:
        """The error for a body that ended without one frame, else None."""
        if self._framed:
            return None
        body = "\n".join(self._unframed).encode()
        return f"{describe(response, body)} (expected an event stream)"


def describe(response: httpx.Response, body: bytes) -> str:
    """A message the user can act on, without leaking the request.

    Endpoints disagree on the error envelope: OpenAI nests under `error`,
    several compatible servers put `message` at the top level, and a
    misconfigured base_url returns HTML. Each case degrades to the next.
    """
    detail = ""
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            detail = str(error.get("message", ""))
        elif isinstance(error, str):
            detail = error
        if not detail:
            detail = str(parsed.get("message", ""))
    if not detail:
        text = body.decode("utf-8", errors="replace").strip()
        detail = text[:200] if text else response.reason_phrase
    return f"HTTP {response.status_code}: {detail}"


def aborted(opts: StreamOptions) -> bool:
    return opts.signal is not None and opts.signal.is_set()


def precondition(
    model: Model, provider: str, config: OpenAIConfig, opts: StreamOptions
) -> StreamError | None:
    """The checks both adapters run before touching the network.

    Returns the event to yield instead of streaming, or None to proceed.
    """
    if not credential(config):
        return failure(model, provider, missing_credential(provider, config))
    if aborted(opts):
        return failure(model, provider, ABORTED_NOTICE, reason="aborted")
    return None


def tool_arguments(raw: str) -> dict[str, Any]:
    """Parse a streamed tool-call argument string.

    Never salvages partial JSON. A truncated object can parse into arguments
    that validate while silently missing fields, and a tool called with
    plausible-but-wrong arguments does real damage — deletes the wrong path,
    edits the wrong file. `{}` makes the model re-issue the call instead.
    """
    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def failure(
    model: Model,
    provider: str,
    error: str,
    *,
    reason: str = "error",
) -> StreamError:
    """The one place a failure becomes a stream event. Nothing here raises."""
    if reason == "aborted":
        _log.debug("Stream aborted for %s via %s", model.id, provider)
    else:
        _log.warning("%s request failed for %s: %s", provider, model.id, error)
    return StreamError(
        message=AssistantMessage(
            content=(),
            model=model.id,
            provider=provider,
            stop_reason="aborted" if reason == "aborted" else "error",
            error_message=error,
        ),
        reason="aborted" if reason == "aborted" else "error",
    )


def missing_credential(provider: str, config: OpenAIConfig) -> str:
    return (
        f"No credentials for provider {provider!r}. Set {config.api_key_env}, "
        f"or providers.openai.{provider}.api_key in config.local.toml."
    )
