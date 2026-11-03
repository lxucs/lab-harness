"""One non-streamed Chat Completions request, replayed as a stream.

Some endpoints cannot be streamed. An RL training gateway that records every
exchange it proxies needs a complete response body to record, so it refuses a
streaming request outright — and a harness that only knows how to stream
cannot be trained against one. A profile turns streaming off with
`stream = false`.

The loop speaks one protocol, so the reply is replayed as the shortest stream
that carries it: a single chunk in delta shape, fed to the adapter's own
accumulator, followed by the same end events the streaming path emits.
Reusing the accumulator is the point — tool-call assembly, the reasoning
fields, usage and the finish-reason mapping are translated by exactly one
piece of code whichever way the reply arrived.

The one visible difference is granularity. A block's whole text arrives in a
single delta, so a terminal prints the answer at once rather than word by
word; nothing downstream can tell the two apart otherwise.

**This module must never raise** for request or runtime failures — see
`lab_harness/llm/provider.py`. Every failure path ends in a `StreamError`
built by the shared `_http.failure`.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Protocol

import httpx

from lab_harness.config import OpenAIConfig
from lab_harness.llm.events import StreamDone, StreamEvent, StreamStart
from lab_harness.llm.provider import StreamOptions
from lab_harness.llm.providers import _http
from lab_harness.llm.types import AssistantMessage, Model, StopReason

#: Top-level reply keys a chunk carries too, so the accumulator reads them from
#: the same place either way. Everything else — `object`, `created`, and the
#: token ids a training server adds — is not part of what it reads.
_CARRIED = ("id", "model", "usage")


class Accumulating(Protocol):
    """What this module needs of the adapter's accumulator.

    Structural, so the accumulator stays private to its own module and this one
    does not import from it — which would be a cycle.
    """

    stop_reason: StopReason

    def feed(self, chunk: dict[str, Any]) -> list[StreamEvent]: ...

    def close(self) -> list[StreamEvent]: ...

    def message(self) -> AssistantMessage: ...


async def events(
    *,
    model: Model,
    provider: str,
    config: OpenAIConfig,
    client: httpx.AsyncClient | None,
    url: str,
    request: dict[str, Any],
    headers: dict[str, str],
    opts: StreamOptions,
    accumulator: Accumulating,
    retries: _http.Retries,
) -> AsyncIterator[StreamEvent]:
    """The stream for one POST that asked for the whole reply at once.

    Retried on a retryable failure, which is simpler here than on the streaming
    path: nothing reaches the consumer until the whole reply has arrived, so
    every attempt up to that point is repeatable.
    """
    # `again` owns the bound: it stops retrying at config.max_retries.
    while True:
        try:
            async with _http.client(client, config) as session:
                response = await _post(session, url, request, headers, opts)
        except Exception as exc:
            # httpx.HTTPError and anything else alike, same as the streaming path.
            stop = await _http.again(
                model=model,
                provider=provider,
                config=config,
                opts=opts,
                retries=retries,
                error=f"{type(exc).__name__}: {exc}",
            )
            if stop is None:
                continue
            yield stop
            return

        if response is None:
            # The signal fired while the request was in flight. Nothing
            # arrived, so there is no partial message to report with it.
            yield _http.failure(model, provider, _http.ABORTED_NOTICE, reason="aborted")
            return

        if response.status_code >= 400:
            stop = await _http.again(
                model=model,
                provider=provider,
                config=config,
                opts=opts,
                retries=retries,
                error=_http.describe(response, response.content),
                response=response,
            )
            if stop is None:
                continue
            yield stop
            return

        payload = _payload(response)
        if payload is None:
            # The mirror of `Frames.not_a_stream`: a gateway can refuse a
            # request with a 200 and a body of its own. Read as a reply it says
            # nothing, so without this the turn would end as a clean `stop`
            # carrying neither an answer nor an error. Not retried — it is a
            # refusal, and another identical request earns another one.
            yield _http.failure(
                model, provider, f"{_http.describe(response, response.content)} (expected a reply)"
            )
            return

        yield StreamStart()
        for translated in accumulator.feed(as_chunk(payload)):
            yield translated
        for closing in accumulator.close():
            yield closing
        yield StreamDone(message=accumulator.message(), reason=accumulator.stop_reason)
        return


async def _post(
    client: httpx.AsyncClient,
    url: str,
    request: dict[str, Any],
    headers: dict[str, str],
    opts: StreamOptions,
) -> httpx.Response | None:
    """The reply, or None when the abort signal fired before it arrived.

    A non-streamed request returns nothing until the whole reply is ready, so
    the signal cannot be read between frames the way the streaming path reads
    it. Racing the two is what keeps an abort answerable on a request that
    takes minutes — the wait the user would otherwise sit through is the entire
    response, not one delta.
    """
    sending: asyncio.Task[httpx.Response] = asyncio.create_task(
        client.post(url, json=request, headers=headers)
    )
    if opts.signal is None:
        return await sending

    waiting: asyncio.Task[Any] = asyncio.create_task(opts.signal.wait())
    racing: set[asyncio.Task[Any]] = {sending, waiting}
    done, pending = await asyncio.wait(racing, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    # Awaited, so a cancelled task is not still pending when it is collected —
    # which asyncio reports much later, against an unrelated line.
    await asyncio.gather(*pending, return_exceptions=True)
    return sending.result() if sending in done else None


def _payload(response: httpx.Response) -> dict[str, Any] | None:
    """The reply body, or None when the response is not a reply at all.

    `choices` is what makes a body one. A gateway can refuse a request with a
    200 and an envelope of its own — `{"code": 1005, "msg": "invalid model"}` —
    which parses as JSON and carries no answer; read as a reply it would end
    the turn as a clean `stop` saying nothing. This is the non-streamed half of
    what `Frames.not_a_stream` catches.
    """
    try:
        parsed = response.json()
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    choices = parsed.get("choices")
    return parsed if isinstance(choices, list) and choices else None


def as_chunk(payload: dict[str, Any]) -> dict[str, Any]:
    """A whole reply, in the shape one streaming chunk would have had.

    `choices[].message` becomes `choices[].delta` — the same fields under
    another name, the reasoning ones included — which is what lets the
    accumulator translate a non-streamed reply with no special case of its own.

    Tool calls are indexed on the way through, and that is the one piece of
    translation that is not a rename. A streamed call may arrive with no
    `index`, and the accumulator reads a missing one as 0 — right for a stream
    carrying a single call. A non-streamed reply indexes none of its calls, so
    without this every call in a parallel batch would accumulate into the same
    block: one tool call with several names concatenated and several argument
    objects run together.
    """
    choices = payload.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    choice: dict[str, Any] = first if isinstance(first, dict) else {}
    message = choice.get("message")
    delta: dict[str, Any] = dict(message) if isinstance(message, dict) else {}

    calls = delta.get("tool_calls")
    if isinstance(calls, list):
        delta["tool_calls"] = [
            call | {"index": call["index"] if isinstance(call.get("index"), int) else position}
            for position, call in enumerate(entry for entry in calls if isinstance(entry, dict))
        ]

    chunk: dict[str, Any] = {key: payload[key] for key in _CARRIED if key in payload}
    chunk["choices"] = [{"index": 0, "delta": delta, "finish_reason": choice.get("finish_reason")}]
    return chunk
