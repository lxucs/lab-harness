"""Web search and fetch.

Two tools and a seam. `web_search` finds pages, `web_fetch` reads one, and
`SearchBackend` is what makes the first of those swappable — Tavily by default,
ddgs when no key is available, anything else a user writes.

The split matters because search backends return *snippets*, not pages: a
title, a URL, and a sentence or two. The model usually learns from a search
where the answer is, then needs `web_fetch` to go read it. Treating fetch as an
afterthought would make search half a feature.

Backends raise `SearchError` for the expected failures — no key, rate limit,
upstream down. The tools turn those into failed `ToolResult`s, because a tool
that raises kills the turn and a tool that explains lets the model recover.
"""

import asyncio
import os
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field

from harness.config import WebConfig
from harness.tools.base import ToolContext, ToolResult, fail, ok, tool
from harness.tools.html_text import html_to_text

#: Content types parsed as HTML. Anything else is treated as text and passed
#: through, so fetching a JSON API or a raw .md file returns what it returned.
_HTML_TYPES = ("text/html", "application/xhtml")


class SearchError(Exception):
    """An expected search failure, phrased for the model to read."""


@dataclass(frozen=True, slots=True)
class SearchHit:
    title: str
    url: str
    #: One or two sentences. Rarely enough to answer from on its own.
    snippet: str
    #: Full page text, when the backend returns it. Tavily does with
    #: `include_raw_content`; ddgs never does.
    content: str = ""


class SearchBackend(Protocol):
    name: str

    async def search(self, query: str, max_results: int) -> tuple[SearchHit, ...]: ...


# --- backends -------------------------------------------------------------


class TavilyBackend:
    """Tavily's search API. Snippets are extracted against the query.

    `client` is injectable so tests can drive the real parsing code through a
    mock transport rather than reaching the network.
    """

    name = "tavily"

    def __init__(self, config: WebConfig, *, client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        self._client = client

    async def search(self, query: str, max_results: int) -> tuple[SearchHit, ...]:
        settings = self._config.tavily
        key = _credential(settings.api_key_env, settings.api_key)
        if not key:
            raise SearchError(
                f"No Tavily API key. Set ${settings.api_key_env}, or "
                f"`[tools.web.tavily] api_key` in .harness/config.local.toml. "
                f'Alternatively set `[tools.web] backend = "ddgs"` to search '
                f"without a key."
            )
        payload = {
            "query": query,
            "max_results": max_results,
            "search_depth": settings.search_depth,
            "include_raw_content": settings.include_raw_content,
        }
        data = await self._post(f"{settings.base_url}/search", payload, key)
        results = data.get("results")
        if not isinstance(results, list):
            raise SearchError("Tavily returned no results field.")
        return tuple(_tavily_hit(item) for item in results if isinstance(item, dict))

    async def _post(self, url: str, payload: dict[str, object], key: str) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": self._config.user_agent,
        }
        try:
            async with _client(self._client, self._config) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise SearchError(f"Could not reach Tavily: {exc}") from exc
        if response.status_code != 200:
            raise SearchError(f"Tavily returned {response.status_code}: {_api_error(response)}")
        try:
            body = response.json()
        except ValueError as exc:
            raise SearchError(f"Tavily returned invalid JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise SearchError("Tavily returned an unexpected payload.")
        return body


def _tavily_hit(item: dict[str, Any]) -> SearchHit:
    raw = item.get("raw_content")
    return SearchHit(
        title=str(item.get("title") or ""),
        url=str(item.get("url") or ""),
        snippet=str(item.get("content") or ""),
        content=str(raw) if raw else "",
    )


def _api_error(response: httpx.Response) -> str:
    """Pull the human-readable reason out of an error body.

    Tavily nests it as `{"detail": {"error": "..."}}`; other shapes fall back
    to the raw text, truncated.
    """
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict):
        detail = body.get("detail", body)
        if isinstance(detail, dict):
            return str(detail.get("error") or detail)
        return str(detail)
    return str(body)[:200]


class DdgsBackend:
    """Keyless metasearch via the `ddgs` package.

    `ddgs` is synchronous and does blocking network I/O, so calls go to a
    worker thread — running them inline would stall the event loop and with it
    the abort signal and every sibling tool call.
    """

    name = "ddgs"

    def __init__(self, config: WebConfig, *, search: Any = None) -> None:
        self._config = config
        self._search = search

    async def search(self, query: str, max_results: int) -> tuple[SearchHit, ...]:
        run = self._search or self._load()
        settings = self._config.ddgs
        try:
            rows = await asyncio.to_thread(
                run,
                query,
                region=settings.region,
                backend=settings.engine,
                max_results=max_results,
            )
        # ddgs wraps several engines; its failure modes are open-ended.
        except Exception as exc:
            raise SearchError(f"ddgs search failed: {type(exc).__name__}: {exc}") from exc
        return tuple(_ddgs_hit(row) for row in rows if isinstance(row, dict))

    def _load(self) -> Any:
        """Import ddgs on first use.

        Optional on purpose: Tavily is the default, and most users should not
        have to install a scraping stack they never call.
        """
        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise SearchError(
                "The ddgs backend needs the `ddgs` package: `uv sync --extra ddgs`."
            ) from exc
        return DDGS(timeout=int(self._config.timeout_seconds)).text


def _ddgs_hit(row: dict[str, Any]) -> SearchHit:
    return SearchHit(
        title=str(row.get("title") or ""),
        url=str(row.get("href") or ""),
        snippet=str(row.get("body") or ""),
    )


#: Every backend by its config name. Adding one is a line here plus a class.
BACKENDS: dict[str, Callable[[WebConfig], SearchBackend]] = {
    "tavily": TavilyBackend,
    "ddgs": DdgsBackend,
}


def build_backend(config: WebConfig) -> SearchBackend:
    """Construct the configured search backend.

    An unknown name raises rather than falling back to a default: silently
    searching with something other than what was configured is worse than
    refusing to start.
    """
    factory = BACKENDS.get(config.backend)
    if factory is None:
        available = ", ".join(sorted(BACKENDS))
        raise SearchError(f"Unknown web.backend {config.backend!r}. Available: {available}")
    return factory(config)


# --- tools ----------------------------------------------------------------


class WebSearchParams(BaseModel):
    query: str = Field(description="What to search for, phrased as a search query.")
    max_results: int | None = Field(
        default=None, ge=1, le=20, description="How many results to return."
    )


@tool(
    name="web_search",
    description=(
        "Search the web and return titles, URLs and short snippets. Snippets "
        "are rarely enough to answer from — use web_fetch on a result URL to "
        "read the page."
    ),
)
async def web_search(p: WebSearchParams, ctx: ToolContext) -> ToolResult:
    settings = ctx.config.tools.web
    try:
        backend = build_backend(settings)
        hits = await backend.search(p.query, p.max_results or settings.max_results)
    except SearchError as exc:
        return fail(str(exc))
    if not hits:
        return ok(f"No results for {p.query!r}.", query=p.query, count=0)
    return ok(
        _render_hits(hits, settings.max_content_chars),
        query=p.query,
        count=len(hits),
        backend=backend.name,
        urls=[hit.url for hit in hits],
    )


def _render_hits(hits: tuple[SearchHit, ...], max_chars: int) -> str:
    blocks = []
    for index, hit in enumerate(hits, start=1):
        lines = [f"{index}. {hit.title}".rstrip(), f"   {hit.url}"]
        if hit.snippet:
            lines.append(f"   {hit.snippet}")
        if hit.content:
            lines.append(_truncate(hit.content, max_chars // max(len(hits), 1)))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


class WebFetchParams(BaseModel):
    url: str = Field(description="Absolute http(s) URL to fetch.")
    max_chars: int | None = Field(
        default=None, ge=500, description="Characters of page text to return."
    )


@tool(
    name="web_fetch",
    description=(
        "Fetch a web page and return its text content as markdown. HTML is "
        "stripped to prose, headings, lists and links; other content types are "
        "returned as-is."
    ),
)
async def web_fetch(p: WebFetchParams, ctx: ToolContext) -> ToolResult:
    settings = ctx.config.tools.web
    if not p.url.startswith(("http://", "https://")):
        return fail(f"web_fetch needs an absolute http(s) URL; got {p.url!r}.")
    try:
        async with _client(None, settings) as client:
            response = await client.get(
                p.url,
                headers={"User-Agent": settings.user_agent},
                follow_redirects=True,
            )
    except httpx.HTTPError as exc:
        return fail(f"Could not fetch {p.url}: {exc}")

    if response.status_code >= 400:
        return fail(f"{p.url} returned HTTP {response.status_code}.")

    body = response.content[: settings.max_bytes]
    content_type = response.headers.get("content-type", "")
    try:
        raw = body.decode(response.encoding or "utf-8", errors="replace")
    except (LookupError, UnicodeDecodeError):
        raw = body.decode("utf-8", errors="replace")

    if any(kind in content_type for kind in _HTML_TYPES):
        title, extracted = html_to_text(raw)
    else:
        title, extracted = "", raw

    limit = p.max_chars or settings.max_content_chars
    header = f"# {title}\n\n" if title and not extracted.startswith("#") else ""
    return ok(
        f"{header}{_truncate(extracted, limit)}",
        url=str(response.url),
        title=title,
        content_type=content_type,
        status=response.status_code,
    )


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n\n[truncated at {limit} characters of {len(text)}]"


# --- shared ---------------------------------------------------------------


def _client(existing: httpx.AsyncClient | None, config: WebConfig) -> Any:
    """An async client context, reusing an injected one if given.

    A fresh client per call is fine for a CLI that searches occasionally, and
    it removes a lifecycle to get wrong. `nullcontext` keeps an injected test
    client from being closed out from under the test.
    """
    if existing is not None:
        return nullcontext(existing)
    return httpx.AsyncClient(timeout=config.timeout_seconds)


def _credential(env_name: str, configured: str) -> str:
    """Resolve a secret: environment first, then config.

    The order is the documented precedence, and it means a key left in a file
    by accident can be overridden rather than being the one that wins.
    """
    return os.environ.get(env_name) or configured


WEB_TOOLS = (web_search, web_fetch)

__all__ = [
    "BACKENDS",
    "WEB_TOOLS",
    "DdgsBackend",
    "SearchBackend",
    "SearchError",
    "SearchHit",
    "TavilyBackend",
    "build_backend",
    "web_fetch",
    "web_search",
]
