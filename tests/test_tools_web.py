"""Web search and fetch.

Everything here is offline. Tavily is driven through `httpx.MockTransport`, so
the real request-building and response-parsing code runs against canned bytes;
ddgs gets an injected callable in place of the package. The canned Tavily
payloads are the shapes the live API actually returned, not invented ones.
"""

from dataclasses import replace
from typing import Any

import httpx
import pytest

from harness.config import Config, DdgsConfig, TavilyConfig, WebConfig
from harness.tools import web
from harness.tools.base import ToolContext
from harness.tools.web import (
    BACKENDS,
    DdgsBackend,
    SearchError,
    SearchHit,
    TavilyBackend,
    build_backend,
    web_fetch,
    web_search,
)

# The live API returns exactly these fields; see docs/tools.md.
TAVILY_BODY = {
    "query": "capital of France",
    "answer": None,
    "response_time": 1.12,
    "results": [
        {
            "title": "List of capitals of France - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/List_of_capitals_of_France",
            "content": "This is a chronological list of capitals of France.",
            "score": 0.8652358,
            "raw_content": None,
            "id": "f54b71-00",
        }
    ],
}


def web_config(**overrides: Any) -> WebConfig:
    """A WebConfig with a key set, so tests exercise the request not the guard."""
    tavily = TavilyConfig(api_key="test-key")
    return replace(WebConfig(tavily=tavily), **overrides)


def context(config: WebConfig) -> ToolContext:
    base = Config()
    return ToolContext(
        cwd=".",
        call_id="c1",
        config=replace(base, tools=replace(base.tools, web=config)),
    )


def transport(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def responder(body: Any, status: int = 200) -> Any:
    def handle(request: httpx.Request) -> httpx.Response:
        handle.request = request  # type: ignore[attr-defined]
        return httpx.Response(status, json=body)

    return handle


# --- Tavily: the request --------------------------------------------------


async def test_tavily_posts_to_the_search_endpoint() -> None:
    handle = responder(TAVILY_BODY)
    config = web_config()
    await TavilyBackend(config, client=transport(handle)).search("q", 5)
    assert str(handle.request.url) == f"{config.tavily.base_url}/search"


async def test_tavily_sends_the_key_as_a_bearer_token() -> None:
    handle = responder(TAVILY_BODY)
    await TavilyBackend(web_config(), client=transport(handle)).search("q", 5)
    assert handle.request.headers["authorization"] == "Bearer test-key"


async def test_tavily_sends_the_query_and_limit() -> None:
    import json

    handle = responder(TAVILY_BODY)
    await TavilyBackend(web_config(), client=transport(handle)).search("penguins", 3)
    payload = json.loads(handle.request.content)
    assert payload["query"] == "penguins" and payload["max_results"] == 3


async def test_tavily_sends_the_configured_search_depth() -> None:
    """Depth is a cost knob: advanced spends two API credits instead of one."""
    import json

    handle = responder(TAVILY_BODY)
    config = web_config(tavily=TavilyConfig(api_key="k", search_depth="advanced"))
    await TavilyBackend(config, client=transport(handle)).search("q", 5)
    assert json.loads(handle.request.content)["search_depth"] == "advanced"


async def test_the_key_comes_from_the_environment_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env beats file, so a key left in a config file can be overridden."""
    monkeypatch.setenv("TAVILY_API_KEY", "from-env")
    handle = responder(TAVILY_BODY)
    await TavilyBackend(web_config(), client=transport(handle)).search("q", 5)
    assert handle.request.headers["authorization"] == "Bearer from-env"


# --- Tavily: the response -------------------------------------------------


async def test_tavily_parses_the_live_response_shape() -> None:
    hits = await TavilyBackend(web_config(), client=transport(responder(TAVILY_BODY))).search(
        "q", 5
    )
    assert len(hits) == 1
    assert hits[0].title == "List of capitals of France - Wikipedia"
    assert hits[0].url == "https://en.wikipedia.org/wiki/List_of_capitals_of_France"
    assert hits[0].snippet.startswith("This is a chronological list")


async def test_a_null_raw_content_is_not_the_string_none() -> None:
    """`raw_content` is null unless requested; `str(None)` would be a bug."""
    hits = await TavilyBackend(web_config(), client=transport(responder(TAVILY_BODY))).search(
        "q", 5
    )
    assert hits[0].content == ""


async def test_raw_content_becomes_page_content_when_present() -> None:
    body = {"results": [{"title": "t", "url": "u", "content": "s", "raw_content": "full page"}]}
    hits = await TavilyBackend(web_config(), client=transport(responder(body))).search("q", 5)
    assert hits[0].content == "full page"


async def test_results_without_a_results_field_are_an_error() -> None:
    client = transport(responder({"query": "q"}))
    with pytest.raises(SearchError, match="no results field"):
        await TavilyBackend(web_config(), client=client).search("q", 5)


# --- Tavily: failure ------------------------------------------------------


async def test_a_missing_key_explains_both_ways_to_set_one() -> None:
    config = replace(WebConfig(), tavily=TavilyConfig(api_key=""))
    with pytest.raises(SearchError) as caught:
        await TavilyBackend(config).search("q", 5)
    message = str(caught.value)
    assert "TAVILY_API_KEY" in message and "config.local.toml" in message


async def test_a_missing_key_points_at_the_keyless_backend() -> None:
    """The useful next step when you have no key is to stop needing one."""
    config = replace(WebConfig(), tavily=TavilyConfig(api_key=""))
    with pytest.raises(SearchError, match="ddgs"):
        await TavilyBackend(config).search("q", 5)


async def test_an_api_error_surfaces_the_upstream_reason() -> None:
    """Tavily nests the reason as {"detail": {"error": ...}}."""
    body = {"detail": {"error": "Query is too short. Min query length is 2 characters."}}
    client = transport(responder(body, status=400))
    with pytest.raises(SearchError, match="Min query length"):
        await TavilyBackend(web_config(), client=client).search("q", 5)


async def test_an_api_error_includes_the_status_code() -> None:
    client = transport(responder({"detail": "nope"}, status=432))
    with pytest.raises(SearchError, match="432"):
        await TavilyBackend(web_config(), client=client).search("q", 5)


async def test_an_unparseable_error_body_still_produces_a_message() -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"<html>Gateway Error</html>")

    with pytest.raises(SearchError, match="500"):
        await TavilyBackend(web_config(), client=transport(handle)).search("q", 5)


async def test_a_transport_failure_becomes_a_search_error() -> None:
    """Never an httpx exception escaping into the loop."""

    def handle(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns failure")

    with pytest.raises(SearchError, match="Could not reach Tavily"):
        await TavilyBackend(web_config(), client=transport(handle)).search("q", 5)


# --- ddgs -----------------------------------------------------------------


def ddgs_rows(*rows: dict[str, Any]) -> Any:
    def run(query: str, **kwargs: Any) -> list[dict[str, Any]]:
        run.calls.append((query, kwargs))  # type: ignore[attr-defined]
        return list(rows)

    run.calls = []  # type: ignore[attr-defined]
    return run


async def test_ddgs_maps_its_field_names() -> None:
    """ddgs returns title/href/body, not title/url/content."""
    run = ddgs_rows({"title": "T", "href": "https://e.com", "body": "B"})
    hits = await DdgsBackend(web_config(), search=run).search("q", 5)
    assert hits == (SearchHit(title="T", url="https://e.com", snippet="B"),)


async def test_ddgs_passes_through_region_and_engine() -> None:
    run = ddgs_rows()
    config = web_config(ddgs=DdgsConfig(engine="brave", region="uk-en"))
    await DdgsBackend(config, search=run).search("q", 7)
    _, kwargs = run.calls[0]
    assert kwargs == {"region": "uk-en", "backend": "brave", "max_results": 7}


async def test_ddgs_never_reports_page_content() -> None:
    """It returns snippets only — which is why web_fetch matters."""
    run = ddgs_rows({"title": "T", "href": "u", "body": "B"})
    hits = await DdgsBackend(web_config(), search=run).search("q", 5)
    assert hits[0].content == ""


async def test_a_ddgs_failure_becomes_a_search_error() -> None:
    def boom(query: str, **kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("rate limited")

    with pytest.raises(SearchError, match="rate limited"):
        await DdgsBackend(web_config(), search=boom).search("q", 5)


async def test_a_missing_ddgs_package_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def deny(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "ddgs":
            raise ImportError("No module named 'ddgs'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny)
    with pytest.raises(SearchError, match="--extra ddgs"):
        await DdgsBackend(web_config()).search("q", 5)


# --- backend selection ----------------------------------------------------


def test_the_default_backend_is_tavily() -> None:
    assert WebConfig().backend == "tavily"
    assert isinstance(build_backend(WebConfig()), TavilyBackend)


def test_ddgs_is_selectable_by_config() -> None:
    assert isinstance(build_backend(replace(WebConfig(), backend="ddgs")), DdgsBackend)


def test_an_unknown_backend_names_the_available_ones() -> None:
    """Falling back silently would search with something you did not choose."""
    with pytest.raises(SearchError, match=r"tavily, ddgs|ddgs, tavily"):
        build_backend(replace(WebConfig(), backend="nope"))


def test_every_registered_backend_reports_its_own_name() -> None:
    for name, factory in BACKENDS.items():
        assert factory(WebConfig()).name == name


# --- the web_search tool --------------------------------------------------


class FakeBackend:
    name = "fake"
    hits: tuple[SearchHit, ...] = ()
    error: str | None = None

    def __init__(self, config: WebConfig) -> None:
        self.config = config

    async def search(self, query: str, max_results: int) -> tuple[SearchHit, ...]:
        FakeBackend.seen = (query, max_results)
        if FakeBackend.error:
            raise SearchError(FakeBackend.error)
        return FakeBackend.hits


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> type[FakeBackend]:
    """Register a fake under a config name, so the real selection path runs."""
    FakeBackend.hits = (SearchHit(title="T", url="https://e.com/a", snippet="S"),)
    FakeBackend.error = None
    monkeypatch.setitem(BACKENDS, "fake", FakeBackend)
    return FakeBackend


async def test_web_search_renders_title_url_and_snippet(
    fake_backend: type[FakeBackend],
) -> None:
    result = await web_search.execute({"query": "q"}, context(web_config(backend="fake")))
    body = result.content[0].text  # type: ignore[union-attr]
    assert "T" in body and "https://e.com/a" in body and "S" in body


async def test_web_search_reports_the_urls_as_details(
    fake_backend: type[FakeBackend],
) -> None:
    """Details are for renderers; the model sees only the text."""
    result = await web_search.execute({"query": "q"}, context(web_config(backend="fake")))
    assert result.details["urls"] == ["https://e.com/a"]
    assert result.details["backend"] == "fake"


async def test_web_search_uses_the_configured_result_count(
    fake_backend: type[FakeBackend],
) -> None:
    """The load-bearing config test: two configs must behave differently."""
    await web_search.execute({"query": "q"}, context(web_config(backend="fake", max_results=2)))
    assert FakeBackend.seen[1] == 2
    await web_search.execute({"query": "q"}, context(web_config(backend="fake", max_results=9)))
    assert FakeBackend.seen[1] == 9


async def test_an_explicit_limit_overrides_the_configured_one(
    fake_backend: type[FakeBackend],
) -> None:
    await web_search.execute(
        {"query": "q", "max_results": 4}, context(web_config(backend="fake", max_results=2))
    )
    assert FakeBackend.seen[1] == 4


async def test_a_search_failure_is_a_failed_result_not_an_exception(
    fake_backend: type[FakeBackend],
) -> None:
    """A raising tool kills the turn; a failed result lets the model recover."""
    FakeBackend.error = "no key configured"
    result = await web_search.execute({"query": "q"}, context(web_config(backend="fake")))
    assert result.is_error
    assert "no key configured" in result.content[0].text  # type: ignore[union-attr]


async def test_an_unknown_backend_fails_the_tool_rather_than_the_turn() -> None:
    result = await web_search.execute({"query": "q"}, context(web_config(backend="nope")))
    assert result.is_error


async def test_no_results_is_a_success_not_a_failure(
    fake_backend: type[FakeBackend],
) -> None:
    """Finding nothing is an answer. An error would invite a pointless retry."""
    FakeBackend.hits = ()
    result = await web_search.execute({"query": "zzz"}, context(web_config(backend="fake")))
    assert not result.is_error
    assert "No results" in result.content[0].text  # type: ignore[union-attr]


async def test_a_missing_query_is_a_validation_failure() -> None:
    result = await web_search.execute({}, context(web_config()))
    assert result.is_error and "query" in result.content[0].text  # type: ignore[union-attr]


# --- the web_fetch tool ---------------------------------------------------


@pytest.fixture
def fetch(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Point web_fetch's client factory at a mock transport."""

    def install(handler: Any) -> None:
        client = transport(handler)
        monkeypatch.setattr(web, "_client", lambda existing, config: _reuse(client))

    return install


def _reuse(client: httpx.AsyncClient) -> Any:
    from contextlib import nullcontext

    return nullcontext(client)


def page(html: str, content_type: str = "text/html") -> Any:
    return lambda request: httpx.Response(
        200, content=html.encode(), headers={"content-type": content_type}
    )


async def test_web_fetch_returns_page_text_as_markdown(fetch: Any) -> None:
    fetch(page("<html><body><h1>Title</h1><p>Body text.</p></body></html>"))
    result = await web_fetch.execute({"url": "https://e.com"}, context(web_config()))
    body = result.content[0].text  # type: ignore[union-attr]
    assert "# Title" in body and "Body text." in body


async def test_web_fetch_strips_scripts(fetch: Any) -> None:
    fetch(page("<p>real</p><script>tracking()</script>"))
    result = await web_fetch.execute({"url": "https://e.com"}, context(web_config()))
    assert "tracking" not in result.content[0].text  # type: ignore[union-attr]


async def test_non_html_content_passes_through_unparsed(fetch: Any) -> None:
    """Fetching a JSON API should return JSON, not JSON run through an HTML parser."""
    fetch(page('{"a": 1}', content_type="application/json"))
    result = await web_fetch.execute({"url": "https://e.com/api"}, context(web_config()))
    assert '{"a": 1}' in result.content[0].text  # type: ignore[union-attr]


async def test_a_relative_url_is_refused() -> None:
    result = await web_fetch.execute({"url": "/local/path"}, context(web_config()))
    assert result.is_error and "absolute" in result.content[0].text  # type: ignore[union-attr]


async def test_a_file_url_is_refused() -> None:
    """`read` is the tool for local files; web_fetch must not become a second one."""
    result = await web_fetch.execute({"url": "file:///etc/passwd"}, context(web_config()))
    assert result.is_error


async def test_an_http_error_is_a_failed_result(fetch: Any) -> None:
    fetch(lambda request: httpx.Response(404))
    result = await web_fetch.execute({"url": "https://e.com/missing"}, context(web_config()))
    assert result.is_error and "404" in result.content[0].text  # type: ignore[union-attr]


async def test_a_transport_failure_is_a_failed_result(fetch: Any) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    fetch(boom)
    result = await web_fetch.execute({"url": "https://e.com"}, context(web_config()))
    assert result.is_error and "Could not fetch" in result.content[0].text  # type: ignore[union-attr]


async def test_long_pages_are_truncated_with_a_marker(fetch: Any) -> None:
    fetch(page(f"<p>{'x' * 5000}</p>"))
    result = await web_fetch.execute(
        {"url": "https://e.com"}, context(web_config(max_content_chars=500))
    )
    body = result.content[0].text  # type: ignore[union-attr]
    assert "truncated at 500 characters" in body


async def test_the_character_budget_comes_from_config(fetch: Any) -> None:
    """Two contexts with different limits must produce different output."""
    fetch(page(f"<p>{'x' * 5000}</p>"))
    small = await web_fetch.execute(
        {"url": "https://e.com"}, context(web_config(max_content_chars=500))
    )
    large = await web_fetch.execute(
        {"url": "https://e.com"}, context(web_config(max_content_chars=4000))
    )
    assert len(small.content[0].text) < len(large.content[0].text)  # type: ignore[union-attr]


async def test_an_explicit_max_chars_overrides_config(fetch: Any) -> None:
    fetch(page(f"<p>{'x' * 5000}</p>"))
    result = await web_fetch.execute(
        {"url": "https://e.com", "max_chars": 600},
        context(web_config(max_content_chars=4000)),
    )
    assert "truncated at 600 characters" in result.content[0].text  # type: ignore[union-attr]


async def test_the_download_is_capped_before_extraction(fetch: Any) -> None:
    """max_bytes guards memory; max_content_chars guards context. Both apply."""
    fetch(page(f"<p>{'y' * 100_000}</p>"))
    result = await web_fetch.execute({"url": "https://e.com"}, context(web_config(max_bytes=1000)))
    assert result.details["status"] == 200
    assert len(result.content[0].text) < 2000  # type: ignore[union-attr]


async def test_the_resolved_url_is_reported(fetch: Any) -> None:
    """Redirects are followed, so the model should see where it landed."""
    fetch(page("<p>x</p>"))
    result = await web_fetch.execute({"url": "https://e.com/a"}, context(web_config()))
    assert result.details["url"] == "https://e.com/a"
