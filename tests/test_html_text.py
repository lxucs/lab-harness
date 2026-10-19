"""HTML to markdown-ish text.

What matters is not fidelity but signal per token: the prose and its structure
survive, the chrome does not, and malformed markup still produces something.
"""

from lab_harness.tools.html_text import html_to_text


def text(html: str) -> str:
    return html_to_text(html)[1]


# --- what survives --------------------------------------------------------


def test_headings_become_markdown() -> None:
    assert "## Section" in text("<h2>Section</h2>")


def test_list_items_become_bullets() -> None:
    body = text("<ul><li>one</li><li>two</li></ul>")
    assert "- one" in body and "- two" in body


def test_absolute_links_become_markdown_links() -> None:
    """A URL is actionable: the model can hand it straight to web_fetch."""
    assert "[docs](https://example.com/a)" in text('<a href="https://example.com/a">docs</a>')


def test_a_relative_link_keeps_its_text_but_is_not_linkified() -> None:
    """A relative href is unresolvable once the text leaves its page."""
    body = text('<a href="/about">about</a>')
    assert "about" in body and "](" not in body


def test_an_empty_anchor_produces_no_link() -> None:
    """`[](url)` is noise; an image-only anchor has nothing to label it."""
    assert "](" not in text('<a href="https://example.com"><img src="x.png"></a>')


def test_entities_are_decoded() -> None:
    assert "Tom & Jerry's" in text("<p>Tom &amp; Jerry&#39;s</p>")


def test_svg_titles_do_not_contaminate_the_page_title() -> None:
    """Found by fetching a real page: inline SVG icons carry <title> elements.

    peps.python.org has three theme-toggle icons, each with its own <title>,
    and concatenating them produced a title ending
    "…peps.python.orgFollowing system colour schemeSelected dark colour scheme".
    """
    title, _ = html_to_text(
        "<head><title>Real Title</title></head>"
        "<body><svg><title>Dark</title></svg><svg><title>Light</title></svg></body>"
    )
    assert title == "Real Title"


def test_only_the_first_title_is_used() -> None:
    """A later <title> is dropped rather than falling through into the body."""
    title, body = html_to_text("<title>First</title><body><title>Second</title><p>x</p></body>")
    assert title == "First" and "Second" not in body


def test_preformatted_text_is_fenced_and_keeps_its_whitespace() -> None:
    body = text("<pre>def f():\n    return 1</pre>")
    assert "```" in body and "    return 1" in body


# --- what is dropped ------------------------------------------------------


def test_scripts_are_dropped() -> None:
    assert "alert" not in text("<p>real</p><script>alert('x')</script>")


def test_styles_are_dropped() -> None:
    assert "color" not in text("<style>p{color:red}</style><p>real</p>")


def test_navigation_is_dropped() -> None:
    """Chrome repeats on every page and would crowd out the article.

    Output is truncated to a character budget, so a nav block is not merely
    wasted tokens — it can push the content off the end.
    """
    body = text("<nav><a href='https://e.com'>Home</a></nav><p>article</p>")
    assert "article" in body and "Home" not in body


def test_the_title_is_not_duplicated_into_the_body() -> None:
    """`<title>` lives in `<head>`, which is skipped wholesale."""
    title, body = html_to_text("<head><title>T</title></head><body><p>b</p></body>")
    assert title == "T" and "T" not in body


# --- shape ----------------------------------------------------------------


def test_block_elements_separate_lines() -> None:
    assert "\n" in text("<p>one</p><p>two</p>")


def test_runs_of_whitespace_collapse() -> None:
    assert "a b" in text("<p>a     \n\n   b</p>")


def test_blank_lines_do_not_accumulate() -> None:
    assert "\n\n\n" not in text("<div><div><div><p>x</p></div></div></div>")


def test_output_has_no_leading_or_trailing_whitespace() -> None:
    body = text("<html><body>\n\n  <p>x</p>\n\n</body></html>")
    assert body == body.strip()


# --- robustness -----------------------------------------------------------


def test_malformed_markup_does_not_raise() -> None:
    """The web is full of broken HTML; a half-parsed page beats a traceback."""
    assert "content" in text("<p>content<div><span></p></div>")


def test_an_unclosed_script_swallows_the_rest() -> None:
    """Deliberate: the parser cannot tell where the script was meant to end,
    and emitting JavaScript to the model is worse than emitting nothing."""
    assert "visible" not in text("<p>ok</p><script>var x = 1;<p>visible</p>")


def test_empty_input_is_empty_output() -> None:
    assert html_to_text("") == ("", "")


def test_plain_text_without_tags_survives() -> None:
    assert text("just words") == "just words"
