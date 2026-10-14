"""HTML to markdown-ish text, using only the standard library.

`web_fetch` exists to put a page's prose in front of the model, not to render
it faithfully. This keeps the structure that carries meaning — headings, lists,
links, code — and drops what costs tokens without adding any: scripts, styling,
navigation chrome.

Dropping navigation matters more than it looks. Output is capped at a character
budget, so a sidebar of two hundred links at the top of the document does not
merely waste tokens: it pushes the actual article off the end.

Hand-rolled rather than pulling in a converter, for the same reason `grep` and
`glob` are — a dependency should earn itself, and this is eighty lines.
"""

import re
from html.parser import HTMLParser

#: Elements whose entire subtree is discarded. Either invisible (script, style)
#: or chrome that repeats on every page of a site (nav, footer, aside).
_SKIP = frozenset(
    {
        "script",
        "style",
        "noscript",
        # <head> is deliberately NOT skipped: <title> lives there and names the
        # page. Everything else in it is either void or separately skipped.
        "svg",
        "template",
        "iframe",
        "object",
        "embed",
        "canvas",
        "form",
        "nav",
        "aside",
        "footer",
    }
)

#: Elements that end the current line.
_BLOCK = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "main",
        "header",
        "blockquote",
        "pre",
        "ul",
        "ol",
        "li",
        "table",
        "tr",
        "hr",
        "figure",
        "figcaption",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }
)

_HEADINGS = {f"h{level}": "#" * level for level in range(1, 7)}

#: Outside <pre>, HTML collapses every run of whitespace — newlines included —
#: to a single space. Source line breaks inside a paragraph are not content.
_WHITESPACE = re.compile(r"\s+")
_BLANK_LINES = re.compile(r"\n{3,}")


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        # convert_charrefs lets the parser resolve &amp; and friends for us.
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._parts: list[str] = []
        self._skip = 0
        self._in_title = False
        self._have_title = False
        self._pre = 0
        self._link: str | None = None
        self._link_at = 0

    # --- structure --------------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self._skip += 1
            return
        if self._skip:
            return
        # Checked after the skip set, not before: inline SVG icons carry their
        # own <title> elements, and a page with three theme-toggle icons would
        # otherwise end up titled "Real TitleDarkLightAuto".
        if tag == "title":
            self._in_title = True
            return
        if tag == "br":
            self._parts.append("\n")
        elif tag == "pre":
            self._pre += 1
            self._parts.append("\n```\n")
        elif tag == "li":
            self._parts.append("\n- ")
        elif tag in _HEADINGS:
            self._parts.append(f"\n\n{_HEADINGS[tag]} ")
        elif tag == "a":
            href = dict(attrs).get("href") or ""
            # Only absolute http(s) links are useful to the model: a relative
            # href is unresolvable once the text leaves its page.
            if href.startswith(("http://", "https://")):
                self._link = href
                self._link_at = len(self._parts)
        elif tag in _BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag == "title":
            self._in_title = False
            self._have_title = True
            return
        if tag == "pre":
            self._pre = max(0, self._pre - 1)
            self._parts.append("\n```\n")
        elif tag == "a":
            self._close_link()
        elif tag in _BLOCK:
            self._parts.append("\n")

    def _close_link(self) -> None:
        """Wrap the anchor's text as a markdown link, if it had any."""
        href, self._link = self._link, None
        if href is None:
            return
        if not "".join(self._parts[self._link_at :]).strip():
            return  # An image-only or empty anchor; a bare "[](url)" helps nobody.
        self._parts.insert(self._link_at, "[")
        self._parts.append(f"]({href})")

    # --- text -------------------------------------------------------------

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._in_title:
            # First title wins, and later ones are dropped rather than falling
            # through into the body.
            if not self._have_title:
                self.title += data.strip()
            return
        self._parts.append(data if self._pre else _WHITESPACE.sub(" ", data))

    def text(self) -> str:
        collapsed = _BLANK_LINES.sub("\n\n", "".join(self._parts))
        return "\n".join(line.rstrip() for line in collapsed.split("\n")).strip()


def html_to_text(html: str) -> tuple[str, str]:
    """Extract `(title, markdown-ish text)` from an HTML document.

    Malformed markup is not an error. Python's parser is lenient by design and
    the web is full of broken HTML; a page that half-parses still beats a
    traceback where the content should be.
    """
    extractor = _Extractor()
    extractor.feed(html)
    extractor.close()
    return extractor.title, extractor.text()
