"""The one tool this program adds: ranking the corpus.

`read`, `grep` and `glob` find a document you can name. They cannot find one
you can only describe, which is the whole shape of the task here — so the
agent gets a search tool over the same files. It is BM25, which is a dozen
lines of arithmetic over term counts and needs no dependency: rare terms weigh
more, and a term repeated inside a long document counts for less than the same
term in a short one.
"""

import math
import re
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from lab_harness.tools import FunctionTool, ToolContext, ToolResult, fail, ok, tool

#: The name as config and the tool list know it.
SEARCH_NAME = "search_docs"

#: BM25's two knobs: how fast a repeated term saturates, and how much a long
#: document is penalised for its length.
_K1 = 1.5
_B = 0.75
_WORD = re.compile(r"[a-z0-9]+")
_SNIPPET = 220


class SearchParams(BaseModel):
    query: str = Field(description="What to look for: names, nouns and phrases, not a sentence")
    top: int = Field(8, ge=1, le=25, description="How many documents to list")


def _body(text: str) -> str:
    """The document without its front matter."""
    if text.startswith("---") and text.count("---") >= 2:
        return text.split("---", 2)[2].strip()
    return text.strip()


def _title(text: str) -> str:
    """The document's title, from the front matter it came with."""
    if not text.startswith("---") or text.count("---") < 2:
        return ""
    for line in text.split("---", 2)[1].splitlines():
        if line.startswith("title:"):
            return line.removeprefix("title:").strip()
    return ""


def _terms(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class _Corpus:
    """Term counts per document, and the totals BM25 reads around them."""

    def __init__(self, directory: Path) -> None:
        self._counts: dict[str, Counter[str]] = {}
        self._titles: dict[str, str] = {}
        self._openings: dict[str, str] = {}
        self._lengths: dict[str, int] = {}
        for path in sorted(directory.glob("*.md")):
            text = path.read_text(encoding="utf-8", errors="replace")
            self._counts[path.stem] = Counter(_terms(_body(text)))
            self._titles[path.stem] = _title(text)
            self._openings[path.stem] = " ".join(_body(text).split())
            self._lengths[path.stem] = sum(self._counts[path.stem].values())
        self._average = sum(self._lengths.values()) / len(self._lengths) if self._lengths else 0.0

    def title(self, docid: str) -> str:
        return self._titles.get(docid, "")

    def snippet(self, docid: str, width: int = _SNIPPET) -> str:
        """The opening of a document, for judging a hit without reading it."""
        text = self._openings.get(docid, "")
        return text[:width] + ("…" if len(text) > width else "")

    def search(self, query: str, top: int) -> list[tuple[str, float]]:
        """The documents best matching `query`, best first, with their scores."""
        if not self._counts:
            return []
        wanted = dict.fromkeys(_terms(query))
        rarity = {term: self._rarity(term) for term in wanted}
        scored = [
            (docid, sum(self._weight(docid, term, rarity[term]) for term in wanted))
            for docid in self._counts
        ]
        scored = [hit for hit in scored if hit[1] > 0]
        scored.sort(key=lambda hit: hit[1], reverse=True)
        return scored[:top]

    def _rarity(self, term: str) -> float:
        """Inverse document frequency: a term in every document says nothing."""
        holding = sum(1 for counts in self._counts.values() if term in counts)
        return math.log(1 + len(self._counts) / (1 + holding))

    def _weight(self, docid: str, term: str, rarity: float) -> float:
        """What one term contributes to one document's score."""
        held = self._counts[docid][term]
        if not held or not self._average:
            return 0.0
        length = self._lengths[docid]
        saturated = held + _K1 * (1 - _B + _B * length / self._average)
        return rarity * held * (_K1 + 1) / saturated


_INDEXES: dict[Path, _Corpus] = {}


def _index(directory: Path) -> _Corpus:
    """The index for `directory`, built once: the corpus does not change
    during a run, and one query may search it a dozen times."""
    if directory not in _INDEXES:
        _INDEXES[directory] = _Corpus(directory)
    return _INDEXES[directory]


def corpus_search(corpus: str) -> FunctionTool[SearchParams]:
    """The search tool over `corpus` — a directory under the agent's own.

    Built for a directory rather than hardcoding one, so the corpus path in
    `search_agent.toml` is the only place it is written down.
    """

    @tool(
        name=SEARCH_NAME,
        description=(
            "Rank the corpus documents against a query. Returns each hit's docid, "
            "title and opening lines; read the ones that look promising."
        ),
    )
    async def search(p: SearchParams, ctx: ToolContext) -> ToolResult:
        directory = Path(ctx.cwd) / corpus
        if not directory.is_dir():
            return fail(f"There is no {corpus}/ directory here. The corpus is {corpus}/<docid>.md.")
        index = _index(directory)
        hits = index.search(p.query, p.top)
        if not hits:
            return fail(
                f"Nothing in the corpus matches {p.query!r}. Try fewer words, or names only."
            )
        lines = [
            f"{number}. {docid} — {index.title(docid) or docid}\n   {index.snippet(docid)}"
            for number, (docid, _) in enumerate(hits, start=1)
        ]
        return ok("\n".join(lines), hits=len(hits))

    return search
