"""The search agent in `search_agent/`: a corpus, a query, what a run leaves.

Driven by a scripted model, the way `tests/test_examples.py` drives the
examples — so this needs no key and touches no network. What it checks is the
program's own work: ranking the corpus, writing the trajectory and the report,
and keeping going when one query cannot run.
"""

import json
from pathlib import Path

import pytest

from lab_harness.config import load_config
from lab_harness.llm.fake import FAKE_MODEL, FakeProvider, text_response, tool_call_response
from lab_harness.llm.types import TextBlock, ToolCallBlock
from lab_harness.tools import ToolContext, ToolResult
from search_agent import cli
from search_agent import run as program
from search_agent.settings import Settings, load_settings
from search_agent.tools import corpus_search

QUERY = {
    "query_id": "793",
    "query": "What day of the year were both children born on?",
    "answer": "29/02",
    "gold_docids": ["1"],
    "evidence_docids": ["1"],
    "negative_docids": ["2"],
}


def corpus(directory: Path) -> Path:
    """Two documents: one that answers, one that sounds as if it might."""
    docs = directory / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "1.md").write_text(
        "---\ntitle: Leap Day baby\n---\n"
        "A couple welcomed a second child born on 29 February at Coney Island.",
        encoding="utf-8",
    )
    (docs / "2.md").write_text(
        "---\ntitle: A baby in October\n---\nA singer welcomed a baby boy on 29 October.",
        encoding="utf-8",
    )
    return docs


def settings(directory: Path) -> Settings:
    return Settings(
        request="Answer the query below.\n\n{query}",
        cwd=directory,
        docs=directory / "docs",
        queries=directory / "queries.jsonl",
        out=directory / "runs",
        tools=("ls", "search_docs"),
    )


def scripted(monkeypatch: pytest.MonkeyPatch, *responses: object) -> FakeProvider:
    provider = FakeProvider(responses)  # type: ignore[arg-type]
    monkeypatch.setattr(program, "route", lambda model_id, config: (FAKE_MODEL, provider))
    return provider


def searching(query: str) -> ToolCallBlock:
    return ToolCallBlock(id="call-1", name="search_docs", arguments={"query": query})


def joined(result: ToolResult) -> str:
    """The tool's output as one string."""
    return "".join(block.text for block in result.content if isinstance(block, TextBlock))


# --- ranking the corpus ---------------------------------------------------


async def test_search_docs_ranks_the_document_that_answers(tmp_path: Path) -> None:
    corpus(tmp_path)
    result = await corpus_search("docs").execute(
        {"query": "leap day baby born February"}, ToolContext(cwd=str(tmp_path), call_id="c")
    )
    assert not result.is_error
    text = joined(result)
    assert "1. 1 —" in text
    assert "2. 2 —" in text


async def test_search_docs_says_so_when_nothing_matches(tmp_path: Path) -> None:
    corpus(tmp_path)
    result = await corpus_search("docs").execute(
        {"query": "xylophone"}, ToolContext(cwd=str(tmp_path), call_id="c")
    )
    assert result.is_error
    assert "Nothing in the corpus matches" in joined(result)


async def test_search_docs_says_so_when_there_is_no_corpus(tmp_path: Path) -> None:
    result = await corpus_search("docs").execute(
        {"query": "anything"}, ToolContext(cwd=str(tmp_path / "empty"), call_id="c")
    )
    assert result.is_error
    assert "no docs/ directory" in joined(result)


# --- one query ------------------------------------------------------------


async def test_a_query_leaves_a_trajectory_and_a_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus(tmp_path)
    scripted(monkeypatch, tool_call_response([searching("leap day")]), text_response("29/02"))

    outcome = await program.run_query(QUERY, settings(tmp_path), load_config())

    assert outcome.answer == "29/02"
    assert outcome.requests == 2
    assert outcome.tool_calls == 1
    directory = tmp_path / "runs" / "793"
    assert (directory / "793.jsonl").exists()

    trajectory = json.loads((directory / "trajectory.json").read_text(encoding="utf-8"))
    assert trajectory["query_id"] == "793"
    assert trajectory["gold"] == "29/02"
    assert trajectory["docs"]["negatives"] == ["2"]
    assert trajectory["request"] == "Answer the query below.\n\n" + QUERY["query"]
    assert trajectory["summary"]["requests"] == 2
    assert trajectory["summary"]["tool_calls"] == 1
    assert trajectory["summary"]["input"] > 0
    assert trajectory["answer"] == "29/02"
    assert trajectory["session"] == {
        "id": "793",
        "file": "793.jsonl",
        "report": "report.html",
    }
    assert [message["role"] for message in trajectory["messages"]] == [
        "user",
        "assistant",
        "tool_result",
        "assistant",
    ]

    report = (directory / trajectory["session"]["report"]).read_text(encoding="utf-8")
    assert QUERY["query"] in report


def test_the_tool_set_is_the_configured_builtins_plus_the_search() -> None:
    assert [t.name for t in program.tools_for(("ls", "search_docs"), "docs")] == [
        "ls",
        "search_docs",
    ]
    assert [t.name for t in program.tools_for(("ls",), "docs")] == ["ls"]


async def test_running_a_query_again_replaces_its_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-run is a new run, not a second half appended to the last one."""
    corpus(tmp_path)
    scripted(
        monkeypatch,
        tool_call_response([searching("leap day")]),
        text_response("29/02"),
        tool_call_response([searching("leap day")]),
        text_response("29/02"),
    )
    chosen = settings(tmp_path)

    await program.run_query(QUERY, chosen, load_config())
    await program.run_query(QUERY, chosen, load_config())

    session = (tmp_path / "runs" / "793" / "793.jsonl").read_text(encoding="utf-8")
    assert session.count('"type": "context"') == 1


# --- the run --------------------------------------------------------------


async def test_the_cli_runs_the_queries_in_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus(tmp_path)
    queries = tmp_path / "queries.jsonl"
    queries.write_text(json.dumps(QUERY) + "\n", encoding="utf-8")
    scripted(monkeypatch, tool_call_response([searching("leap day")]), text_response("29/02"))

    code = await cli.main(["--queries", str(queries), "--out-dir", str(tmp_path / "runs")])

    assert code == 0
    assert (tmp_path / "runs" / "793" / "trajectory.json").exists()
    out = capsys.readouterr().out
    assert "1/1 run" in out
    assert "29/02" in out


async def test_a_query_that_cannot_run_does_not_end_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A malformed line is reported and the rest of the file still runs."""
    corpus(tmp_path)
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps({"query_id": "bad"}) + "\n" + json.dumps(QUERY) + "\n", encoding="utf-8"
    )
    scripted(monkeypatch, tool_call_response([searching("leap day")]), text_response("29/02"))

    code = await cli.main(["--queries", str(queries), "--out-dir", str(tmp_path / "runs")])

    out = capsys.readouterr().out
    assert code == 1
    assert "KeyError" in out
    assert (tmp_path / "runs" / "793" / "trajectory.json").exists()


async def test_the_cli_reports_a_query_file_that_is_not_there(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert await cli.main(["--queries", str(tmp_path / "none.jsonl")]) == 1
    assert "no query file" in capsys.readouterr().err


async def test_a_config_that_is_not_a_file_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert await cli.main(["--config", str(tmp_path / "none.toml")]) == 1
    assert "is not a file" in capsys.readouterr().err


# --- settings -------------------------------------------------------------


def test_the_shipped_config_asks_with_the_query_in_it() -> None:
    settings = load_settings()
    assert "Q1" in settings.first_turn("Q1")
    assert settings.tools == ("ls", "glob", "grep", "read", "search_docs")


def test_a_config_file_supplies_the_request_and_the_paths(tmp_path: Path) -> None:
    path = tmp_path / "search_agent.toml"
    path.write_text(
        '[request]\ntemplate = "Do {query} now"\n\n'
        '[run]\nqueries = "q.jsonl"\nout_dir = "out"\ntools = ["read"]\n',
        encoding="utf-8",
    )
    settings = load_settings(path)
    assert settings.first_turn("it") == "Do it now"
    assert settings.queries == tmp_path / "q.jsonl"
    assert settings.out == tmp_path / "out"
    assert settings.tools == ("read",)


def test_a_missing_config_file_means_the_defaults(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "not-written.toml")
    assert settings.tools == ("ls", "glob", "grep", "read", "search_docs")
    assert "{query}" not in settings.first_turn("Q1")
