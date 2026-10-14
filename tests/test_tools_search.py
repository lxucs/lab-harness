"""grep and glob against tmp_path."""

import os
import time
from pathlib import Path

from lab_harness.config import Config, SearchConfig, ToolsConfig
from lab_harness.llm.types import TextBlock
from lab_harness.tools.base import ToolContext, ToolResult
from lab_harness.tools.search import glob, grep


def _ctx(tmp_path: Path, config: Config | None = None) -> ToolContext:
    return ToolContext(cwd=str(tmp_path), call_id="c1", config=config or Config())


def _text(result: ToolResult) -> str:
    return "".join(b.text for b in result.content if isinstance(b, TextBlock))


def _with_search(**kwargs: object) -> Config:
    return Config(tools=ToolsConfig(search=SearchConfig(**kwargs)))  # type: ignore[arg-type]


def _tree(root: Path) -> None:
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("import os\n\n\ndef run():\n    return os.getcwd()\n")
    (root / "src" / "util.py").write_text("def helper():\n    return 42\n")
    (root / "README.md").write_text("# Project\n\nRun with `run()`.\n")


# --- grep -----------------------------------------------------------------


async def test_grep_returns_path_line_and_text(tmp_path: Path) -> None:
    _tree(tmp_path)

    result = await grep.execute({"pattern": r"def run"}, _ctx(tmp_path))

    assert not result.is_error
    assert "src/app.py:4:def run():" in _text(result)


async def test_grep_searches_recursively_from_the_given_path(tmp_path: Path) -> None:
    _tree(tmp_path)

    result = await grep.execute({"pattern": "def ", "path": "src"}, _ctx(tmp_path))

    body = _text(result)
    assert "app.py" in body
    assert "util.py" in body


async def test_grep_filters_by_glob(tmp_path: Path) -> None:
    _tree(tmp_path)

    result = await grep.execute({"pattern": "run", "glob": "*.md"}, _ctx(tmp_path))

    body = _text(result)
    assert "README.md" in body
    assert "app.py" not in body


async def test_grep_case_insensitive_is_opt_in(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Hello World\n")

    sensitive = await grep.execute({"pattern": "hello"}, _ctx(tmp_path))
    insensitive = await grep.execute({"pattern": "hello", "case_sensitive": False}, _ctx(tmp_path))

    assert "No matches" in _text(sensitive)
    assert "a.txt:1:Hello World" in _text(insensitive)


async def test_grep_reports_an_invalid_regex_actionably(tmp_path: Path) -> None:
    """The model wrote the pattern; it needs to know what was wrong."""
    result = await grep.execute({"pattern": "("}, _ctx(tmp_path))

    assert result.is_error
    assert "Invalid regular expression" in _text(result)


async def test_grep_skips_ignored_directories(tmp_path: Path) -> None:
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("needle\n")
    (tmp_path / "mine.js").write_text("needle\n")

    result = await grep.execute({"pattern": "needle"}, _ctx(tmp_path))

    body = _text(result)
    assert "mine.js" in body
    assert "node_modules" not in body


async def test_grep_skips_binary_files(tmp_path: Path) -> None:
    (tmp_path / "data.bin").write_bytes(b"needle\x00\x01binary")
    (tmp_path / "text.txt").write_text("needle\n")

    result = await grep.execute({"pattern": "needle"}, _ctx(tmp_path))

    body = _text(result)
    assert "text.txt" in body
    assert "data.bin" not in body


async def test_grep_caps_results_and_says_so(tmp_path: Path) -> None:
    (tmp_path / "many.txt").write_text("hit\n" * 100)
    config = _with_search(max_results=5)

    result = await grep.execute({"pattern": "hit"}, _ctx(tmp_path, config))

    assert result.details["truncated"] is True
    assert "stopped at 5 matches" in _text(result)
    assert len([ln for ln in _text(result).splitlines() if ":1:" in ln or ":2:" in ln]) <= 5


async def test_grep_clips_very_long_matching_lines(tmp_path: Path) -> None:
    (tmp_path / "long.txt").write_text("needle" + "x" * 1000 + "\n")
    config = _with_search(max_line_length=20)

    result = await grep.execute({"pattern": "needle"}, _ctx(tmp_path, config))

    assert "…" in _text(result)
    assert len(_text(result).splitlines()[0]) < 100


async def test_grep_reports_no_matches_without_erroring(tmp_path: Path) -> None:
    _tree(tmp_path)

    result = await grep.execute({"pattern": "zzz_absent"}, _ctx(tmp_path))

    assert not result.is_error
    assert "No matches" in _text(result)
    assert result.details["matches"] == 0


async def test_grep_can_search_a_single_file(tmp_path: Path) -> None:
    _tree(tmp_path)

    result = await grep.execute({"pattern": "def", "path": "src/util.py"}, _ctx(tmp_path))

    assert "util.py:1:def helper():" in _text(result)


async def test_grep_reports_a_missing_path(tmp_path: Path) -> None:
    result = await grep.execute({"pattern": "x", "path": "absent"}, _ctx(tmp_path))

    assert result.is_error
    assert "not found" in _text(result).lower()


# --- glob -----------------------------------------------------------------


async def test_glob_finds_files_by_pattern(tmp_path: Path) -> None:
    _tree(tmp_path)

    result = await glob.execute({"pattern": "**/*.py"}, _ctx(tmp_path))

    body = _text(result)
    assert "src/app.py" in body
    assert "src/util.py" in body
    assert "README.md" not in body


async def test_glob_sorts_newest_first(tmp_path: Path) -> None:
    """Recency beats alphabetical order in a repo someone is working in."""
    (tmp_path / "old.py").write_text("")
    (tmp_path / "new.py").write_text("")
    old = time.time() - 10_000
    os.utime(tmp_path / "old.py", (old, old))

    result = await glob.execute({"pattern": "*.py"}, _ctx(tmp_path))

    assert _text(result).splitlines() == ["new.py", "old.py"]


async def test_glob_skips_ignored_directories(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.py").write_text("")
    (tmp_path / "real.py").write_text("")

    result = await glob.execute({"pattern": "**/*.py"}, _ctx(tmp_path))

    body = _text(result)
    assert "real.py" in body
    assert ".git" not in body


async def test_glob_caps_results_and_says_so(tmp_path: Path) -> None:
    for i in range(20):
        (tmp_path / f"f{i}.py").write_text("")
    config = _with_search(max_files=5)

    result = await glob.execute({"pattern": "*.py"}, _ctx(tmp_path, config))

    assert "showing 5 of 20 matches" in _text(result)
    assert result.details["total_files"] == 20


async def test_glob_reports_no_matches_without_erroring(tmp_path: Path) -> None:
    result = await glob.execute({"pattern": "*.zzz"}, _ctx(tmp_path))

    assert not result.is_error
    assert "No files match" in _text(result)


async def test_glob_rejects_a_file_as_the_search_root(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("")

    result = await glob.execute({"pattern": "*", "path": "a.py"}, _ctx(tmp_path))

    assert result.is_error
    assert "not a directory" in _text(result)


async def test_glob_reports_a_missing_directory(tmp_path: Path) -> None:
    result = await glob.execute({"pattern": "*", "path": "absent"}, _ctx(tmp_path))

    assert result.is_error
    assert "not found" in _text(result).lower()


# --- shared ---------------------------------------------------------------


async def test_output_never_contains_absolute_paths(tmp_path: Path) -> None:
    """Transcripts get committed; machine layout should not ride along."""
    _tree(tmp_path)

    found = await glob.execute({"pattern": "**/*.py"}, _ctx(tmp_path))
    matched = await grep.execute({"pattern": "def"}, _ctx(tmp_path))

    assert str(tmp_path) not in _text(found)
    assert str(tmp_path) not in _text(matched)


async def test_both_tools_run_in_parallel(tmp_path: Path) -> None:
    """Read-only searches have no reason to serialize."""
    assert grep.execution_mode == "parallel"
    assert glob.execution_mode == "parallel"
