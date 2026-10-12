"""Filesystem tools against tmp_path. Nothing here touches the real cwd."""

from dataclasses import replace
from pathlib import Path

from harness.config import Config, ListConfig, ReadConfig, ToolsConfig
from harness.llm.types import TextBlock
from harness.tools.base import ToolContext, ToolResult
from harness.tools.fs import edit, ls, read, write


def _ctx(tmp_path: Path, config: Config | None = None) -> ToolContext:
    return ToolContext(cwd=str(tmp_path), call_id="c1", config=config or Config())


def _text(result: ToolResult) -> str:
    return "".join(b.text for b in result.content if isinstance(b, TextBlock))


# --- read -----------------------------------------------------------------


async def test_read_returns_numbered_lines(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n")

    result = await read.execute({"path": "a.txt"}, _ctx(tmp_path))

    assert not result.is_error
    assert _text(result) == "1\tone\n2\ttwo\n3\tthree"


async def test_read_resolves_relative_to_ctx_cwd_not_the_process(tmp_path: Path) -> None:
    """This is what makes tools testable and multi-root safe."""
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "a.txt").write_text("found\n")

    result = await read.execute({"path": "a.txt"}, _ctx(nested))

    assert _text(result) == "1\tfound"


async def test_read_honors_offset_and_limit(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("\n".join(str(i) for i in range(1, 11)))

    result = await read.execute({"path": "a.txt", "offset": 3, "limit": 2}, _ctx(tmp_path))

    assert _text(result) == "3\t3\n4\t4\n\n[truncated at 4 of 10 lines; re-read with offset=5]"


async def test_read_truncation_note_points_at_the_next_offset(tmp_path: Path) -> None:
    """The hint must be directly usable as the next call's offset."""
    (tmp_path / "a.txt").write_text("\n".join(str(i) for i in range(1, 6)))
    limits = ToolsConfig(read=ReadConfig(default_limit=2))

    first = await read.execute({"path": "a.txt"}, _ctx(tmp_path, Config(tools=limits)))
    assert "offset=3" in _text(first)

    second = await read.execute(
        {"path": "a.txt", "offset": 3}, _ctx(tmp_path, Config(tools=limits))
    )
    assert _text(second).startswith("3\t3\n4\t4")


async def test_read_clips_very_long_lines(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x" * 50)
    config = Config(tools=ToolsConfig(read=ReadConfig(max_line_length=10)))

    result = await read.execute({"path": "a.txt"}, _ctx(tmp_path, config))

    assert "line truncated at 10 characters" in _text(result)
    assert "x" * 11 not in _text(result)


async def test_read_refuses_files_over_the_byte_limit(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("x" * 100)
    config = Config(tools=ToolsConfig(read=ReadConfig(max_bytes=50)))

    result = await read.execute({"path": "big.txt"}, _ctx(tmp_path, config))

    assert result.is_error
    assert "over the 50-byte limit" in _text(result)


async def test_read_reports_a_missing_file(tmp_path: Path) -> None:
    result = await read.execute({"path": "nope.txt"}, _ctx(tmp_path))

    assert result.is_error
    assert "not found" in _text(result).lower()


async def test_read_points_at_ls_for_a_directory(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()

    result = await read.execute({"path": "sub"}, _ctx(tmp_path))

    assert result.is_error
    assert "ls" in _text(result)


async def test_read_reports_binary_files_rather_than_crashing(tmp_path: Path) -> None:
    (tmp_path / "b.bin").write_bytes(b"\xff\xfe\x00\x01")

    result = await read.execute({"path": "b.bin"}, _ctx(tmp_path))

    assert result.is_error
    assert "not UTF-8" in _text(result)


async def test_read_handles_an_empty_file(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text("")

    result = await read.execute({"path": "empty.txt"}, _ctx(tmp_path))

    assert not result.is_error
    assert _text(result) == "[file is empty]"


async def test_read_rejects_an_offset_past_the_end(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("one\ntwo\n")

    result = await read.execute({"path": "a.txt", "offset": 99}, _ctx(tmp_path))

    assert result.is_error
    assert "past the end" in _text(result)


async def test_read_output_does_not_leak_absolute_paths(tmp_path: Path) -> None:
    """Transcripts get committed; machine layout should not ride along."""
    result = await read.execute({"path": "nope.txt"}, _ctx(tmp_path))

    assert str(tmp_path) not in _text(result)


# --- write ----------------------------------------------------------------


async def test_write_creates_a_file_and_its_parents(tmp_path: Path) -> None:
    result = await write.execute(
        {"path": "deep/nested/a.txt", "content": "hello\n"}, _ctx(tmp_path)
    )

    assert not result.is_error
    assert (tmp_path / "deep/nested/a.txt").read_text() == "hello\n"
    assert result.details["created"] is True


async def test_write_overwrites_and_says_so(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("old")

    result = await write.execute({"path": "a.txt", "content": "new"}, _ctx(tmp_path))

    assert (tmp_path / "a.txt").read_text() == "new"
    assert result.details["created"] is False
    assert "Updated" in _text(result)


async def test_write_refuses_a_directory(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()

    result = await write.execute({"path": "sub", "content": "x"}, _ctx(tmp_path))

    assert result.is_error


def test_write_and_edit_are_sequential() -> None:
    """Concurrent mutation of one path is not left to chance."""
    assert write.execution_mode == "sequential"
    assert edit.execution_mode == "sequential"


# --- edit -----------------------------------------------------------------


async def test_edit_replaces_exact_text(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n")

    result = await edit.execute({"path": "a.py", "old": "y = 2", "new": "y = 3"}, _ctx(tmp_path))

    assert not result.is_error
    assert (tmp_path / "a.py").read_text() == "x = 1\ny = 3\n"


async def test_edit_refuses_ambiguous_matches(tmp_path: Path) -> None:
    """Replacing the first of several matches edits a line nobody looked at."""
    (tmp_path / "a.py").write_text("v = 1\nv = 1\n")

    result = await edit.execute({"path": "a.py", "old": "v = 1", "new": "v = 2"}, _ctx(tmp_path))

    assert result.is_error
    assert "appears 2 times" in _text(result)
    assert (tmp_path / "a.py").read_text() == "v = 1\nv = 1\n"


async def test_edit_replace_all_is_opt_in(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("v = 1\nv = 1\n")

    result = await edit.execute(
        {"path": "a.py", "old": "v = 1", "new": "v = 2", "replace_all": True}, _ctx(tmp_path)
    )

    assert not result.is_error
    assert (tmp_path / "a.py").read_text() == "v = 2\nv = 2\n"
    assert result.details["replacements"] == 2


async def test_edit_reports_missing_text_actionably(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")

    result = await edit.execute({"path": "a.py", "old": "absent", "new": "y"}, _ctx(tmp_path))

    assert result.is_error
    assert "not found" in _text(result)
    assert "exactly" in _text(result)


async def test_edit_rejects_identical_old_and_new(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")

    result = await edit.execute({"path": "a.py", "old": "x", "new": "x"}, _ctx(tmp_path))

    assert result.is_error
    assert "identical" in _text(result)


async def test_edit_reports_a_missing_file(tmp_path: Path) -> None:
    result = await edit.execute({"path": "nope.py", "old": "a", "new": "b"}, _ctx(tmp_path))

    assert result.is_error
    assert "not found" in _text(result)


# --- ls -------------------------------------------------------------------


async def test_ls_lists_directories_first_and_marks_them(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "a_dir").mkdir()

    result = await ls.execute({}, _ctx(tmp_path))

    assert _text(result) == "a_dir/\nb.txt"


async def test_ls_hides_dotfiles_unless_asked(tmp_path: Path) -> None:
    (tmp_path / ".hidden").write_text("")
    (tmp_path / "shown.txt").write_text("")

    default = await ls.execute({}, _ctx(tmp_path))
    with_all = await ls.execute({"all": True}, _ctx(tmp_path))

    assert _text(default) == "shown.txt"
    assert ".hidden" in _text(with_all)


async def test_ls_truncates_at_the_configured_limit(tmp_path: Path) -> None:
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("")
    config = Config(tools=ToolsConfig(list=ListConfig(max_entries=2)))

    result = await ls.execute({}, _ctx(tmp_path, config))

    assert "[truncated at 2 of 5 entries]" in _text(result)


async def test_ls_reports_an_empty_directory(tmp_path: Path) -> None:
    result = await ls.execute({}, _ctx(tmp_path))

    assert _text(result) == "[empty]"


async def test_ls_points_at_read_for_a_file(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("")

    result = await ls.execute({"path": "a.txt"}, _ctx(tmp_path))

    assert result.is_error
    assert "read" in _text(result)


async def test_ls_reports_a_missing_directory(tmp_path: Path) -> None:
    result = await ls.execute({"path": "nope"}, _ctx(tmp_path))

    assert result.is_error
    assert "not found" in _text(result).lower()


# --- shared ---------------------------------------------------------------


async def test_absolute_paths_are_honored(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.txt"
    target.write_text("abs\n")
    other = tmp_path / "other"
    other.mkdir()

    result = await read.execute({"path": str(target)}, _ctx(other))

    assert _text(result) == "1\tabs"


async def test_every_filesystem_tool_declares_a_json_schema() -> None:
    for item in (read, write, edit, ls):
        assert item.parameters["type"] == "object"
        assert "properties" in item.parameters


async def test_config_is_threaded_through_rather_than_read_globally(tmp_path: Path) -> None:
    """Two contexts with different limits must behave differently."""
    (tmp_path / "a.txt").write_text("1\n2\n3\n")
    tight = Config(tools=ToolsConfig(read=ReadConfig(default_limit=1)))
    loose = replace(tight, tools=ToolsConfig(read=ReadConfig(default_limit=10)))

    truncated = await read.execute({"path": "a.txt"}, _ctx(tmp_path, tight))
    full = await read.execute({"path": "a.txt"}, _ctx(tmp_path, loose))

    assert "truncated" in _text(truncated)
    assert "truncated" not in _text(full)
