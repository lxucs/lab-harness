"""Project instructions loaded from the working directory.

What matters here is not "a file is read" but the rules around it: which file
wins when several could, that the cap is enforced *before* the file reaches
memory, and that an unreadable file is loud rather than silently ignored —
instructions that quietly fail to apply are the whole failure mode this
feature has.
"""

import os
import sys
from pathlib import Path
from typing import BinaryIO, cast

import pytest

from lab_harness.config import ConfigError, ProjectConfig
from lab_harness.project import load_project_context

#: Permission bits mean nothing to root, so the unreadable-file tests cannot
#: run there. Containers that run the suite as root skip them rather than
#: reporting a pass they did not earn.
unreadable_files_work = pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs POSIX permission bits and a non-root user",
)


def write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


# --- finding the file -----------------------------------------------------


def test_no_file_means_no_context(tmp_path: Path) -> None:
    """The common case: most directories have neither file."""
    assert load_project_context(tmp_path) is None


def test_harness_md_is_loaded(tmp_path: Path) -> None:
    write(tmp_path / "HARNESS.md", "Use tabs.")
    context = load_project_context(tmp_path)
    assert context is not None
    assert context.text == "Use tabs."
    assert context.path.name == "HARNESS.md"
    assert context.truncated is False


def test_agents_md_is_loaded_when_harness_md_is_absent(tmp_path: Path) -> None:
    write(tmp_path / "AGENTS.md", "Run the tests.")
    context = load_project_context(tmp_path)
    assert context is not None
    assert context.path.name == "AGENTS.md"


def test_the_first_configured_name_wins(tmp_path: Path) -> None:
    """Both present is not an error; precedence is the configured order."""
    write(tmp_path / "HARNESS.md", "first")
    write(tmp_path / "AGENTS.md", "second")
    context = load_project_context(tmp_path)
    assert context is not None
    assert context.text == "first"


def test_precedence_follows_config_not_the_filenames(tmp_path: Path) -> None:
    write(tmp_path / "HARNESS.md", "first")
    write(tmp_path / "AGENTS.md", "second")
    reversed_order = ProjectConfig(context_files=("AGENTS.md", "HARNESS.md"))
    context = load_project_context(tmp_path, reversed_order)
    assert context is not None
    assert context.text == "second"


def test_a_custom_filename_is_honored(tmp_path: Path) -> None:
    write(tmp_path / "CONVENTIONS.md", "house style")
    config = ProjectConfig(context_files=("CONVENTIONS.md",))
    context = load_project_context(tmp_path, config)
    assert context is not None
    assert context.text == "house style"


def test_an_empty_context_file_list_loads_nothing(tmp_path: Path) -> None:
    write(tmp_path / "HARNESS.md", "ignored")
    assert load_project_context(tmp_path, ProjectConfig(context_files=())) is None


def test_a_directory_with_the_right_name_is_not_a_file(tmp_path: Path) -> None:
    """`AGENTS.md/` as a directory must not raise — it is simply not present."""
    (tmp_path / "HARNESS.md").mkdir()
    assert load_project_context(tmp_path) is None


def test_a_directory_does_not_shadow_a_real_file(tmp_path: Path) -> None:
    (tmp_path / "HARNESS.md").mkdir()
    write(tmp_path / "AGENTS.md", "the real one")
    context = load_project_context(tmp_path)
    assert context is not None
    assert context.text == "the real one"


# --- blank files ----------------------------------------------------------


def test_a_blank_file_yields_no_context(tmp_path: Path) -> None:
    write(tmp_path / "HARNESS.md", "   \n\n\t\n")
    assert load_project_context(tmp_path) is None


def test_a_blank_first_file_does_not_fall_through(tmp_path: Path) -> None:
    """Documented behavior: an empty HARNESS.md opts the project out entirely.

    Falling through would add a special case to a one-line rule, and would
    take away the only way to suppress an AGENTS.md written for another tool.
    """
    write(tmp_path / "HARNESS.md", "")
    write(tmp_path / "AGENTS.md", "not reached")
    assert load_project_context(tmp_path) is None


def test_surrounding_whitespace_is_stripped(tmp_path: Path) -> None:
    write(tmp_path / "HARNESS.md", "\n\n  Use tabs.  \n\n")
    context = load_project_context(tmp_path)
    assert context is not None
    assert context.text == "Use tabs."


# --- the size cap ---------------------------------------------------------


def test_a_long_file_is_truncated_and_says_so(tmp_path: Path) -> None:
    write(tmp_path / "HARNESS.md", "x" * 500)
    context = load_project_context(tmp_path, ProjectConfig(max_bytes=100))
    assert context is not None
    assert context.truncated is True
    assert context.text == "x" * 100


def test_a_file_exactly_at_the_cap_is_not_truncated(tmp_path: Path) -> None:
    """Off-by-one guard: `max_bytes` is a limit, not the first rejected size."""
    write(tmp_path / "HARNESS.md", "x" * 100)
    context = load_project_context(tmp_path, ProjectConfig(max_bytes=100))
    assert context is not None
    assert context.truncated is False
    assert context.text == "x" * 100


class RecordingHandle:
    """A file handle that remembers the sizes `read()` was asked for."""

    def __init__(self, handle: BinaryIO, sizes: list[int]) -> None:
        self._handle = handle
        self._sizes = sizes

    def read(self, size: int = -1) -> bytes:
        self._sizes.append(size)
        return self._handle.read(size)

    def __enter__(self) -> "RecordingHandle":
        return self

    def __exit__(self, *exc: object) -> None:
        self._handle.close()


def test_the_cap_is_applied_before_reading_the_whole_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of the cap is bounded memory, so the read itself must be bounded.

    Reading the file whole and slicing afterwards would pass every other test
    in this module while loading a gigabyte in order to keep a kilobyte.
    """
    path = write(tmp_path / "HARNESS.md", "y" * 10_000)
    sizes: list[int] = []
    real_open = Path.open

    def recording_open(self: Path, *args: object, **kwargs: object) -> object:
        handle = real_open(self, *args, **kwargs)  # type: ignore[arg-type]
        if self != path:
            return handle
        return RecordingHandle(cast(BinaryIO, handle), sizes)

    monkeypatch.setattr(Path, "open", recording_open)
    load_project_context(tmp_path, ProjectConfig(max_bytes=100))

    # One bounded read of max_bytes + 1 — the extra byte is how truncation is
    # detected without a second read or a stat.
    assert sizes == [101]


def test_a_multibyte_character_split_by_the_cap_does_not_crash(tmp_path: Path) -> None:
    """The cut lands mid-codepoint; a replacement character beats a traceback."""
    write(tmp_path / "HARNESS.md", "é" * 50)  # two bytes each
    context = load_project_context(tmp_path, ProjectConfig(max_bytes=9))
    assert context is not None
    assert context.text.startswith("é" * 4)
    assert context.truncated is True


# --- failure --------------------------------------------------------------


@unreadable_files_work
def test_an_unreadable_file_raises_rather_than_being_skipped(tmp_path: Path) -> None:
    """Silently dropping instructions is the failure this must never have."""
    path = write(tmp_path / "HARNESS.md", "secret conventions")
    path.chmod(0o000)
    try:
        with pytest.raises(ConfigError, match=r"cannot read project instructions"):
            load_project_context(tmp_path)
    finally:
        path.chmod(0o644)


@unreadable_files_work
def test_the_error_names_the_file(tmp_path: Path) -> None:
    path = write(tmp_path / "HARNESS.md", "secret conventions")
    path.chmod(0o000)
    try:
        with pytest.raises(ConfigError, match=r"HARNESS\.md"):
            load_project_context(tmp_path)
    finally:
        path.chmod(0o644)


# --- defaults -------------------------------------------------------------


def test_the_default_config_is_used_when_none_is_passed(tmp_path: Path) -> None:
    """Callers outside the CLI get the documented names without wiring config."""
    write(tmp_path / ProjectConfig().context_files[0], "default path works")
    context = load_project_context(tmp_path)
    assert context is not None
    assert context.text == "default path works"


def test_cwd_defaults_to_the_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write(tmp_path / "HARNESS.md", "found via cwd")
    monkeypatch.chdir(tmp_path)
    context = load_project_context()
    assert context is not None
    assert context.text == "found via cwd"
