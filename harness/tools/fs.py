"""Filesystem tools: read, write, edit, ls.

Every path is resolved against `ctx.cwd`, never the process working directory —
that is what lets the suite run them against `tmp_path` and what keeps two
agents with different roots from interfering.

Errors are returned, not raised. The messages are written for the model: they
say what went wrong and what to do instead, because the model's next action is
chosen from that text.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from harness.tools.base import ToolContext, ToolResult, fail, ok, tool

_TRUNCATION_NOTE = "\n\n[truncated at {shown} of {total} lines; re-read with offset={next_offset}]"
_LINE_CUT = "… [line truncated at {limit} characters]"


def resolve(path: str, ctx: ToolContext) -> Path:
    """Resolve a tool-supplied path against the call's working directory.

    Absolute paths are honored as given. `expanduser` runs first so `~/x` means
    what the user expects rather than a literal directory named `~`.
    """
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(ctx.cwd).expanduser() / candidate
    return candidate


def _display(path: Path, ctx: ToolContext) -> str:
    """Render a path for the model: relative to cwd when it is inside it.

    Absolute paths leak the machine's layout into the transcript, which is both
    noisy and the kind of thing that ends up committed.
    """
    try:
        return str(path.relative_to(Path(ctx.cwd).expanduser().resolve()))
    except ValueError:
        return str(path)


# --- read -----------------------------------------------------------------


class ReadParams(BaseModel):
    path: str = Field(description="Path to the file, absolute or relative to the working directory")
    offset: int | None = Field(
        default=None, ge=1, description="1-indexed line to start from. Omit to start at the top."
    )
    limit: int | None = Field(default=None, ge=1, description="Maximum number of lines to return.")


@tool(name="read", description="Read a text file. Returns 1-indexed numbered lines.")
async def read(p: ReadParams, ctx: ToolContext) -> ToolResult:
    path = resolve(p.path, ctx)
    limits = ctx.config.tools.read

    if not path.exists():
        return fail(f"File not found: {_display(path, ctx)}")
    if path.is_dir():
        return fail(f"{_display(path, ctx)} is a directory. Use `ls` to list it.")

    size = path.stat().st_size
    if size > limits.max_bytes:
        return fail(
            f"{_display(path, ctx)} is {size} bytes, over the {limits.max_bytes}-byte "
            f"limit. Read a range with offset and limit, or use grep to find what you need."
        )

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return fail(f"{_display(path, ctx)} is not UTF-8 text. It may be a binary file.")
    except OSError as exc:
        return fail(f"Could not read {_display(path, ctx)}: {exc}")

    lines = content.splitlines()
    start = (p.offset or 1) - 1
    if start >= len(lines) and lines:
        return fail(f"offset {p.offset} is past the end of the file ({len(lines)} lines).")

    count = p.limit or limits.default_limit
    window = lines[start : start + count]

    rendered = "\n".join(
        f"{start + i + 1}\t{_clip(line, limits.max_line_length)}" for i, line in enumerate(window)
    )
    shown = start + len(window)
    if shown < len(lines):
        rendered += _TRUNCATION_NOTE.format(shown=shown, total=len(lines), next_offset=shown + 1)
    if not lines:
        rendered = "[file is empty]"

    return ok(rendered, path=str(path), lines=len(window), total_lines=len(lines))


def _clip(line: str, limit: int) -> str:
    return line if len(line) <= limit else line[:limit] + _LINE_CUT.format(limit=limit)


# --- write ----------------------------------------------------------------


class WriteParams(BaseModel):
    path: str = Field(description="Path to write, absolute or relative to the working directory")
    content: str = Field(description="Full file contents. Overwrites any existing file.")


@tool(
    name="write",
    description="Write a file, creating parent directories as needed. Overwrites existing files.",
    # Sequential: concurrent writes to one path are a race nobody wants to debug.
    execution_mode="sequential",
)
async def write(p: WriteParams, ctx: ToolContext) -> ToolResult:
    path = resolve(p.path, ctx)
    if path.is_dir():
        return fail(f"{_display(path, ctx)} is a directory.")

    existed = path.exists()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(p.content, encoding="utf-8")
    except OSError as exc:
        return fail(f"Could not write {_display(path, ctx)}: {exc}")

    verb = "Updated" if existed else "Created"
    line_count = len(p.content.splitlines())
    return ok(
        f"{verb} {_display(path, ctx)} ({line_count} lines)",
        path=str(path),
        created=not existed,
        lines=line_count,
    )


# --- edit -----------------------------------------------------------------


class EditParams(BaseModel):
    path: str = Field(description="Path to edit, absolute or relative to the working directory")
    old: str = Field(
        description="Exact text to replace, including indentation. Must appear exactly once "
        "unless replace_all is set."
    )
    new: str = Field(description="Replacement text.")
    replace_all: bool = Field(default=False, description="Replace every occurrence of `old`.")


@tool(
    name="edit",
    description="Replace exact text in a file. Fails if the text is absent or ambiguous.",
    execution_mode="sequential",
)
async def edit(p: EditParams, ctx: ToolContext) -> ToolResult:
    path = resolve(p.path, ctx)
    if not path.exists():
        return fail(f"File not found: {_display(path, ctx)}")
    if path.is_dir():
        return fail(f"{_display(path, ctx)} is a directory.")
    if p.old == p.new:
        return fail("`old` and `new` are identical; nothing to do.")

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return fail(f"{_display(path, ctx)} is not UTF-8 text.")
    except OSError as exc:
        return fail(f"Could not read {_display(path, ctx)}: {exc}")

    occurrences = content.count(p.old)
    if occurrences == 0:
        return fail(
            f"`old` text not found in {_display(path, ctx)}. Read the file and copy the "
            f"text exactly, including whitespace."
        )
    # Ambiguity is refused rather than guessed: replacing the first of several
    # matches edits a line the caller never looked at.
    if occurrences > 1 and not p.replace_all:
        return fail(
            f"`old` text appears {occurrences} times in {_display(path, ctx)}. Add "
            f"surrounding context to make it unique, or set replace_all."
        )

    updated = content.replace(p.old, p.new) if p.replace_all else content.replace(p.old, p.new, 1)
    try:
        path.write_text(updated, encoding="utf-8")
    except OSError as exc:
        return fail(f"Could not write {_display(path, ctx)}: {exc}")

    replaced = occurrences if p.replace_all else 1
    return ok(
        f"Replaced {replaced} occurrence(s) in {_display(path, ctx)}",
        path=str(path),
        replacements=replaced,
    )


# --- ls -------------------------------------------------------------------


class ListParams(BaseModel):
    path: str = Field(default=".", description="Directory to list. Defaults to the working dir.")
    all: bool = Field(default=False, description="Include entries beginning with a dot.")


@tool(name="ls", description="List the entries of a directory. Directories are marked with /.")
async def ls(p: ListParams, ctx: ToolContext) -> ToolResult:
    path = resolve(p.path, ctx)
    if not path.exists():
        return fail(f"Directory not found: {_display(path, ctx)}")
    if not path.is_dir():
        return fail(f"{_display(path, ctx)} is a file. Use `read` to read it.")

    try:
        entries = sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name))
    except OSError as exc:
        return fail(f"Could not list {_display(path, ctx)}: {exc}")

    if not p.all:
        entries = [e for e in entries if not e.name.startswith(".")]

    limit = ctx.config.tools.list.max_entries
    shown = entries[:limit]
    names = [f"{e.name}/" if e.is_dir() else e.name for e in shown]
    rendered = "\n".join(names) if names else "[empty]"
    if len(entries) > limit:
        rendered += f"\n[truncated at {limit} of {len(entries)} entries]"

    return ok(rendered, path=str(path), entries=len(shown), total_entries=len(entries))


FILESYSTEM_TOOLS = (read, write, edit, ls)
