"""Where sessions live on disk, and how to find them.

Layout:

    <home>/sessions/<project-slug>/<session-id>.jsonl

Sessions are grouped by project so `--resume` can mean "the last session *here*"
without scanning every session ever recorded. The slug is derived from the
working directory and carries a hash, because two checkouts of the same
repository in different places are different projects and must not share a
history.
"""

import hashlib
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from lab_harness.config import SessionConfig, lab_harness_home

_SLUG_UNSAFE = re.compile(r"[^a-zA-Z0-9._-]+")
#: Enough hash to make collisions implausible, short enough to stay readable.
_SLUG_HASH_LENGTH = 8
_MAX_SLUG_NAME = 40


def home(config: SessionConfig) -> Path:
    """The harness home directory, honoring the configured env override.

    Defined in `config` because `load_config` needs it to locate the user
    config file, and config sits below this module.
    """
    return lab_harness_home(config)


def project_slug(cwd: str | Path) -> str:
    """A filesystem-safe, collision-resistant name for a working directory.

    The directory name alone is not enough: `~/work/api` and `~/fork/api` are
    different projects. The hash of the full resolved path disambiguates them.
    """
    resolved = Path(cwd).expanduser().resolve()
    digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:_SLUG_HASH_LENGTH]
    name = _SLUG_UNSAFE.sub("-", resolved.name).strip("-") or "root"
    return f"{name[:_MAX_SLUG_NAME]}-{digest}"


def project_directory(cwd: str | Path, config: SessionConfig) -> Path:
    return home(config) / config.directory_name / project_slug(cwd)


def new_session_id() -> str:
    """A sortable, human-readable id: timestamp plus random suffix.

    Sortable so "most recent" is a filename comparison rather than a stat of
    every candidate; random-suffixed so two sessions started in the same second
    cannot collide.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{os.urandom(3).hex()}"


def session_path(cwd: str | Path, session_id: str, config: SessionConfig) -> Path:
    return project_directory(cwd, config) / f"{session_id}.jsonl"


def list_sessions(cwd: str | Path, config: SessionConfig) -> list[Path]:
    """Every session for this project, newest first.

    Sorted by name rather than mtime: ids lead with a UTC timestamp, so the
    name ordering is the chronological one and does not require stat-ing files
    that may be on a slow filesystem.
    """
    directory = project_directory(cwd, config)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.jsonl"), key=lambda p: p.name, reverse=True)


def latest_session(cwd: str | Path, config: SessionConfig) -> Path | None:
    sessions = list_sessions(cwd, config)
    return sessions[0] if sessions else None


def find_session(cwd: str | Path, session_id: str, config: SessionConfig) -> Path | None:
    """Resolve a session id to a path, accepting a unique prefix.

    Ids are long enough to be unpleasant to type in full, and a prefix is
    unambiguous in practice. An ambiguous prefix is an error rather than a
    guess — resuming the wrong conversation is worse than being asked again.
    """
    exact = session_path(cwd, session_id, config)
    if exact.exists():
        return exact
    matches = [p for p in list_sessions(cwd, config) if p.stem.startswith(session_id)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(p.stem for p in matches[:5])
        raise ValueError(f"Session id {session_id!r} is ambiguous: {names}")
    return None
