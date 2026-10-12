"""Guarantees about the source files themselves.

These exist because the normal suite cannot catch them: Python compiles each
module once and caches the bytecode, so a `SyntaxWarning` fires on the first
run after an edit and is silent on every run after that — including in CI,
whenever the cache is warm. Compiling fresh here makes them deterministic.
"""

import warnings
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "harness"


def sources() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def test_the_package_is_found() -> None:
    """Guards against the glob silently matching nothing."""
    assert len(sources()) > 20


@pytest.mark.parametrize("path", sources(), ids=lambda p: p.name)
def test_compiles_without_warnings(path: Path) -> None:
    """No invalid escape sequences, chiefly.

    An ASCII diagram in a docstring is the easy way to introduce one — a `\\`
    before a space is not an escape Python knows, and the module still imports.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compile(path.read_text(), str(path), "exec")
