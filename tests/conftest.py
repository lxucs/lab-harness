"""Shared fixtures.

Two autouse guards make the suite safe by construction rather than by
convention: it cannot reach the network, and it cannot write to the real
`~/.lab-harness`.
"""

import logging
import socket
from collections.abc import Iterator
from pathlib import Path

import pytest

from lab_harness.config import SessionConfig
from lab_harness.logs import ROOT_LOGGER


class NetworkBlockedError(AssertionError):
    """Raised when a test attempts a network connection."""


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly on any outbound connection.

    Tests must never reach a real API: it costs tokens, goes flaky, and hides
    missing fakes. Real end-to-end verification is manual.
    """

    def deny(self: socket.socket, *args: object, **kwargs: object) -> None:
        raise NetworkBlockedError(
            "Network access is blocked in tests. Use FakeProvider or an injected fake."
        )

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)


@pytest.fixture(autouse=True)
def _isolate_lab_harness_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point LAB_HARNESS_HOME at a per-test directory.

    A test that persists a session must not touch the developer's real
    `~/.lab-harness`. Setting the env var every test makes that impossible rather
    than merely discouraged — `store.home()` reads it first.
    """
    isolated = tmp_path / "lab-harness-home"
    monkeypatch.setenv(SessionConfig().home_env, str(isolated))
    return isolated


#: Env vars cleared before every test: credentials, plus the overrides that
#: change behavior. A developer who exports either locally must not get
#: different results from CI, which exports none.
_CLEARED_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "TAVILY_API_KEY",
    # The shipped OpenAI-compatible profiles. A developer with any of these
    # exported must get the same results as CI, which exports none.
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "ZHIPUAI_API_KEY",
    "ZHIPUAI_BASE_URL",
    # Not a credential, but it outranks `agent.model`, so leaving it set would
    # make every default-model assertion depend on the developer's shell.
    "LAB_HARNESS_MODEL",
    # Same reasoning: it outranks `logging.level`.
    "LAB_HARNESS_LOG_LEVEL",
)


@pytest.fixture(autouse=True)
def _clear_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset real credentials and overrides so no test can depend on one."""
    for name in _CLEARED_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """Undo any logging configuration a test installs.

    `configure_logging` mutates a process-global logger, so a single test that
    runs `main()` would otherwise leave its handler, level, and `propagate`
    setting in place for every test that follows — including the ones that
    assert on captured output.
    """
    logger = logging.getLogger(ROOT_LOGGER)
    before = (list(logger.handlers), logger.level, logger.propagate)
    yield
    handlers, level, propagate = before
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate
