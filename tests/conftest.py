"""Shared fixtures.

Two autouse guards make the suite safe by construction rather than by
convention: it cannot reach the network, and it cannot write to the real
`~/.harness`.
"""

import socket
from pathlib import Path

import pytest

from harness.config import SessionConfig


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
def _isolate_harness_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point HARNESS_HOME at a per-test directory.

    A test that persists a session must not touch the developer's real
    `~/.harness`. Setting the env var every test makes that impossible rather
    than merely discouraged — `store.home()` reads it first.
    """
    isolated = tmp_path / "harness-home"
    monkeypatch.setenv(SessionConfig().home_env, str(isolated))
    return isolated


#: Credential env vars cleared before every test. A developer who exports a key
#: locally must not get different results from CI, which exports none.
_CREDENTIAL_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "TAVILY_API_KEY",
)


@pytest.fixture(autouse=True)
def _clear_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset real credentials so no test can accidentally depend on one."""
    for name in _CREDENTIAL_VARS:
        monkeypatch.delenv(name, raising=False)
