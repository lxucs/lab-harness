"""Shared fixtures. The autouse network blocker is the offline guarantee."""

import socket

import pytest


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
