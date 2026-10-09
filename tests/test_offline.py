"""The offline guarantee itself. If these fail, no other test can be trusted."""

import socket

import pytest

from tests.conftest import NetworkBlockedError


def test_network_is_blocked() -> None:
    # closing=True so the socket is released even though connect() raises;
    # otherwise the leak surfaces as PytestUnraisableExceptionWarning under -W error.
    with pytest.raises(NetworkBlockedError), socket.socket() as sock:
        sock.connect(("example.com", 80))


def test_loopback_is_blocked_too() -> None:
    """Blocking must not carve out localhost — a local mock server is still a server."""
    with pytest.raises(NetworkBlockedError), socket.socket() as sock:
        sock.connect(("127.0.0.1", 9))


def test_connect_ex_is_blocked_too() -> None:
    """connect_ex returns an errno rather than raising, so it needs its own guard."""
    with pytest.raises(NetworkBlockedError), socket.socket() as sock:
        sock.connect_ex(("127.0.0.1", 9))
