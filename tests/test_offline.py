import socket

import pytest

from tests.conftest import NetworkBlockedError


def test_network_is_blocked() -> None:
    with pytest.raises(NetworkBlockedError):
        socket.create_connection(("example.com", 80), timeout=1)


def test_loopback_is_blocked_too() -> None:
    with pytest.raises(NetworkBlockedError):
        socket.socket().connect(("127.0.0.1", 9))
