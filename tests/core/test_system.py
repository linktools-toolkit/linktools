# -*- coding: utf-8 -*-
"""Tests for linktools.system platform/port helpers."""
import socket

import pytest

from linktools import system


# SYS-001 ---------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("amd64", "x86_64"),
    ("AMD64", "x86_64"),
    ("aarch64", "arm64"),
    ("armv7l", "arm"),
    ("armv6l", "arm"),
    ("i386", "x86"),
    ("i686", "x86"),
    ("x86_64", "x86_64"),
    ("arm64", "arm64"),
    ("arm", "arm"),
    ("unknown-arch", "unknown-arch"),  # unknown passes through lowercased
    ("", ""),
])
def test_normalize_arch(raw, expected):
    assert system.normalize_arch(raw) == expected


def test_normalize_platform():
    assert system.normalize_platform("Linux") == "linux"
    assert system.normalize_platform("  Darwin ") == "darwin"


def test_get_system_get_machine_are_cached_strings():
    s = system.get_system()
    m = system.get_machine()
    assert isinstance(s, str) and s == s.lower()
    assert isinstance(m, str) and m == m.lower()
    # cached -> same object on repeat call
    assert system.get_system() is s


# SYS-002 reserve_tcp_port ----------------------------------------------

def test_reserve_tcp_port_binds_and_releases():
    with system.reserve_tcp_port() as (host, port):
        assert host == "127.0.0.1"
        assert 0 < port < 65536
        # while reserved, the port is held by our socket; binding it again fails.
        clash = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(OSError):
                clash.bind(("127.0.0.1", port))
        finally:
            clash.close()
    # after release, it can be bound again
    again = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        again.bind(("127.0.0.1", port))
    finally:
        again.close()


def test_get_free_port_returns_int():
    port = system.get_free_port()
    assert isinstance(port, int) and 0 < port < 65536
