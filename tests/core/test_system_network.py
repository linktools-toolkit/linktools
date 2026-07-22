# -*- coding: utf-8 -*-
"""Tests for system.network WAN-IP validation."""

import pytest

from linktools.system import network


class _Resp(object):
    def __init__(self, body):
        self._body = body if isinstance(body, bytes) else body.encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body


def _transport(body):
    return lambda url, timeout: _Resp(body)


def test_get_wan_ip_valid():
    out = network.get_wan_ip(url="http://example/ip", transport=_transport("203.0.113.5"))
    assert out == "203.0.113.5"


def test_get_wan_ip_rejects_error_page():
    # A captive portal / error HTML must not be returned as an IP.
    out = network.get_wan_ip(url="http://example/ip", transport=_transport("<html>error</html>"))
    assert out is None


def test_get_wan_ip_failure_returns_none():
    def boom(url, timeout):
        raise OSError("net down")
    assert network.get_wan_ip(url="http://example/ip", transport=boom) is None


def test_get_wan_ip_strips_whitespace():
    out = network.get_wan_ip(url="http://example/ip", transport=_transport("  198.51.100.1\n"))
    assert out == "198.51.100.1"
