#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Network helpers: LAN/WAN IP (spec §14.1, §14.4 SYS-003)."""

import re

from .. import utils

# Basic IPv4 shape check -- the WAN-IP service must return an address, not an
# HTML error page (spec §14.4: do not silently return a wrong-page body).
_IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


def get_lan_ip():
    """Return the primary LAN IPv4 address, or None if it cannot be determined."""
    s = None
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return None
    finally:
        if s is not None:
            utils.ignore_errors(s.close)


def _default_transport(url, timeout):
    # type: (str, float) -> "contextlib.AbstractContextManager"
    from urllib.request import urlopen
    return urlopen(url, timeout=timeout)


def get_wan_ip(url=None, timeout=10.0, transport=None):
    # type: (str, float, object) -> "str | None"
    """Return the public WAN IP via the configured service, or None on failure.

    Spec §14.4 SYS-003: the URL comes from Environment config (DEFAULT_WAN_IP_URL)
    unless overridden. The response is validated as an IPv4 address so a captive
    portal / error page is never returned as the IP. ``transport`` is injectable
    for tests (a callable ``(url, timeout) -> context manager yielding a response
    with ``.read()``). On any failure or invalid response, returns None and logs
    a warning rather than raising.
    """
    environ = utils.get_environ()
    if url is None:
        url = environ.get_config("DEFAULT_WAN_IP_URL")
    fetch = transport or _default_transport
    logger = environ.get_logger("system.network")
    try:
        with fetch(url, timeout) as response:
            body = response.read().decode().strip()
    except Exception as exc:
        logger.warning("WAN IP lookup failed: %s", exc)
        return None
    if not _IPV4_RE.match(body):
        logger.warning("WAN IP service returned a non-IP response: %r", body[:64])
        return None
    return body
