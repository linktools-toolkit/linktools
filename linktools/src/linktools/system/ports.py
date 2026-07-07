#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Port helpers (spec §14.1, §14.3 SYS-002)."""

from ..errors import NoFreePortFoundError


def bind(port, socket_type, socket_proto):
    """Try to bind ``port`` on both AF_INET6 and AF_INET; return the bound port
    or None if it cannot be bound."""
    import socket

    got_socket = False
    for family in (socket.AF_INET6, socket.AF_INET):
        try:
            sock = socket.socket(family, socket_type, socket_proto)
            got_socket = True
        except socket.error:
            continue
        try:
            sock.bind(("0.0.0.0", port))
            if socket_type == socket.SOCK_STREAM:
                sock.listen(1)
            port = sock.getsockname()[1]
        except socket.error:
            return None
        finally:
            sock.close()
    return port if got_socket else None


def is_port_free(port):
    """Return whether ``port`` is free for both TCP and UDP binding.

    Note (§14.3): there is an inherent TOCTOU race between this check and an
    actual bind; prefer :func:`reserve_tcp_port` when the port is needed
    immediately. ``is_port_free`` is for advisory/hint scenarios.
    """
    import socket
    return bind(port, socket.SOCK_STREAM, socket.IPPROTO_TCP) is not None and \
        bind(port, socket.SOCK_DGRAM, socket.IPPROTO_UDP) is not None


def get_free_port():
    """Return a likely-free port (advisory only, spec §14.3)."""
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        try:
            return s.getsockname()[1]
        finally:
            s.close()
    except OSError:
        import random

        for _ in range(20):
            port = random.randint(30000, 40000)
            if is_port_free(port):
                return port
        raise NoFreePortFoundError("No free port found")


import contextlib


@contextlib.contextmanager
def reserve_tcp_port(host="127.0.0.1", port=0):
    """Bind a TCP socket and yield (host, port) without closing it (§14.3).

    Unlike :func:`get_free_port`, the socket stays bound for the duration of the
    context, eliminating the TOCTOU race before the caller binds the same port.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        bound = sock.getsockname()
        yield bound[0], bound[1]
    finally:
        sock.close()
