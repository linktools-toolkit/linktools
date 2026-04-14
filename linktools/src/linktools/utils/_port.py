#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Copyright 2007 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import typing

from ..types import NoFreePortFoundError

if typing.TYPE_CHECKING:
    import socket


# from: https://github.com/google/python_portpicker

def bind(port: int, socket_type: "socket.SocketKind", socket_proto: int):
    """Try to bind to a socket of the specified type, protocol, and port.

    Args:
        port (int): Remote port number.
        socket_type (socket.SocketKind): The socket_type value.
        socket_proto (int): The socket_proto value.

    Returns:
        Any: The operation result.
    """
    import socket

    got_socket = False
    for family in (socket.AF_INET6, socket.AF_INET):
        try:
            sock = socket.socket(family, socket_type, socket_proto)
            got_socket = True
        except socket.error:
            continue
        try:
            # sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('0.0.0.0', port))
            if socket_type == socket.SOCK_STREAM:
                sock.listen(1)
            port = sock.getsockname()[1]
        except socket.error:
            return None
        finally:
            sock.close()
    return port if got_socket else None


def is_port_free(port: int):
    """Check if specified port is free.

    Args:
        port (int): Remote port number.

    Returns:
        Any: The operation result.
    """
    import socket

    return bind(port, socket.SOCK_STREAM, socket.IPPROTO_TCP) is not None and \
        bind(port, socket.SOCK_DGRAM, socket.IPPROTO_UDP) is not None


def get_free_port():
    """Return an available TCP port on the requested host.

    Returns:
        Any: The operation result.

    Raises:
        Exception: Propagates errors raised while completing the operation.
    """
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('127.0.0.1', 0))
        try:
            return s.getsockname()[1]
        finally:
            s.close()
    except OSError:
        import random

        for _ in range(20):
            port = random.randint(30000, 40000)
            if not is_port_free(port):
                return port
        raise NoFreePortFoundError("No free port found")
