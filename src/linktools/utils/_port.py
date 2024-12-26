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
import random
import socket

from ..types import Error


# from: https://github.com/google/python_portpicker


class NoFreePortFoundError(Error):
    """Exception indicating that no free port could be found."""


def bind(port: int, socket_type: socket.SocketKind, socket_proto: int):
    """Try to bind to a socket of the specified type, protocol, and port.

    This is primarily a helper function for PickUnusedPort, used to see
    if a particular port number is available.

    For the port to be considered available, the kernel must support at least
    one of (IPv6, IPv4), and the port must be available on each supported
    family.

    Args:
      port: The port number to bind to, or 0 to have the OS pick a free port.
      socket_type: The type of the socket (ex: socket.SOCK_STREAM).
      socket_proto: The protocol of the socket (ex: socket.IPPROTO_TCP).

    Returns:
      The port number on success or None on failure.
    """
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
      port: integer, port to check
    Returns:
      boolean, whether it is free to use for both TCP and UDP
    """
    return bind(port, socket.SOCK_STREAM, socket.IPPROTO_TCP) is not None and \
           bind(port, socket.SOCK_DGRAM, socket.IPPROTO_UDP) is not None


def get_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('127.0.0.1', 0))
        try:
            return s.getsockname()[1]
        finally:
            s.close()
    except OSError:
        for _ in range(20):
            port = random.randint(30000, 40000)
            if not is_port_free(port):
                return port
        raise NoFreePortFoundError("No free port found")
