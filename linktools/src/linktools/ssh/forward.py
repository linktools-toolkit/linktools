#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SSH local and reverse port forwarding (spec §13.7/§13.8).

Both forwarders are :class:`~linktools.types.Stoppable`: ``stop()`` is
idempotent, closes channels before the transport, and joins the server thread.
The classes take a connected ``SSHClient`` instance, so this module does not
import the client (no circular dependency).
"""

import select
import socket
import threading

from linktools import utils
from linktools.core import environ
from linktools.types import Stoppable

try:
    import SocketServer
except ImportError:
    import socketserver as SocketServer

_logger = environ.get_logger("ssh")


class SSHForward(Stoppable):
    """Manage local-to-remote SSH port forwarding."""
    local_host = property(lambda self: self._local_host)
    local_port = property(lambda self: self._local_port)
    forward_host = property(lambda self: self._forward_host)
    forward_port = property(lambda self: self._forward_port)

    def __init__(self, client, local_host, local_port, forward_host, forward_port):
        self._local_host = local_host
        self._local_port = local_port
        self._forward_host = forward_host
        self._forward_port = forward_port

        self._lock = lock = threading.RLock()
        self._channels = channels = []
        self._transport = transport = client.get_transport()

        self._forward_server = None
        self._forward_thread = None

        def start():

            class ForwardHandler(SocketServer.BaseRequestHandler):

                def handle(self):
                    try:
                        channel = transport.open_channel(
                            "direct-tcpip",
                            (forward_host, forward_port),
                            self.request.getpeername(),
                        )
                    except Exception as e:
                        _logger.error("Incoming request to %s:%s failed: %s" % (forward_host, forward_port, e))
                        return

                    if channel is None:
                        _logger.error("Incoming request to %s:%s was rejected by the SSH server." % (forward_host, forward_port))
                        return

                    _logger.debug(
                        "Connected!  Tunnel open %s -> %s -> %s" % (
                            self.request.getpeername(),
                            channel.getpeername(),
                            (forward_host, forward_port),
                        ))

                    with lock:
                        channels.append(channel)

                    try:
                        while not channel.closed:
                            r, w, x = select.select([self.request, channel], [], [])
                            if self.request in r:
                                data = self.request.recv(1024)
                                if len(data) == 0:
                                    break
                                channel.send(data)
                            if channel in r:
                                data = channel.recv(1024)
                                if len(data) == 0:
                                    break
                                self.request.send(data)
                    except Exception as e:
                        _logger.debug("Forwarding request to %s:%s failed: %s" % (forward_host, forward_port, e))
                    finally:
                        peername = utils.ignore_errors(self.request.getpeername)
                        utils.ignore_errors(channel.close)
                        utils.ignore_errors(self.request.close)
                        _logger.debug("Tunnel closed from %s" % (peername,))

                        with lock:
                            channels.remove(channel)

            class ForwardServer(SocketServer.ThreadingTCPServer):
                daemon_threads = True
                allow_reuse_address = True

            self._forward_server = ForwardServer((self._local_host, local_port), ForwardHandler)
            self._forward_thread = threading.Thread(target=self._forward_server.serve_forever)
            self._forward_thread.daemon = True
            self._forward_thread.start()

        self._stop_on_error(start)

    def stop(self):
        """Stop local SSH port forwarding."""
        if self._forward_server is not None:
            try:
                self._forward_server.shutdown()
                if self._forward_thread is not None:
                    self._forward_thread.join()
            except Exception as e:
                _logger.error("Cancel port forward failed: %r" % e)

        with self._lock:
            for channel in self._channels:
                utils.ignore_errors(channel.close)
            self._channels = []


class SSHReverse(Stoppable):
    """Manage remote-to-local SSH port forwarding."""
    remote_host = property(lambda self: self._remote_host)
    remote_port = property(lambda self: self._remote_port)
    forward_host = property(lambda self: self._forward_host)
    forward_port = property(lambda self: self._forward_port)

    def __init__(self, client, forward_host, forward_port, remote_host, remote_port):
        self._remote_host = remote_host
        self._remote_port = None
        self._forward_host = forward_host
        self._forward_port = forward_port
        self._lock = lock = threading.RLock()
        self._channels = channels = []
        self._transport = transport = client.get_transport()
        self._forward_thread = None

        def start():
            self._remote_port = self._transport.request_port_forward(remote_host, remote_port or 0)

            def forward_handler(channel):

                sock = socket.socket()
                try:
                    sock.connect((forward_host, forward_port))
                except Exception as e:
                    utils.ignore_errors(channel.close)
                    utils.ignore_errors(sock.close)
                    _logger.error("Forwarding request to %s:%s failed: %s" % (forward_host, forward_port, e))
                    return

                _logger.debug(
                    "Connected!  Tunnel open %s -> %s -> %s" % (
                        channel.origin_addr,
                        channel.getpeername(),
                        (forward_host, forward_port),
                    ))

                with lock:
                    channels.append(channel)

                try:
                    while not channel.closed:
                        r, w, x = select.select([sock, channel], [], [])
                        if sock in r:
                            data = sock.recv(1024)
                            if len(data) == 0:
                                break
                            channel.send(data)
                        if channel in r:
                            data = channel.recv(1024)
                            if len(data) == 0:
                                break
                            sock.send(data)
                except Exception as e:
                    _logger.debug("Forwarding request to %s:%s failed: %s" % (forward_host, forward_port, e))
                finally:
                    utils.ignore_errors(channel.close)
                    utils.ignore_errors(sock.close)
                    _logger.debug("Tunnel closed from %s" % (channel.origin_addr,))

                    with lock:
                        channels.remove(channel)

            class ForwardThread(threading.Thread):

                def __init__(self):
                    super().__init__()
                    self.event = threading.Event()

                def run(self):
                    while not self.event.is_set():
                        channel = transport.accept(.5)
                        if channel is None:
                            continue
                        thread = threading.Thread(
                            target=forward_handler, args=(channel,)
                        )
                        thread.daemon = True
                        thread.start()

                def shutdown(self):
                    self.event.set()

            self._forward_thread = ForwardThread()
            self._forward_thread.daemon = True
            self._forward_thread.start()

        self._stop_on_error(start)

    def stop(self):
        """Stop remote SSH port forwarding."""
        if self._remote_port is not None:
            try:
                self._transport.cancel_port_forward(self._remote_host, self._remote_port)
            except Exception as e:
                _logger.warning("Cancel port forward failed: %s" % e)

        if self._forward_thread is not None:
            try:
                self._forward_thread.shutdown()
                self._forward_thread.join()
            except Exception as e:
                _logger.warning("Shutdown forward thread failed: %s" % e)

        with self._lock:
            for channel in self._channels:
                utils.ignore_errors(channel.close)
            self._channels = []
