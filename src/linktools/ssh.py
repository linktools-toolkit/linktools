#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import contextlib
import getpass
import logging
import os
import select
import socket
import sys
import threading
import time
from typing import Any

import paramiko
from paramiko.ssh_exception import AuthenticationException, SSHException
from scp import SCPClient

from . import utils
from ._environ import environ
from .rich import prompt, create_progress
from .types import Stoppable
from .utils import list2cmdline, ignore_errors, is_unix_like

try:
    import SocketServer
except ImportError:
    import socketserver as SocketServer

_logger = environ.get_logger("ssh")

_channel_logger = environ.get_logger("ssh.channel")
_channel_logger.setLevel(logging.CRITICAL)


class SSHClient(paramiko.SSHClient):

    def __init__(self):
        super().__init__()
        self.set_log_channel(_channel_logger.name)

    def connect_with_pwd(self, hostname, port=22, username=None, password=None, **kwargs):
        if username is None:
            username = getpass.getuser()

        try:
            super().connect(
                hostname,
                port=port,
                username=username,
                # password=password,
                **kwargs
            )
        except SSHException:
            transport = self.get_transport()
            if transport is None:
                raise
            elif not transport.is_alive():
                raise
            elif transport.is_authenticated():
                raise

            if password is not None:
                try:
                    transport.auth_password(username, password)
                except AuthenticationException as e:
                    _logger.warning(f"Authentication (password) failed.")
                    raise e from None

            else:
                auth_exception = None
                for i in range(3):
                    password = prompt(
                        f"{username}@{hostname}'s password",
                        password=True
                    )
                    try:
                        transport.auth_password(username, password)
                        auth_exception = None
                        break
                    except AuthenticationException as e:
                        _logger.warning(f"Authentication (password) failed.")
                        auth_exception = e

                if auth_exception is not None:
                    raise auth_exception from None

    def open_shell(self, *args: Any):
        if len(args) > 0:
            stdin, stdout, stderr = self.exec_command(
                list2cmdline([str(arg) for arg in args]),
                get_pty=True
            )

            def iter_lines(io1, io2):
                for line in iter(io1.readline, ""):
                    print(line, end="", file=io2)

            threads = [
                threading.Thread(target=iter_lines, args=(stdout, sys.stdout)),
                threading.Thread(target=iter_lines, args=(stderr, sys.stderr)),
            ]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()
        else:
            chan = self.invoke_shell()
            try:
                self._open_shell(chan)
            finally:
                ignore_errors(chan.close)

    if utils.is_windows():

        @classmethod
        def _open_shell(cls, channel: "paramiko.Channel"):
            import msvcrt

            def write_all(sock):
                while not channel.closed:
                    try:
                        data = sock.recv(1024)
                        if len(data) == 0:
                            sys.stdout.flush()
                            break
                        sys.stdout.write(data.decode())
                        sys.stdout.flush()
                    except OSError:
                        break

            write_thread = threading.Thread(target=write_all, args=(channel,))
            write_thread.start()

            try:
                delay = 0.001
                while not channel.closed:
                    if not msvcrt.kbhit():
                        delay = min(delay * 2, 0.1)
                        time.sleep(delay)
                        continue
                    delay = 0.001
                    char = msvcrt.getch()
                    if char == b"\xe0":
                        char = b"\x1b"
                    buff = char
                    while msvcrt.kbhit():
                        char = msvcrt.getch()
                        buff += char
                    channel.send(buff)
            except OSError:
                pass

    elif is_unix_like():

        @classmethod
        def _open_shell(cls, channel: "paramiko.Channel"):
            import select
            import termios
            import tty

            orig_tty = None
            stdin_fileno = sys.stdin.fileno()

            try:
                orig_tty = termios.tcgetattr(stdin_fileno)
                tty.setraw(stdin_fileno)
                tty.setcbreak(stdin_fileno)
            except Exception as e:
                _logger.debug(f"Set tty error: {e}")

            try:
                channel.settimeout(0.0)
                while not channel.closed:
                    rlist, wlist, xlist = select.select([channel, sys.stdin], [], [], 1.0)
                    if channel in rlist:
                        try:
                            data = channel.recv(1024)
                            if len(data) == 0:
                                break
                            sys.stdout.write(data.decode())
                            sys.stdout.flush()
                        except socket.timeout:
                            pass
                    if sys.stdin in rlist:
                        data = os.read(stdin_fileno, 1)
                        if len(data) == 0:
                            break
                        channel.send(data)
            except OSError:
                pass
            finally:
                if orig_tty:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_tty)

    else:

        def _open_shell(self, channel: "paramiko.Channel"):
            raise NotImplementedError("Unsupported platform")

    def get_file(self, remote_path: str, local_path: str):
        with self._open_scp() as scp:
            return scp.get(remote_path, local_path, recursive=True)

    def put_file(self, local_path: str, remote_path: str):
        with self._open_scp() as scp:
            return scp.put(local_path, remote_path, recursive=True)

    @contextlib.contextmanager
    def _open_scp(self):

        with create_progress() as progress:
            tasks = {}

            def update_progress(filename, size, sent):
                if isinstance(filename, bytes):
                    filename = filename.decode()
                task_id = tasks.get(filename, None)
                if task_id is None:
                    task_id = progress.add_task(filename, total=size)
                    tasks[filename] = task_id
                progress.update(
                    task_id,
                    completed=sent,
                    description=filename,
                    total=size
                )

            with SCPClient(self.get_transport(), progress=update_progress) as scp:
                yield scp

    def forward(self, forward_host: str, forward_port: int, local_port: int = None) -> "SSHForward":
        """
        :param forward_host: The host to forward to.
        :param forward_port: The port to forward to.
        :param local_port: The local port to listen on.
        :return: A Stoppable object.
        """

        if local_port is None:
            local_port = utils.get_free_port()

        return SSHForward(self, "", local_port, forward_host, forward_port)

    def reverse(self, forward_host: str, forward_port: int, remote_port: int = None):
        """
        :param forward_host: The host to forward to.
        :param forward_port: The port to forward to.
        :param remote_port: The remote port to listen on.
        :return: A Stoppable object.
        """
        return SSHReverse(self, forward_host, forward_port, "", remote_port)


class SSHForward(Stoppable):
    local_host = property(lambda self: self._local_host)
    local_port = property(lambda self: self._local_port)
    forward_host = property(lambda self: self._forward_host)
    forward_port = property(lambda self: self._forward_port)

    def __init__(self, client: SSHClient, local_host: str, local_port: int, forward_host: str, forward_port: int):
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
                        _logger.error(f"Incoming request to {forward_host}:{forward_port} failed: {e}")
                        return

                    if channel is None:
                        _logger.error(f"Incoming request to {forward_host}:{forward_port} was rejected by the SSH server.")
                        return

                    _logger.debug(
                        f"Connected!  Tunnel open "
                        f"{self.request.getpeername()} -> "
                        f"{channel.getpeername()} -> "
                        f"{(forward_host, forward_port)}")

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
                        _logger.debug(f"Forwarding request to {forward_host}:{forward_port} failed: {e}")
                    finally:
                        peername = utils.ignore_errors(self.request.getpeername)
                        utils.ignore_errors(channel.close)
                        utils.ignore_errors(self.request.close)
                        _logger.debug(f"Tunnel closed from {peername}")

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
    remote_host = property(lambda self: self._remote_host)
    remote_port = property(lambda self: self._remote_port)
    forward_host = property(lambda self: self._forward_host)
    forward_port = property(lambda self: self._forward_port)

    def __init__(self, client: SSHClient, forward_host: str, forward_port: int, remote_host: str, remote_port: int):
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

            def forward_handler(channel: paramiko.Channel):

                sock = socket.socket()
                try:
                    sock.connect((forward_host, forward_port))
                except Exception as e:
                    utils.ignore_errors(channel.close)
                    utils.ignore_errors(sock.close)
                    _logger.error(f"Forwarding request to {forward_host}:{forward_port} failed: {e}")
                    return

                _logger.debug(
                    f"Connected!  Tunnel open "
                    f"{channel.origin_addr} -> "
                    f"{channel.getpeername()} -> "
                    f"{(forward_host, forward_port)}")

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
                    _logger.debug(f"Forwarding request to {forward_host}:{forward_port} failed: {e}")
                finally:
                    utils.ignore_errors(channel.close)
                    utils.ignore_errors(sock.close)
                    _logger.debug(f"Tunnel closed from {channel.origin_addr}")

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
        if self._remote_port is not None:
            try:
                self._transport.cancel_port_forward(self._remote_host, self._remote_port)
            except Exception as e:
                _logger.warning(f"Cancel port forward failed: {e}")

        if self._forward_thread is not None:
            try:
                self._forward_thread.shutdown()
                self._forward_thread.join()
            except Exception as e:
                _logger.warning(f"Sutdown forward thread failed: {e}")

        with self._lock:
            for channel in self._channels:
                utils.ignore_errors(channel.close)
            self._channels = []
