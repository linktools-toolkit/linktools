#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SSH client (spec §13): paramiko-backed, never a system ``ssh`` wrapper.

Public API (``SSHClient``) is stable -- linktools-mobile subclasses it. This
module owns the connection, interactive shell, SCP transfer, and forward/reverse
forwarding setup; the forwarder implementations live in :mod:`linktools.ssh.forward`.
"""

import contextlib
import getpass
import logging
import os
import sys
import threading
import time
from typing import TYPE_CHECKING

import paramiko
from paramiko.ssh_exception import AuthenticationException, SSHException
from scp import SCPClient

from linktools import utils
from linktools.core import environ
from linktools.system import get_free_port, is_unix_like, is_windows
from linktools.rich import prompt, create_progress
from linktools.utils import ignore_errors

from .forward import SSHForward, SSHReverse

if TYPE_CHECKING:
    from typing import Any

_logger = environ.get_logger("ssh")

_channel_logger = environ.get_logger("ssh.channel")
# route level through LoggingManager, not direct setLevel.
environ.logging.set_level("ssh.channel", logging.CRITICAL)


class SSHClient(paramiko.SSHClient):
    """Paramiko SSH client with shell, transfer, and forwarding helpers.

    Default host-key policy is STRICT (RejectPolicy) per v2 §11.3. Callers
    that need to accept unknown keys (e.g. iOS USB-forwarded loopback
    connections) must explicitly set INSECURE/AutoAddPolicy.
    """

    def __init__(self):
        super().__init__()
        self.set_log_channel(_channel_logger.name)
        # STRICT default; overrides paramiko's AutoAddPolicy.
        self.set_missing_host_key_policy(paramiko.RejectPolicy())

    def connect_with_pwd(self, hostname, port=22, username=None, password=None, **kwargs):
        """Connect and fall back to password authentication when needed."""
        if username is None:
            username = getpass.getuser()

        try:
            super().connect(
                hostname,
                port=port,
                username=username,
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
                    _logger.warning("Authentication (password) failed.")
                    raise e from None

            else:
                auth_exception = None
                for i in range(3):
                    password = prompt(
                        "%s@%s's password" % (username, hostname),
                        password=True
                    )
                    try:
                        transport.auth_password(username, password)
                        auth_exception = None
                        break
                    except AuthenticationException as e:
                        _logger.warning("Authentication (password) failed.")
                        auth_exception = e

                if auth_exception is not None:
                    raise auth_exception from None

    def open_shell(self, *args: "Any") -> None:
        """Open an interactive or command-backed SSH shell."""
        if len(args) > 0:
            stdin, stdout, stderr = self.exec_command(
                utils.list2cmdline([str(arg) for arg in args]),
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

    if is_windows():

        @classmethod
        def _open_shell(cls, channel):
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
        def _open_shell(cls, channel):
            import select
            import termios
            import tty
            import socket

            orig_tty = None
            stdin_fileno = sys.stdin.fileno()

            try:
                orig_tty = termios.tcgetattr(stdin_fileno)
                tty.setraw(stdin_fileno)
                tty.setcbreak(stdin_fileno)
            except Exception as e:
                _logger.debug("Set tty error: %s" % e)

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

        def _open_shell(self, channel):
            raise NotImplementedError("Unsupported platform")

    def get_file(self, remote_path, local_path):
        """Download a remote file or directory through SCP."""
        with self._open_scp() as scp:
            return scp.get(remote_path, local_path, recursive=True, preserve_times=True)

    def put_file(self, local_path, remote_path):
        """Upload a local file or directory through SCP."""
        with self._open_scp() as scp:
            return scp.put(local_path, remote_path, recursive=True, preserve_times=True)

    @contextlib.contextmanager
    def _open_scp(self):
        with create_progress(transfer=True) as progress:
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

    def forward(self, forward_host, forward_port, local_port=None):
        """:param forward_host: The host to forward to."""
        if local_port is None:
            local_port = get_free_port()
        return SSHForward(self, "", local_port, forward_host, forward_port)

    def reverse(self, forward_host, forward_port, remote_port=None):
        return SSHReverse(self, forward_host, forward_port, "", remote_port)
