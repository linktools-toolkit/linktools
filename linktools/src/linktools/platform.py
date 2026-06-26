#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

from . import utils
from .decorator import timeoutable
from .errors import NoFreePortFoundError

_is_windows_like = _is_unix_like = False
_interpreter = _interpreter_ident = None
_system = _machine = None

try:
    import msvcrt
except ModuleNotFoundError:
    try:
        import pwd
    except ModuleNotFoundError:
        ...
    else:
        _is_unix_like = True
else:
    _is_windows_like = True


def get_lan_ip() -> "str | None":
    s = None
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except:
        return None
    finally:
        if s is not None:
            utils.ignore_errors(s.close)


def get_wan_ip() -> "str | None":
    from urllib.request import urlopen
    try:
        with urlopen(utils.get_environ().get_config("DEFAULT_WAN_IP_URL")) as response:
            return response.read().decode().strip()
    except:
        return None


def get_interpreter():
    global _interpreter
    if _interpreter is None:
        _interpreter = sys.executable
    return _interpreter


def get_interpreter_ident() -> str:
    global _interpreter_ident
    if _interpreter_ident is None:
        import platform
        _interpreter_ident = f"{utils.get_hash_ident(sys.exec_prefix)}_{platform.python_version()}"
    return _interpreter_ident


def get_system() -> str:
    global _system
    if _system is None:
        import platform
        _system = platform.system().lower()
    return _system


def get_machine() -> str:
    global _machine
    if _machine is None:
        import platform
        _machine = platform.machine().lower()
    return _machine


def is_unix_like(system: str = None) -> bool:
    return system in ("darwin", "linux") if system else _is_unix_like


def is_windows(system: str = None) -> bool:
    return system == "windows" if system else _is_windows_like


if is_windows():

    def get_user(uid: int = None):
        import getpass
        return getpass.getuser()

    def get_uid(user: str = None):
        return 0

    def get_gid(user: str = None):
        return 0

    def get_shell_path():
        import shutil
        shell_path = shutil.which("powershell") or shutil.which("cmd")
        if shell_path:
            return shell_path
        if "ComSpec" in os.environ:
            shell_path = os.environ["ComSpec"]
            if shell_path and os.path.exists(shell_path):
                return shell_path
        raise NotImplementedError(f"Unsupported system `{get_system()}`")

elif is_unix_like():

    def get_user(uid: int = None) -> str:
        if uid is not None:
            import pwd
            return pwd.getpwuid(int(uid)).pw_name
        import getpass
        return getpass.getuser()

    def get_uid(user: str = None):
        if user is not None:
            import pwd
            return pwd.getpwnam(str(user)).pw_uid
        return os.getuid()

    def get_gid(user: str = None):
        if user is not None:
            import pwd
            return pwd.getpwnam(str(user)).pw_gid
        return os.getgid()

    def get_shell_path():
        if "SHELL" in os.environ:
            shell_path = os.environ["SHELL"]
            if shell_path and os.path.exists(shell_path):
                return shell_path
        try:
            import pwd
            return pwd.getpwnam(get_user()).pw_shell
        except:
            import shutil
            return shutil.which("zsh") or shutil.which("bash") or shutil.which("sh")

else:

    def get_user(uid: int = None) -> str:
        raise NotImplementedError(f"Unsupported system `{get_system()}`")

    def get_uid(user: str = None) -> int:
        raise NotImplementedError(f"Unsupported system `{get_system()}`")

    def get_gid(user: str = None) -> int:
        raise NotImplementedError(f"Unsupported system `{get_system()}`")

    def get_shell_path() -> str:
        raise NotImplementedError(f"Unsupported system `{get_system()}`")


def bind(port: int, socket_type: "socket.SocketKind", socket_proto: int):
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


def is_port_free(port: int):
    import socket
    return bind(port, socket.SOCK_STREAM, socket.IPPROTO_TCP) is not None and \
        bind(port, socket.SOCK_DGRAM, socket.IPPROTO_UDP) is not None


def get_free_port():
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


@timeoutable
def wait_event(event: "threading.Event", timeout) -> bool:
    interval = 1
    while True:
        t = timeout.remain
        if t is None:
            t = interval
        elif t <= 0:
            return False
        if event.wait(min(t, interval)):
            return True


@timeoutable
def wait_thread(thread: "threading.Thread", timeout) -> bool:
    interval = 1
    while True:
        t = timeout.remain
        if t is None:
            t = interval
        elif t <= 0:
            return False
        try:
            thread.join(min(t, interval))
        except:
            pass
        if not thread.is_alive():
            return True


@timeoutable
def wait_process(process: "subprocess.Popen", timeout) -> "int | None":
    import subprocess

    interval = 1
    while True:
        t = timeout.remain
        if t is None:
            t = interval
        elif t <= 0:
            return None
        try:
            return process.wait(min(t, interval))
        except subprocess.TimeoutExpired:
            pass
