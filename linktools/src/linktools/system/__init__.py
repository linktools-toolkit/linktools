#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""System / platform helpers (spec §14).

Splits the legacy ``platform.py`` into focused submodules (platform, user,
network, ports, wait, interpreter) and adds the §14.2 platform/arch
normalisation API. Consumers import from ``linktools.system``.
"""

from .platform import (
    get_system, get_machine, is_unix_like, is_windows,
    normalize_platform, normalize_arch,
)
from .user import get_user, get_uid, get_gid, get_shell_path
from .network import get_lan_ip, get_wan_ip
from .ports import bind, is_port_free, get_free_port, reserve_tcp_port
from .wait import wait_event, wait_thread, wait_process
from .interpreter import get_interpreter, get_interpreter_ident
from .shell import SUPPORTED_SHELLS, ShellScript, get_default_shell, get_shell
from .stub import CommandStub

__all__ = [
    # platform
    "get_system", "get_machine", "is_unix_like", "is_windows",
    "normalize_platform", "normalize_arch",
    # user
    "get_user", "get_uid", "get_gid", "get_shell_path",
    # network
    "get_lan_ip", "get_wan_ip",
    # ports
    "bind", "is_port_free", "get_free_port", "reserve_tcp_port",
    # wait
    "wait_event", "wait_thread", "wait_process",
    # interpreter
    "get_interpreter", "get_interpreter_ident",
    # shell-script generation (source/eval snippets)
    "SUPPORTED_SHELLS", "ShellScript", "get_default_shell", "get_shell",
    # executable wrapper scripts (.bat / POSIX sh)
    "CommandStub",
]
