#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure-Python (paramiko) SSH support (spec §13).

Public API (stable -- linktools-mobile subclasses it)::

    from linktools.ssh import SSHClient, SSHForward, SSHReverse
    from linktools.ssh import SSHHostKeyPolicy

``import linktools.ssh`` does NOT require paramiko/scp: only ``SSHClient`` does
(it needs the optional ``linktools[ssh]`` extra). ``SSHForward`` / ``SSHReverse``
/ ``SSHHostKeyPolicy`` / ``host_key_policy_class`` are paramiko-free at import
time (paramiko is imported lazily when actually used). Without paramiko,
``SSHClient`` becomes a placeholder that raises ``ModuleError`` on use.
"""

from linktools.errors import (
    SSHError, SSHConnectionError, SSHAuthenticationError, SSHHostKeyError,
    SSHCommandError, SSHChannelError, SSHTransferError, SSHForwardError,
    SSHTimeoutError, missing_optional_class,
)

__all__ = [
    "SSHClient", "SSHForward", "SSHReverse",
    "SSHHostKeyPolicy", "host_key_policy_class",
    "SSHError", "SSHConnectionError", "SSHAuthenticationError", "SSHHostKeyError",
    "SSHCommandError", "SSHChannelError", "SSHTransferError", "SSHForwardError",
    "SSHTimeoutError",
]


# paramiko-free at import time (paramiko is imported lazily inside methods).
from .forward import SSHForward, SSHReverse
from .hostkey import SSHHostKeyPolicy, host_key_policy_class

try:
    from .client import SSHClient
except ImportError as _exc:  # paramiko/scp not installed
    SSHClient = missing_optional_class("SSHClient", "ssh", _exc)
