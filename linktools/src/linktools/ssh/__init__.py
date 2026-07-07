#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure-Python (paramiko) SSH support (spec §13).

Public API (stable -- linktools-mobile subclasses it)::

    from linktools.ssh import SSHClient, SSHForward, SSHReverse
    from linktools.ssh import SSHHostKeyPolicy
"""

from linktools.errors import (
    SSHError, SSHConnectionError, SSHAuthenticationError, SSHHostKeyError,
    SSHCommandError, SSHChannelError, SSHTransferError, SSHForwardError,
    SSHTimeoutError,
)

from .client import SSHClient
from .forward import SSHForward, SSHReverse
from .hostkey import SSHHostKeyPolicy, host_key_policy_class

__all__ = [
    "SSHClient", "SSHForward", "SSHReverse",
    "SSHHostKeyPolicy", "host_key_policy_class",
    "SSHError", "SSHConnectionError", "SSHAuthenticationError", "SSHHostKeyError",
    "SSHCommandError", "SSHChannelError", "SSHTransferError", "SSHForwardError",
    "SSHTimeoutError",
]
