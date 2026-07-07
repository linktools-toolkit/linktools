#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unified CLI exit codes (spec §16.4 CLI-003).

Maps an exception to a process exit code by its domain, so callers get a stable
contract::

    0   success
    2   user input error (CommandError / CliError)
    3   configuration error
    4   network / remote error (download, git, ssh)
    5   external-tool error
    10  internal error
    130 interrupted (Ctrl-C)
"""

from ..errors import (
    LinktoolsError, CliError, ConfigError, DownloadError, GitError, SSHError,
    ToolError,
)

EXIT_SUCCESS = 0
EXIT_USER_INPUT = 2
EXIT_CONFIG = 3
EXIT_NETWORK = 4
EXIT_TOOL = 5
EXIT_INTERNAL = 10
EXIT_INTERRUPT = 130

__all__ = [
    "EXIT_SUCCESS", "EXIT_USER_INPUT", "EXIT_CONFIG", "EXIT_NETWORK",
    "EXIT_TOOL", "EXIT_INTERNAL", "EXIT_INTERRUPT", "exit_code_for",
]


def exit_code_for(error):
    # type: (BaseException) -> int
    """Return the §16.4 exit code for ``error`` by its domain."""
    if isinstance(error, KeyboardInterrupt):
        return EXIT_INTERRUPT
    if isinstance(error, CliError):
        # CommandError and other CLI/user-input errors.
        return EXIT_USER_INPUT
    if isinstance(error, ConfigError):
        return EXIT_CONFIG
    if isinstance(error, (DownloadError, GitError, SSHError)):
        return EXIT_NETWORK
    if isinstance(error, ToolError):
        return EXIT_TOOL
    if isinstance(error, LinktoolsError):
        return EXIT_INTERNAL
    return EXIT_INTERNAL
