#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Python interpreter identification (spec §14.1)."""

import sys

from .. import utils

_interpreter = _interpreter_ident = None


def get_interpreter():
    """Return the current Python interpreter executable path."""
    global _interpreter
    if _interpreter is None:
        _interpreter = sys.executable
    return _interpreter


def get_interpreter_ident():
    # type: () -> str
    """Return a stable identifier for the current interpreter (prefix + version)."""
    global _interpreter_ident
    if _interpreter_ident is None:
        import platform
        _interpreter_ident = "%s_%s" % (
            utils.get_hash_ident(sys.exec_prefix), platform.python_version())
    return _interpreter_ident
