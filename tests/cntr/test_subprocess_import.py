#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Import cntr's public modules in a fresh interpreter.

A fresh interpreter is required because sys.modules in the pytest process can
hide import cycles that only surface on a module's first import.
"""
import subprocess
import sys

_CHECK = """
import linktools.cntr.container
import linktools.cntr.manager
import linktools.cntr.__main__
from linktools.cntr.__main__ import command
assert command is not None
"""


def test_public_modules_import_in_a_fresh_interpreter():
    result = subprocess.run(
        [sys.executable, "-c", _CHECK],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cli_help_runs_in_a_fresh_interpreter():
    result = subprocess.run(
        [sys.executable, "-m", "linktools.cntr", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
