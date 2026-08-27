#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""User-visible CLI behavior for removed and invalid commands."""

import os
import subprocess
import sys
import tempfile

import pytest


def _run(*args: str) -> "subprocess.CompletedProcess[str]":
    environment = dict(os.environ)
    environment["HOME"] = tempfile.mkdtemp()
    return subprocess.run(
        [sys.executable, "-m", "linktools.cntr", *args],
        capture_output=True,
        text=True,
        env=environment,
    )


@pytest.mark.parametrize("name", ("lock", "diff", "unknown"))
def test_invalid_commands_exit_nonzero(name: str) -> None:
    assert _run(name).returncode != 0


@pytest.mark.parametrize(
    "args",
    (
        ("compose", "--check", "--format", "json"),
        ("add", "this-container-does-not-exist"),
    ),
)
def test_business_errors_exit_nonzero(args: "tuple[str, ...]") -> None:
    assert _run(*args).returncode != 0
