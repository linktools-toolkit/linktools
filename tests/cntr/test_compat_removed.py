#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Breaking-change contract: no compatibility aliases remain anywhere in
``linktools.cntr.__main__`` or the wider cntr command surface -- downstream
must call the formal entry points (``commands._shared.manager``,
``commands.compose.ComposeCommand``, ...) directly."""
import re
import subprocess
import sys
from pathlib import Path

import linktools.cntr.__main__ as entry

REPO_ROOT = Path(__file__).resolve().parents[2]

_FORBIDDEN_PATTERN = re.compile(
    r"compatibility-period|deprecated|_iter_container_names|_iter_installed_container_names"
    r"|Kept for compatibility|Suggestions are kept for compatibility",
    re.IGNORECASE,
)


def test_main_module_exports_only_command():
    assert hasattr(entry, "Command")
    assert hasattr(entry, "command")

    assert not hasattr(entry, "manager")
    assert not hasattr(entry, "RepoCommand")
    assert not hasattr(entry, "ConfigCommand")
    assert not hasattr(entry, "ExecCommand")
    assert not hasattr(entry, "_iter_container_names")
    assert not hasattr(entry, "_iter_installed_container_names")


_EXCLUDED_FROM_SCAN = frozenset({
    # Names the forbidden phrases literally, as the pattern being checked for.
    "test_compat_removed.py",
    # Asserts the *absence* of "deprecated" in real CLI output -- a
    # regression test for this exact removal, not leftover compat language.
    "test_compose_namespace.py",
})


def test_no_compatibility_language_in_cntr_source_and_tests():
    files = []
    for d in ("linktools-cntr/src/linktools/cntr", "tests/cntr"):
        base = REPO_ROOT / d
        files.extend(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)

    offenders = []
    for path in files:
        if path.name in _EXCLUDED_FROM_SCAN:
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _FORBIDDEN_PATTERN.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "Compatibility-era language found:\n" + "\n".join(offenders)


def test_bare_config_help_has_no_compose_or_deprecation_output():
    result = subprocess.run(
        [sys.executable, "-m", "linktools.cntr", "config"],
        capture_output=True, text=True,
    )
    assert "deprecated" not in result.stderr.lower()
    assert "deprecated" not in result.stdout.lower()


def test_lock_command_is_unknown():
    # Note: this sandbox's argparse-error exit code is 0 even for an
    # unknown COMMAND (a pre-existing linktools.cli framework quirk,
    # unrelated to this change) -- assert on the error text instead.
    result = subprocess.run(
        [sys.executable, "-m", "linktools.cntr", "lock"],
        capture_output=True, text=True,
    )
    assert "invalid choice: 'lock'" in result.stderr


def test_diff_command_is_unknown():
    result = subprocess.run(
        [sys.executable, "-m", "linktools.cntr", "diff"],
        capture_output=True, text=True,
    )
    assert "invalid choice: 'diff'" in result.stderr
