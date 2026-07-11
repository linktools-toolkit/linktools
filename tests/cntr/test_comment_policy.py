#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Production code and tests describe current behavior/invariants only --
never a citation of the spec document that drove a change, and never a
narrative of what a piece of code used to be or was renamed from."""
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_FORBIDDEN_PATTERN = re.compile(
    r"Spec Part|Spec section|Phase [0-9]|Formerly|pre-registry|pre-refactor", re.IGNORECASE,
)


# This file's own path -- it necessarily contains the forbidden phrases
# literally (as the pattern being checked for), so it must exclude itself
# from both scans below.
_SELF = Path(__file__).resolve()


def _tracked_or_present_py_files(*dirs):
    files = []
    for d in dirs:
        base = REPO_ROOT / d
        if base.is_dir():
            files.extend(base.rglob("*.py"))
    return [f for f in files if "__pycache__" not in f.parts and f.resolve() != _SELF]


def test_no_spec_or_history_references_in_cntr_source_and_tests():
    files = _tracked_or_present_py_files("linktools-cntr/src/linktools/cntr", "tests/cntr")
    offenders = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _FORBIDDEN_PATTERN.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "Spec/history references found:\n" + "\n".join(offenders)


def test_grep_based_check_matches_the_python_scan():
    """Belt-and-suspenders: the exact grep command the spec's static check
    runs must also report zero hits (excluding this file itself, which
    necessarily contains the forbidden phrases literally)."""
    result = subprocess.run(
        ["grep", "-RInE", "--exclude", _SELF.name,
         "Spec Part|Spec section|Phase [0-9]|Formerly|pre-registry|pre-refactor",
         "linktools-cntr/src/linktools/cntr", "tests/cntr"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    # grep exits 1 when there are no matches.
    assert result.returncode == 1, f"unexpected grep hits:\n{result.stdout}"
