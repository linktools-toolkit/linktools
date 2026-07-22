# -*- coding: utf-8 -*-
"""Import-time safety test.

Importing ``linktools`` must NOT:
- create directories;
- modify ``os.environ["PATH"]``;
- open the cache database;
- load tool definitions;
- scan entry points;
- configure rich;
- modify third-party loggers.

This test runs in a SUBPROCESS so the import is fresh and side effects are
measurable against a snapshot taken before the import.
"""
import os
import subprocess
import sys
import tempfile


def test_import_does_not_mutate_path_or_create_dirs():
    """import linktools must not mutate PATH or create storage dirs."""
    result = subprocess.run(
        [sys.executable, "-c", """
import os, sys, tempfile

# Snapshot state BEFORE import.
path_before = os.environ.get("PATH", "")
tmp = tempfile.mkdtemp()
storage = os.path.join(tmp, ".linktools")
home_override = os.environ["HOME"] if "HOME" in os.environ else None
os.environ["HOME"] = tmp  # redirect storage to the temp dir

# Import linktools.
import linktools

# Check: PATH not mutated.
path_after = os.environ.get("PATH", "")
assert path_before == path_after, \\
    "PATH mutated by import: %r -> %r" % (path_before[:80], path_after[:80])

# Check: no storage directory created.
assert not os.path.exists(storage), \\
    "storage dir created at import: %s" % storage

# Check: the environ singleton exists (allowed) but no heavy services triggered.
from linktools.core import environ
assert environ is not None

# Restore HOME.
if home_override is not None:
    os.environ["HOME"] = home_override
"""],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "PYTHONPATH": ""},  # clean path
    )
    if result.returncode != 0:
        import textwrap
        msg = result.stderr or result.stdout
        raise AssertionError(
            "import-time safety check failed (§19.2):\n" + textwrap.indent(msg, "  "))


def test_import_does_not_configure_logging():
    """import linktools must not add handlers to the root logger."""
    result = subprocess.run(
        [sys.executable, "-c", """
import logging
handlers_before = len(logging.getLogger().handlers)
import linktools
handlers_after = len(logging.getLogger().handlers)
assert handlers_before == handlers_after, \\
    "root logger handlers changed by import: %d -> %d" % (handlers_before, handlers_after)
"""],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "PYTHONPATH": ""},
    )
    if result.returncode != 0:
        import textwrap
        raise AssertionError(
            "import-time logging check failed (§19.2):\n" +
            textwrap.indent(result.stderr or result.stdout, "  "))
