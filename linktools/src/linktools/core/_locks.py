#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unified locking (spec §7.11 CAC-009).

Cache no longer owns general-purpose locks. :class:`LockManager` provides the
two lock kinds the rest of the system needs -- a named inter-process lock
(``process_lock``) for cross-process mutual exclusion keyed by a stable name
(download, tool install, git repo), and a path-addressed lock (``file_lock``)
for guarding a specific file. Both are file-based so they work across
processes, not just threads.

Acquired locks are context managers::

    with environ.locks.process_lock("download:<hash>"):
        ...

    with environ.locks.file_lock(path):
        ...
"""
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Union

__all__ = ["LockManager"]

PathLike = Union[str, "os.PathLike[str]"]  # noqa: F821

# Keep lock-file names to one filesystem component so a hostile name can never
# escape the lock directory (slashes become ``_``).
_NAME_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize(name):
    # type: (str) -> str
    cleaned = _NAME_SAFE.sub("_", name).strip("._") or "lock"
    return cleaned[:128]


class LockManager(object):
    """File-based process/file locks rooted at ``lock_dir``."""

    def __init__(self, lock_dir):
        # type: (PathLike) -> None
        self._lock_dir = Path(str(lock_dir))

    @property
    def lock_dir(self):
        # type: () -> Path
        return self._lock_dir

    def file_lock(self, path):
        # type: (PathLike) -> Any
        """Return a lock guarding the given file path (spec §6.2).

        The lock file lives under ``lock_dir`` (keyed by the sha256 of the
        absolute target path), NOT on the business file itself — so the target
        is never polluted with a ``.lock`` sidecar.
        """
        from filelock import FileLock
        import hashlib

        self._lock_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(os.path.abspath(str(path)).encode()).hexdigest()[:16]
        target = self._lock_dir / (digest + ".lock")
        return FileLock(str(target))

    def process_lock(self, name):
        # type: (str) -> Any
        """Return a named inter-process lock (context manager).

        The lock file lives under ``lock_dir``; ``name`` is sanitised to a
        single filesystem component so it cannot traverse out of the directory.
        """
        from filelock import FileLock

        self._lock_dir.mkdir(parents=True, exist_ok=True)
        target = self._lock_dir / (_sanitize(name) + ".lock")
        return FileLock(str(target))

    def __repr__(self):
        # type: () -> str
        return "LockManager(lock_dir=%r)" % (str(self._lock_dir),)
