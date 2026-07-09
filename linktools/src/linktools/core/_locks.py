"""Unified locking (LockManager).

Cache no longer owns general-purpose locks. LockManager provides
two lock kinds: a named inter-process lock (process_lock) and a
path-addressed lock (file_lock). Both are file-based.
"""

import hashlib
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from typing import Any

__all__ = ["LockManager"]

PathLike = Union[str, "os.PathLike[str]"]  # noqa: F821

_NAME_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize(name: str) -> str:
    cleaned = _NAME_SAFE.sub("_", name).strip("._") or "lock"
    return cleaned[:128]


class LockManager:
    """File-based process/file locks rooted at lock_dir."""

    def __init__(self, lock_dir: "PathLike") -> None:
        self._lock_dir = Path(str(lock_dir))

    @property
    def lock_dir(self) -> "Path":
        return self._lock_dir

    def file_lock(self, path: "PathLike") -> "Any":
        """Return a lock guarding the given file path.

        The lock file lives under lock_dir (keyed by sha256 of the
        absolute target path), NOT on the business file itself.
        """
        from filelock import FileLock

        self._lock_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(os.path.abspath(str(path)).encode()).hexdigest()[:16]
        target = self._lock_dir / (digest + ".lock")
        return FileLock(str(target))

    def process_lock(self, name: str) -> "Any":
        """Return a named inter-process lock (context manager)."""
        from filelock import FileLock

        self._lock_dir.mkdir(parents=True, exist_ok=True)
        target = self._lock_dir / (_sanitize(name) + ".lock")
        return FileLock(str(target))

    def __repr__(self) -> str:
        return "LockManager(lock_dir=%r)" % (str(self._lock_dir),)
