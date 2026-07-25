#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FilesystemKeyedCoordinator: generic per-key mutual exclusion via POSIX
advisory file locks (fcntl.flock).

One ``flock(LOCK_EX)`` lock file per key under ``<root>/.locks/``. ``flock``
is advisory and per-open-file-description, so two processes that both open the
same lock file and flock it are mutually exclusive -- the coordination spans
process boundaries on a shared filesystem. Every file operation (open, lock,
unlock, close) runs via ``asyncio.to_thread`` so blocking POSIX I/O never runs
on the event loop.

POSIX-only and ``O_NOFOLLOW``-only: ``fcntl`` is unavailable on Windows and
``O_NOFOLLOW`` is the defense that stops a symlink from being opened as a lock
file. The constructor raises :class:`StorageCoordinationNotSupportedError`
off-POSIX or where ``O_NOFOLLOW`` is absent rather than degrading to a weaker
mode. The production runtime is Linux; tests run on Linux CI.

The locks directory is created once at construction with mode ``0700``, is
rejected if it is itself a symlink, and must resolve inside ``root``. A key is
percent-encoded before becoming a lock-file name, so an arbitrary caller-owned
key (a digest, a capability path, ...) can never traverse outside the locks
directory."""

import asyncio
import fcntl
import os
import urllib.parse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from ...errors import StorageCoordinationNotSupportedError
from ..features import CoordinationScope

# Open flags for the lock file. O_NOFOLLOW refuses to open a symlink at the lock
# path; O_CLOEXEC closes the fd across exec (fork-safety for any worker that
# spawns). Both are required; O_NOFOLLOW's absence is a fail-closed stop.
_LOCK_OPEN_FLAGS = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW


class FilesystemKeyedCoordinator:
    """Per-key ``flock(LOCK_EX)`` over a lock file under ``root/.locks/``.
    Coordinates across processes sharing the filesystem.

    Declares PROCESS_LOCAL scope: ``flock`` coordinates processes sharing ONE
    filesystem/host, not a distributed multi-worker deployment across separate
    storage. A true multi-worker deployment injects a distributed
    KeyedCoordinator instead."""

    scope = CoordinationScope.PROCESS_LOCAL

    def __init__(self, *, root: Path) -> None:
        if os.name != "posix":
            raise StorageCoordinationNotSupportedError(
                "FilesystemKeyedCoordinator requires a POSIX platform "
                "(fcntl.flock); inject a distributed coordinator on non-POSIX"
            )
        if not hasattr(os, "O_NOFOLLOW"):
            # No symlink refusal available -- fail closed rather than opening a
            # path an attacker could substitute with a symlink.
            raise StorageCoordinationNotSupportedError(
                "FilesystemKeyedCoordinator requires O_NOFOLLOW to refuse "
                "symlink lock files; inject a distributed coordinator"
            )
        root_path = Path(root)
        locks_dir = root_path / ".locks"
        # Refuse a symlink at the locks path itself (an attacker pre-placing one
        # to redirect lock creation outside root). lstat distinguishes the link
        # from its target.
        if locks_dir.is_symlink():
            raise StorageCoordinationNotSupportedError(
                "coordination locks directory is a symlink; refusing to use it"
            )
        # Create once at construction, mode 0700 (owner-only). exist_ok tolerates
        # a prior construction on the same root.
        locks_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(locks_dir, 0o700)
        # Must resolve inside root after creation (defense against a root that
        # is itself a relative or symlinked path escaping its stated location).
        # Non-strict: the root may not exist yet at construction.
        try:
            resolved = locks_dir.resolve(strict=False)
            root_resolved = root_path.resolve(strict=False)
        except OSError as exc:
            raise StorageCoordinationNotSupportedError(
                "coordination locks directory is not accessible"
            ) from exc
        try:
            resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise StorageCoordinationNotSupportedError(
                "coordination locks directory must live inside the root"
            ) from exc
        self._locks_dir = locks_dir

    @asynccontextmanager
    async def hold(self, key: str) -> AsyncIterator[None]:
        # Percent-encode so an arbitrary key (a digest, a capability path with
        # slashes, ...) always becomes a single path component under the locks
        # dir -- never a traversal.
        lock_path = self._locks_dir / urllib.parse.quote(key, safe="")
        if lock_path.parent != self._locks_dir:
            raise StorageCoordinationNotSupportedError(
                "resolved lock path escapes the coordination locks directory"
            )
        fd = await asyncio.to_thread(self._open_and_lock, lock_path)
        try:
            yield
        finally:
            await asyncio.to_thread(self._unlock_and_close, fd)

    def _open_and_lock(self, path: Path) -> int:
        fd = os.open(path, _LOCK_OPEN_FLAGS, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        return fd

    def _unlock_and_close(self, fd: int) -> None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


__all__: "list[str]" = ["FilesystemKeyedCoordinator"]
