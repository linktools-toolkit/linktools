#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filesystem-backed artifact digest coordinator.

One ``flock(LOCK_EX)`` lock file per digest under ``<root>/.locks/``. ``flock``
is advisory and per-open-file-description, so two processes that both open the
same lock file and flock it are mutually exclusive -- the coordination spans
process boundaries on a shared filesystem. Every file operation (open, lock,
unlock, close) runs via ``asyncio.to_thread`` so blocking POSIX I/O never runs
on the event loop.

POSIX-only: ``fcntl`` is unavailable on Windows. The constructor raises
:class:`UnsupportedArtifactCoordinationError` off-POSIX rather than pretending
to coordinate. The production runtime is Linux; tests run on Linux CI."""

import asyncio
import fcntl
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from ...artifact.coordination import UnsupportedArtifactCoordinationError


class FilesystemArtifactDigestCoordinator:
    """Per-digest ``flock(LOCK_EX)`` over a lock file under ``root/.locks/``.
    Coordinates across processes sharing the filesystem; use it for
    FilesystemArtifactBlobStore-backed storage that may be swept by a separate
    worker process."""

    def __init__(self, *, root: Path) -> None:
        if os.name != "posix":
            raise UnsupportedArtifactCoordinationError(
                "FilesystemArtifactDigestCoordinator requires a POSIX platform "
                "(fcntl.flock); inject a distributed coordinator on non-POSIX"
            )
        self._locks_dir = Path(root) / ".locks"

    @asynccontextmanager
    async def hold(self, digest: str) -> AsyncIterator[None]:
        path = self._locks_dir / digest
        fd = await asyncio.to_thread(self._open_and_lock, path)
        try:
            yield
        finally:
            await asyncio.to_thread(self._unlock_and_close, fd)

    def _open_and_lock(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        return fd

    def _unlock_and_close(self, fd: int) -> None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


__all__: "list[str]" = ["FilesystemArtifactDigestCoordinator"]
