#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FileResourceCoordinator: provides real cross-process mutual exclusion via
POSIX advisory file locks (fcntl.flock) when every instance shares the same
filesystem. Only meaningful under that condition; never stores Resource
content; never replaces the database transaction as the source of correctness."""

import asyncio
import fcntl
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator


class FileResourceCoordinator:
    def __init__(self, *, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._hint_file = self._root / "revision-hint"
        self._lock_dir = self._root / "locks"
        self._lock_dir.mkdir(parents=True, exist_ok=True)

    async def revision_hint(self) -> "int | None":
        if not self._hint_file.exists():
            return None
        text = self._hint_file.read_text().strip()
        return int(text) if text else None

    async def publish_revision(self, revision: int) -> None:
        self._hint_file.write_text(str(revision))

    def _lock_path(self, key: str) -> Path:
        safe_name = key.replace("/", "__")
        return self._lock_dir / f"{safe_name}.lock"

    @asynccontextmanager
    async def lock(self, key: str) -> "AsyncIterator[None]":
        lock_path = self._lock_path(key)

        def _acquire():
            fd = open(lock_path, "w")
            fcntl.flock(fd, fcntl.LOCK_EX)
            return fd

        def _release(fd):
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()

        fd = await asyncio.to_thread(_acquire)
        try:
            yield
        finally:
            await asyncio.to_thread(_release, fd)
