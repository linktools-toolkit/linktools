#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Best-effort file revision hints."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from ..filesystem.atomic import atomic_write_bytes


class FileRevisionHint:
    """Publish and read a small revision marker atomically."""

    def __init__(self, path: "str | Path") -> None:
        self.path = Path(path)

    async def read(self) -> "str | None":
        try:
            return await asyncio.to_thread(self.path.read_text, encoding="utf-8")
        except (FileNotFoundError, OSError):
            return None

    async def write(self, revision: str) -> None:
        await asyncio.to_thread(atomic_write_bytes, self.path, revision.encode("utf-8"))


class FileAssetCoordinator:
    """Coordinate a named local asset and publish its revision hint."""

    def __init__(self, root: "str | Path") -> None:
        self._root = Path(root)
        self._lock = asyncio.Lock()
        self._hint = FileRevisionHint(self._root / ".revision")

    @asynccontextmanager
    async def lock(self) -> "AsyncIterator[None]":
        async with self._lock:
            yield

    async def revision_hint(self) -> "str | None":
        return await self._hint.read()

    async def publish_revision(self, revision: str) -> None:
        await self._hint.write(revision)


__all__ = ["FileAssetCoordinator", "FileRevisionHint"]
