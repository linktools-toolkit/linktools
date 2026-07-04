#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FileResourceCoordinator: for shared-filesystem deployments. Only meaningful when
every instance shares the same filesystem; never stores Resource content, never
replaces the database transaction as the source of correctness (spec section 17)."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator


class FileResourceCoordinator:
    def __init__(self, *, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._hint_file = self._root / "revision-hint"
        self._locks: "dict[str, asyncio.Lock]" = {}

    async def revision_hint(self) -> "int | None":
        if not self._hint_file.exists():
            return None
        text = self._hint_file.read_text().strip()
        return int(text) if text else None

    async def publish_revision(self, revision: int) -> None:
        self._hint_file.write_text(str(revision))

    @asynccontextmanager
    async def lock(self, key: str) -> "AsyncIterator[None]":
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            yield
