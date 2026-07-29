"""Ordered keyed locks for single-process local stores."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class KeyedLocks:
    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._locks: dict[tuple[str, str], tuple[asyncio.Lock, int]] = {}

    @asynccontextmanager
    async def acquire(self, *keys: tuple[str, str]) -> AsyncIterator[None]:
        rank = {"session": 0, "run": 1}
        ordered = tuple(sorted(set(keys), key=lambda item: (rank.get(item[0], 2), item)))
        acquired: list[asyncio.Lock] = []
        async with self._guard:
            locks = []
            for key in ordered:
                lock, users = self._locks.get(key, (asyncio.Lock(), 0))
                self._locks[key] = (lock, users + 1)
                locks.append((key, lock))
        try:
            for _, lock in locks:
                await lock.acquire()
                acquired.append(lock)
            yield
        finally:
            for lock in reversed(acquired):
                lock.release()
            async with self._guard:
                for key, lock in locks:
                    current, users = self._locks[key]
                    if current is lock and users == 1 and not lock.locked():
                        self._locks.pop(key, None)
                    else:
                        self._locks[key] = (current, users - 1)


__all__ = ["KeyedLocks"]
