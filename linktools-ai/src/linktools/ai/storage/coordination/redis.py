#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RedisResourceCoordinator: revision-change hints and optional distributed locking
via a Redis SET NX lock. Never stores Resource content."""

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

_RELEASE_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


class RedisResourceCoordinator:
    def __init__(self, *, redis, key_prefix: str = "linktools:ai:resource") -> None:
        self._redis = redis
        self._key_prefix = key_prefix

    def _hint_key(self) -> str:
        return f"{self._key_prefix}:revision"

    def _lock_key(self, key: str) -> str:
        return f"{self._key_prefix}:lock:{key}"

    async def revision_hint(self) -> "int | None":
        value = await self._redis.get(self._hint_key())
        return int(value) if value is not None else None

    async def publish_revision(self, revision: int) -> None:
        await self._redis.set(self._hint_key(), str(revision))

    @asynccontextmanager
    async def lock(self, key: str) -> "AsyncIterator[None]":
        token = uuid.uuid4().hex
        lock_key = self._lock_key(key)
        while not await self._redis.set(lock_key, token, nx=True, ex=30):
            await asyncio.sleep(0.01)
        try:
            yield
        finally:
            await self._redis.eval(_RELEASE_SCRIPT, 1, lock_key, token)
