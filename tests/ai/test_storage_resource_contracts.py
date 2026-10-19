#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Borrowed cache capabilities and persistence resource identity."""

from collections.abc import Hashable, Sequence

import pytest

from linktools.ai.storage import (
    BatchWriteScope,
    CacheBackend,
    InMemoryCacheBackend,
    StorageReadCache,
    StorageRoute,
    WriteBufferBackend,
    WriteBufferEntry,
)


class _WriteOnly:
    def __init__(self) -> None:
        self.backend = InMemoryCacheBackend(max_bytes=1024)

    async def append(
        self, scope: str, id: str, key: Hashable, content: bytes | None, *, wait: bool,
    ) -> bool:
        return await self.backend.append(scope, id, key, content, wait=wait)

    async def read_batch(self, scope: str, *, limit: int) -> tuple[WriteBufferEntry, ...]:
        return await self.backend.read_batch(scope, limit=limit)

    async def ack(self, scope: str, ids: Sequence[str]) -> None:
        await self.backend.ack(scope, ids)


class _HotOnly:
    def __init__(self) -> None:
        self.items: dict[str, bytes] = {}

    async def get(self, key: str) -> bytes | None:
        return self.items.get(key)

    async def put(self, key: str, content: bytes) -> None:
        self.items[key] = content

    async def delete(self, key: str) -> None:
        self.items.pop(key, None)


@pytest.mark.parametrize("kind", ["memory", "transient", "filesystem", "sqlite"])
def test_storage_route_borrows_cache_without_changing_backend_identity(kind, tmp_path) -> None:
    factory = getattr(StorageRoute, kind)
    args = () if kind in {"memory", "transient"} else (tmp_path / "state",)
    cache = _HotOnly()
    options = {"transaction_root": tmp_path} if kind == "filesystem" else {}
    plain = factory(*args, **options)
    cached = factory(*args, cache=cache, **options)
    assert cached.cache is cache
    assert plain.cache is None
    assert cached == plain
    assert hash(cached) == hash(plain)
    assert repr(cached) == repr(plain)
    replacement = _WriteOnly()
    copied = cached.with_cache(replacement)
    assert copied is not cached
    assert copied == cached
    assert copied.transaction_root == cached.transaction_root
    assert copied.cache is replacement
    assert cached.cache is cache
    assert copied.with_cache(None).cache is None


@pytest.mark.asyncio
async def test_write_only_cache_supports_batch_scope_without_hot_methods() -> None:
    cache = _WriteOnly()
    assert isinstance(cache, WriteBufferBackend)
    assert not isinstance(cache, CacheBackend)
    received: list[WriteBufferEntry] = []

    async def write(entries: tuple[WriteBufferEntry, ...]) -> None:
        received.extend(entries)

    scope = BatchWriteScope(cache, write, batch_size=1)
    await scope.submit("key", b"value")
    await scope.close()
    assert [(key, value) for _, key, value in received] == [("key", b"value")]


@pytest.mark.asyncio
async def test_hot_only_cache_supports_read_cache_without_buffer_methods() -> None:
    cache = _HotOnly()
    assert isinstance(cache, CacheBackend)
    assert not isinstance(cache, WriteBufferBackend)
    reader = StorageReadCache(cache)
    assert StorageRoute.memory(cache=cache).cache is cache
    await reader.put("key", b"cached")

    async def load() -> bytes:
        raise AssertionError("fresh cache hit must not load the source")

    assert await reader.get("key", load) == b"cached"
    await reader.close()


def test_runtime_storage_factories_assign_cache_to_each_route() -> None:
    from linktools.ai.runtime import RuntimeDomain, RuntimeStorage

    cache = _HotOnly()
    storage = RuntimeStorage.in_memory(cache=cache)
    assert storage.cache is cache
    assert all(storage.plan.route(domain).cache is cache for domain in RuntimeDomain)


@pytest.mark.asyncio
async def test_explicit_module_resource_clears_cache_while_omission_inherits() -> None:
    from linktools.ai.runtime import RuntimeDomain, RuntimeStorage

    cache = _HotOnly()
    storage = RuntimeStorage.in_memory(cache=cache)
    execution = storage.plan.route(RuntimeDomain.EXECUTION)
    await storage.initialize(
        namespace="resources", tenant_id="tenant",
        module_routes={RuntimeDomain.TRACE: execution.with_cache(None)},
    )
    try:
        assert storage.plan.route(RuntimeDomain.TRACE).cache is None
        assert storage.plan.route(RuntimeDomain.CHECKPOINT).cache is cache
        assert storage.plan.route(RuntimeDomain.EXECUTION).cache is cache
    finally:
        await storage.close()
