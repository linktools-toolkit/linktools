#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime cache injection preserves source identity and resource ownership."""

import hashlib
from collections.abc import AsyncIterator

import pytest
from linktools.ai.asset import AssetContentCache, AssetInfo, AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeDomain, RuntimeStorage
from linktools.ai.storage import ContentCache, InMemoryCacheBackend, InMemoryObjectStore, ObjectRef, StorageOverlay, InMemoryStreamBackend


class CountingObjects(InMemoryObjectStore):
    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    async def open(self, key: str) -> AsyncIterator[bytes]:
        self.reads += 1
        async for chunk in super().open(key):
            yield chunk


async def _put(store: InMemoryObjectStore, content: bytes) -> ObjectRef:
    async def chunks() -> AsyncIterator[bytes]:
        yield content

    digest = hashlib.sha256(content).hexdigest()
    await store.put("body", chunks(), expected_size=len(content), expected_digest=digest)
    return ObjectRef(store.store_id, "body", digest, len(content))


@pytest.mark.asyncio
async def test_shared_cache_reuses_body_across_storage_lifetimes(tmp_path) -> None:
    cache = InMemoryCacheBackend(max_bytes=1024)
    objects = CountingObjects()
    ref = await _put(objects, b"persisted body")
    for _ in range(2):
        storage = RuntimeStorage.sqlite(
            tmp_path / "runtime.db", object_store=objects, cache=cache,
        )
        await storage.initialize(namespace="runtime", tenant_id="tenant")
        try:
            assert await storage.read_object(RuntimeDomain.EXECUTION, ref) == b"persisted body"
        finally:
            await storage.close()
        with pytest.raises(AIError) as error:
            await storage.read_object(RuntimeDomain.EXECUTION, ref)
        assert error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
    assert objects.reads == 1


@pytest.mark.asyncio
async def test_warm_shared_cache_does_not_replace_another_origins_membership(tmp_path) -> None:
    cache = InMemoryCacheBackend(max_bytes=1024)
    present = CountingObjects()
    absent = CountingObjects()
    ref = await _put(present, b"persisted body")
    for source in (present, absent):
        storage = RuntimeStorage.sqlite(
            tmp_path / "runtime.db", object_store=source, cache=cache,
        )
        await storage.initialize(namespace="runtime", tenant_id="tenant")
        try:
            if source is present:
                assert await storage.read_object(RuntimeDomain.EXECUTION, ref) == b"persisted body"
            else:
                with pytest.raises(AIError) as error:
                    await storage.read_object(RuntimeDomain.EXECUTION, ref)
                assert error.value.code is ErrorCode.STORAGE_NOT_FOUND
        finally:
            await storage.close()
    assert absent.reads == 0


@pytest.mark.asyncio
async def test_domain_content_cache_and_runtime_borrow_same_backend(tmp_path) -> None:
    class BorrowedBackend(InMemoryCacheBackend):
        def __init__(self) -> None:
            super().__init__(max_bytes=4096)
            self.closes = 0

        async def close(self) -> None:
            self.closes += 1

    cache_backend = BorrowedBackend()
    content_cache: ContentCache[AssetKey, bytes, AssetInfo] = AssetContentCache(cache_backend)
    assets = InMemoryAssetBackend()
    asset_store = AssetStore(StorageOverlay(
        assets, writer=assets, content_cache=content_cache,
    ))
    objects = CountingObjects()
    ref = await _put(objects, b"runtime body")
    storage = RuntimeStorage.sqlite(
        tmp_path / "runtime.db", object_store=objects, cache=cache_backend,
    )
    await asset_store.initialize()
    await storage.initialize(namespace="runtime", tenant_id="tenant")
    try:
        key = AssetKey("text", "asset")
        info = await asset_store.put(key, b"asset body")
        assert await asset_store.get(key) == b"asset body"
        assert await storage.read_object(RuntimeDomain.EXECUTION, ref) == b"runtime body"
        await asset_store.close()
        assert await storage.read_object(RuntimeDomain.EXECUTION, ref) == b"runtime body"
        assert objects.reads == 1
        cached = await cache_backend.get(content_cache.cache_key(key, info))
        assert cached is not None and content_cache.from_cache(cached) == b"asset body"
    finally:
        await asset_store.close()
        await storage.close()
    assert content_cache.backend is cache_backend
    assert cache_backend.closes == 0


class WriteOnlyCache:
    async def append(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("Runtime must not use write batches as content")

    async def read_batch(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("Runtime must not use write batches as content")

    async def ack(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("Runtime must not use write batches as content")


@pytest.mark.asyncio
@pytest.mark.parametrize("cache", (WriteOnlyCache(), InMemoryStreamBackend(max_bytes=1024)))
async def test_cache_without_key_value_capabilities_reads_objects_from_source(tmp_path, cache) -> None:
    objects = CountingObjects()
    reference = await _put(objects, b"source body")
    storage = RuntimeStorage.sqlite(tmp_path / "state.db", object_store=objects, cache=cache)
    await storage.initialize(namespace="runtime", tenant_id="tenant")
    try:
        for _ in range(2):
            assert await storage.read_object(RuntimeDomain.EXECUTION, reference) == b"source body"
        assert objects.reads == 2
    finally:
        await storage.close()
