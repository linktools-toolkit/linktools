#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime cache injection preserves source identity and resource ownership."""

import hashlib
from collections.abc import AsyncIterator

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeDomain, RuntimeStorage
from linktools.ai.storage import InMemoryContentCache, InMemoryObjectStore, ObjectRef


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
    cache = InMemoryContentCache(max_bytes=1024)
    objects = CountingObjects()
    ref = await _put(objects, b"persisted body")
    for _ in range(2):
        storage = RuntimeStorage.sqlite(
            tmp_path / "runtime.db", object_store=objects, content_cache=cache,
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
    cache = InMemoryContentCache(max_bytes=1024)
    present = CountingObjects()
    absent = CountingObjects()
    ref = await _put(present, b"persisted body")
    for source in (present, absent):
        storage = RuntimeStorage.sqlite(
            tmp_path / "runtime.db", object_store=source, content_cache=cache,
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
