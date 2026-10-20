#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Archive reads preserve cache membership and isolate business-object sources."""

import hashlib
from collections.abc import AsyncIterator, Sequence

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state._object_router import _RuntimeObjectRouter, runtime_object_source_reads
from linktools.ai.runtime.state._plan import RuntimeDomain
from linktools.ai.storage import InMemoryObjectStore, ObjectRef


class Cache:
    def __init__(self) -> None:
        self.items: dict[str, bytes] = {}

    async def get(self, key: str) -> bytes | None:
        return self.items.get(key)

    async def put(self, key: str, content: bytes) -> None:
        self.items[key] = content

    async def delete(self, key: str) -> None:
        self.items.pop(key, None)

    async def contains_many(self, keys: Sequence[str]) -> frozenset[str]:
        return frozenset(key for key in keys if key in self.items)


class Objects(InMemoryObjectStore):
    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    async def open(self, key: str) -> AsyncIterator[bytes]:
        self.reads += 1
        async for chunk in super().open(key):
            yield chunk


@pytest.mark.asyncio
async def test_archive_cached_reads_validate_source_and_do_not_warm_business_store() -> None:
    archive, business, cache = Objects(), Objects(), Cache()
    domain = RuntimeDomain.EXECUTION
    router = _RuntimeObjectRouter({domain: business}, archive_stores={domain: archive},
                                  close_guard_stores=(), cache=cache, cache_domains=(domain,))
    payload = b"shared content"
    reference = ObjectRef("runtime", "v1/archive/scope/tenant/digest", hashlib.sha256(payload).hexdigest(), len(payload))

    async def chunks() -> AsyncIterator[bytes]:
        yield payload

    await archive.put(reference.key, chunks(), expected_digest=reference.digest, expected_size=reference.size)
    assert await router.read_archive_object(domain, reference) == payload
    assert await router.read_archive_object(domain, reference) == payload
    assert archive.reads == 1
    with pytest.raises(AIError) as error:
        await router.read_object(domain, reference)
    assert error.value.code is ErrorCode.STORAGE_NOT_FOUND
    await archive.delete_object(reference.key, expected_digest=reference.digest)
    with runtime_object_source_reads(), pytest.raises(AIError) as error:
        await router.read_archive_object(domain, reference)
    assert error.value.code is ErrorCode.STORAGE_NOT_FOUND
    await router.drain_reads()
