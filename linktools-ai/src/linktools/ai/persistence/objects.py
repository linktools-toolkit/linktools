#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned opaque Object key derivation."""

import hashlib
from collections.abc import AsyncIterator

from ..core import RUNTIME_OBJECT_STORE_ID
from ..errors import AIError, ErrorCode
from ..storage import (
    ObjectRef,
    ObjectStore,
    read_object,
)
from ._object_key import ObjectKeyFactory
from ._plan import PersistenceDomain


async def put_runtime_object(
    store: ObjectStore,
    factory: ObjectKeyFactory,
    runtime_domain: PersistenceDomain,
    tenant_id: str,
    data: bytes,
) -> ObjectRef:
    digest = hashlib.sha256(data).hexdigest()
    key = factory.key(runtime_domain, tenant_id, digest)

    async def chunks() -> AsyncIterator[bytes]:
        yield data

    stat = await store.put(key, chunks(), expected_size=len(data), expected_digest=digest)
    return ObjectRef(RUNTIME_OBJECT_STORE_ID, key, stat.digest, stat.size)


async def read_runtime_object(store: ObjectStore, reference: ObjectRef) -> bytes:
    if reference.store_id != RUNTIME_OBJECT_STORE_ID:
        raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
    return await read_object(
        store,
        reference.key,
        expected_digest=reference.digest,
        expected_size=reference.size,
    )


from ._object_cleanup import ObjectCleanupResult
from ._object_router import runtime_object_source_reads

__all__ = ['ObjectCleanupResult', 'ObjectKeyFactory', 'put_runtime_object', 'read_runtime_object', 'runtime_object_source_reads']
