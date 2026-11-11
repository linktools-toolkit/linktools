#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned opaque Object key derivation."""

import hashlib
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

from ..core import RUNTIME_OBJECT_STORE_ID, validate_persistence_namespace, validate_tenant_id
from ..errors import AIError, ErrorCode
from ..storage import (
    ObjectRef,
    ObjectStore,
    read_object,
)
from ._plan import PersistenceDomain, runtime_domain_uses_object_store

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class ObjectKeyFactory:
    namespace: str

    def __post_init__(self) -> None:
        validate_persistence_namespace(self.namespace)

    def key(self, runtime_domain: PersistenceDomain, tenant_id: str, digest: str) -> str:
        if not runtime_domain_uses_object_store(runtime_domain):
            raise ValueError("Runtime object identity is invalid")
        tenant_id = validate_tenant_id(tenant_id)
        if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
            raise ValueError("Runtime object digest is invalid")
        # Character lengths frame opaque components, including slashes and colons.
        return f"v1/runtime/{len(self.namespace)}:{self.namespace}/{len(tenant_id)}:{tenant_id}/{digest}"


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

__all__ = ['ObjectCleanupResult', 'ObjectKeyFactory', 'put_runtime_object', 'read_runtime_object', 'runtime_domain_uses_object_store', 'runtime_object_source_reads']
