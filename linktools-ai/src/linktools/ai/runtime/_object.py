#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned opaque Object key derivation."""

import hashlib
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

from ..core import validate_tenant_id
from ..errors import AIError, ErrorCode
from ..storage import ObjectRef, ObjectStore, namespace_key, read_object
from .state._plan import RuntimeDomain

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_OBJECT_DOMAINS = frozenset({RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.MEMORY, RuntimeDomain.ARTIFACT, RuntimeDomain.RECOVERY})


@dataclass(frozen=True, slots=True)
class RuntimeObjectKeyFactory:
    namespace: str

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str) or not self.namespace:
            raise ValueError("namespace must be a non-empty string")

    @property
    def namespace_key(self) -> str:
        return namespace_key(self.namespace)

    def key(self, runtime_domain: RuntimeDomain, tenant_id: str, digest: str) -> str:
        if runtime_domain not in _OBJECT_DOMAINS:
            raise ValueError("Runtime object identity is invalid")
        tenant_id = validate_tenant_id(tenant_id)
        if _DIGEST.fullmatch(digest) is None:
            raise ValueError("Runtime object digest is invalid")
        tenant_scope_key = hashlib.sha256(("tenant:" + tenant_id).encode("utf-8")).hexdigest()
        return f"v1/runtime/{self.namespace_key}/{runtime_domain.value}/{tenant_scope_key}/{digest}"


async def put_runtime_object(
    store: ObjectStore,
    factory: RuntimeObjectKeyFactory,
    runtime_domain: RuntimeDomain,
    tenant_id: str,
    data: bytes,
) -> ObjectRef:
    digest = hashlib.sha256(data).hexdigest()
    key = factory.key(runtime_domain, tenant_id, digest)

    async def chunks() -> AsyncIterator[bytes]:
        yield data

    stat = await store.put(key, chunks(), expected_size=len(data), expected_digest=digest)
    return ObjectRef(store.store_id, key, stat.digest, stat.size)


async def read_runtime_object(store: ObjectStore, reference: ObjectRef) -> bytes:
    if reference.store_id != store.store_id:
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
    return await read_object(store, reference.key, expected_digest=reference.digest, expected_size=reference.size)


__all__ = ["RuntimeObjectKeyFactory", "put_runtime_object", "read_runtime_object"]
