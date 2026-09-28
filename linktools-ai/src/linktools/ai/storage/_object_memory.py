#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In-memory ObjectStore implementations."""

import hashlib
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path

from ..errors import AIError, ErrorCode
from ._object import (
    ObjectStat,
    ObjectStore,
    _CHUNK_SIZE,
    _digest,
    _validate_key,
    _validate_store_id,
)


class InMemoryObjectStore:
    def __init__(self, store_id: str = "memory") -> None:
        _validate_store_id(store_id)
        self._store_id = store_id
        self._objects: dict[str, bytes] = {}

    @property
    def store_id(self) -> str:
        return self._store_id

    def local_paths(self) -> tuple[Path, ...]:
        return ()

    async def put(
        self,
        key: str,
        chunks: AsyncIterator[bytes],
        *,
        expected_size: int,
        expected_digest: str,
    ) -> ObjectStat:
        data, digest = await _spool_memory(chunks, expected_size)
        if len(data) != expected_size or digest != expected_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        current = self._objects.get(key)
        if current is not None and current != data:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._objects[key] = data
        return ObjectStat(key, digest, len(data))

    async def stat(self, key: str) -> ObjectStat | None:
        _validate_key(key)
        value = self._objects.get(key)
        return None if value is None else ObjectStat(key, _digest(value), len(value))

    async def validate_integrity(self) -> None:
        return None

    @asynccontextmanager
    async def offline_exclusivity(self) -> AsyncIterator[None]:
        yield

    async def _list_objects(self) -> AsyncIterator[ObjectStat]:
        for key, value in self._objects.items():
            yield ObjectStat(key, _digest(value), len(value))

    def list_objects(self) -> AsyncIterator[ObjectStat]:
        return self._list_objects()

    async def delete_object(self, key: str, *, expected_digest: str) -> bool:
        _validate_key(key)
        value = self._objects.get(key)
        if value is None:
            return False
        if _digest(value) != expected_digest:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        del self._objects[key]
        return True

    async def _open(self, key: str) -> AsyncIterator[bytes]:
        _validate_key(key)
        value = self._objects.get(key)
        if value is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        for offset in range(0, len(value), _CHUNK_SIZE):
            yield value[offset : offset + _CHUNK_SIZE]

    def open(self, key: str) -> AsyncIterator[bytes]:
        return self._open(key)


class TransientObjectStore(InMemoryObjectStore):
    def __init__(self) -> None:
        super().__init__("transient")
        self._scopes: dict[str, set[str]] = {}

    def scoped(self, scope: str) -> ObjectStore:
        if not scope or "/" in scope:
            raise ValueError("transient object scope is invalid")
        return _ScopedObjectStore(self, scope)

    async def release_scope(self, scope: str) -> None:
        for key in self._scopes.pop(scope, set()):
            self._objects.pop(key, None)

    def clear(self) -> None:
        self._objects.clear()
        self._scopes.clear()


class _ScopedObjectStore:
    def __init__(self, parent: TransientObjectStore, scope: str) -> None:
        self._parent = parent
        self._scope = scope

    @property
    def store_id(self) -> str:
        return self._parent.store_id

    def local_paths(self) -> tuple[Path, ...]:
        return self._parent.local_paths()

    async def put(
        self,
        key: str,
        chunks: AsyncIterator[bytes],
        *,
        expected_size: int,
        expected_digest: str,
    ) -> ObjectStat:
        physical_key = self._physical_key(key)
        value = await self._parent.put(
            physical_key,
            chunks,
            expected_size=expected_size,
            expected_digest=expected_digest,
        )
        self._parent._scopes.setdefault(self._scope, set()).add(physical_key)
        return ObjectStat(key, value.digest, value.size)

    async def stat(self, key: str) -> ObjectStat | None:
        value = await self._parent.stat(self._physical_key(key))
        return None if value is None else ObjectStat(key, value.digest, value.size)

    async def validate_integrity(self) -> None:
        await self._parent.validate_integrity()

    def offline_exclusivity(self) -> AbstractAsyncContextManager[None]:
        return self._parent.offline_exclusivity()

    async def _list_objects(self) -> AsyncIterator[ObjectStat]:
        values = [value async for value in self._parent.list_objects()]
        prefix = f"{self._scope}/"
        for value in values:
            if value.key.startswith(prefix):
                yield ObjectStat(value.key[len(prefix) :], value.digest, value.size)

    def list_objects(self) -> AsyncIterator[ObjectStat]:
        return self._list_objects()

    async def delete_object(self, key: str, *, expected_digest: str) -> bool:
        return await self._parent.delete_object(
            self._physical_key(key),
            expected_digest=expected_digest,
        )

    def open(self, key: str) -> AsyncIterator[bytes]:
        return self._parent.open(self._physical_key(key))

    def _physical_key(self, key: str) -> str:
        _validate_key(key)
        return f"{self._scope}/{key}"


async def _spool_memory(
    chunks: AsyncIterator[bytes],
    expected_size: int,
) -> tuple[bytes, str]:
    data = bytearray()
    digest = hashlib.sha256()
    async for chunk in chunks:
        if not isinstance(chunk, bytes) or not chunk:
            raise ValueError("object chunks must be non-empty bytes")
        data.extend(chunk)
        digest.update(chunk)
        if len(data) > expected_size:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return bytes(data), digest.hexdigest()


__all__ = ["InMemoryObjectStore", "TransientObjectStore"]
