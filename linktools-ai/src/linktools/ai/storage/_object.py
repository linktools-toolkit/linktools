#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable ObjectStore contracts and backend-neutral helpers."""

import asyncio
import hashlib
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

from ..errors import AIError, ErrorCode

_CHUNK_SIZE = 1024 * 1024
_TaskT = TypeVar("_TaskT")


@dataclass(frozen=True, slots=True)
class ObjectRef:
    store_id: str
    key: str
    digest: str
    size: int

    def __post_init__(self) -> None:
        _validate_store_id(self.store_id)
        _validate_key(self.key)
        _validate_digest(self.digest)
        if not isinstance(self.size, int) or self.size < 0:
            raise ValueError("object size must be non-negative")


@dataclass(frozen=True, slots=True)
class ObjectStat:
    key: str
    digest: str
    size: int


class ObjectStore(Protocol):
    @property
    def store_id(self) -> str: ...

    async def put(
        self,
        key: str,
        chunks: AsyncIterator[bytes],
        *,
        expected_size: int,
        expected_digest: str,
    ) -> ObjectStat: ...

    async def stat(self, key: str) -> ObjectStat | None: ...

    async def validate_integrity(self) -> None: ...

    def open(self, key: str) -> AsyncIterator[bytes]: ...

    def local_paths(self) -> tuple[Path, ...]: ...


def runtime_object_key(
    *,
    namespace_digest: str,
    tenant_digest: str,
    stored_digest: str,
) -> str:
    """Build the tenant-scoped physical key for immutable Runtime bytes."""
    for value in (namespace_digest, tenant_digest, stored_digest):
        _validate_digest(value)
    return f"v1/runtime/{namespace_digest}/{tenant_digest}/{stored_digest}"


@runtime_checkable
class ObjectStoreInspection(Protocol):
    def list_objects(self) -> AsyncIterator[ObjectStat]: ...


@runtime_checkable
class ObjectStoreMaintenance(ObjectStoreInspection, Protocol):
    async def delete_object(self, key: str, *, expected_digest: str) -> bool: ...

    def offline_exclusivity(self) -> AbstractAsyncContextManager[None]: ...


async def read_object(
    store: ObjectStore,
    key: str,
    *,
    expected_digest: str,
    expected_size: int,
) -> bytes:
    digest = hashlib.sha256()
    size = 0
    data = bytearray()
    async for chunk in store.open(key):
        data.extend(chunk)
        digest.update(chunk)
        size += len(chunk)
    if size != expected_size or digest.hexdigest() != expected_digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return bytes(data)


async def _settle_task(task: "asyncio.Task[_TaskT]") -> tuple[_TaskT, bool]:
    cancelled = False
    while True:
        try:
            return await asyncio.shield(task), cancelled
        except asyncio.CancelledError:
            if task.done():
                if task.cancelled():
                    raise
                return task.result(), True
            cancelled = True


async def _finish_owned_task(task: "asyncio.Task[_TaskT]") -> _TaskT:
    value, cancelled = await _settle_task(task)
    if cancelled:
        raise asyncio.CancelledError
    return value


async def _spool_file(
    chunks: AsyncIterator[bytes],
    path: Path,
    expected_size: int,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    buffer = bytearray()
    handle = await asyncio.to_thread(path.open, "wb")
    try:
        async for chunk in chunks:
            if not isinstance(chunk, bytes) or not chunk:
                raise ValueError("object chunks must be non-empty bytes")
            size += len(chunk)
            if size > expected_size:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            digest.update(chunk)
            for offset in range(0, len(chunk), _CHUNK_SIZE):
                buffer.extend(chunk[offset : offset + _CHUNK_SIZE])
                while len(buffer) >= _CHUNK_SIZE:
                    value = bytes(buffer[:_CHUNK_SIZE])
                    del buffer[:_CHUNK_SIZE]
                    await asyncio.to_thread(handle.write, value)
        if buffer:
            await asyncio.to_thread(handle.write, bytes(buffer))
        await asyncio.to_thread(handle.flush)
    finally:
        await asyncio.to_thread(handle.close)
    return size, digest.hexdigest()


def _track_object_task(
    tasks: set[asyncio.Task[Any]],
    task: asyncio.Task[Any],
    label: str,
) -> None:
    del label
    tasks.add(task)
    task.add_done_callback(tasks.discard)


def _key_digest(store_id: str, key: str) -> bytes:
    return hashlib.sha256(
        store_id.encode("utf-8") + b"\0" + key.encode("utf-8")
    ).digest()


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_store_id(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or any(
            character
            not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
            for character in value
        )
    ):
        raise ValueError("object store id is invalid")


def _validate_key(value: str) -> None:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("object key is invalid")


def _validate_digest(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("object digest is invalid")


def _validate_put(key: str, size: int, digest: str) -> None:
    _validate_key(key)
    _validate_digest(digest)
    if not isinstance(size, int) or size < 0:
        raise ValueError("object size is invalid")


__all__ = [
    "ObjectRef",
    "ObjectStat",
    "ObjectStore",
    "ObjectStoreInspection",
    "ObjectStoreMaintenance",
    "read_object",
    "runtime_object_key",
]
