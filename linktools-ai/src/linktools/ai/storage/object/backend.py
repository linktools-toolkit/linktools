#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The storage kernel's external-backend Protocols (the adapter surface).

An external backend implements ``ObjectReaderBackend`` (raw physical read +
list + native revision). A backend that accepts writes implements
``ObjectWriterBackend`` (the atomic checked operations that fold
precondition-check + idempotency-reservation + mutate into ONE call so a
concurrent writer cannot interleave the three steps). ``TransactionalObjectBackend``
adds multi-object transactions; ``VersionedObjectBackend`` adds per-key history.

The single-backend ``ObjectStore`` wraps a writer backend with public input
validation + request-hash + error semantics; ``OverlayObjectStore`` composes a
primary writer over ordered reader overlays. Domains depend on the ObjectStore
Protocols (protocols.py), never on these backend Protocols directly."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Protocol, runtime_checkable

from .models import (
    Depth,
    LookupResult,
    ObjectInfo,
    ObjectPage,
    ObjectVersionPage,
    StorageKey,
    StoredObject,
    WriteOptions,
)


@runtime_checkable
class ObjectReaderBackend(Protocol):
    """Raw read surface every backend implements (including read-only overlays).
    ``backend_id`` is a stable identifier the OverlayObjectStore tags so a
    multi-backend listing cursor can attribute each item to its source."""

    backend_id: str

    async def raw_get(
        self, key: StorageKey, *, include_content: bool = True
    ) -> LookupResult: ...

    async def raw_stat(self, key: StorageKey) -> "ObjectInfo | None": ...

    async def raw_list(
        self,
        prefix: StorageKey,
        *,
        depth: "Depth",
        limit: int,
        cursor: "str | None",
    ) -> ObjectPage: ...

    async def revision(self) -> str: ...


@runtime_checkable
class ObjectWriterBackend(ObjectReaderBackend, Protocol):
    """Write surface. The checked operations fold precondition-check +
    idempotency-reservation + mutate into ONE atomic call (a backend with a
    real transaction primitive runs all three inside it; a process-local
    backend serializes with a lock). ``raw_move_checked`` is one atomic
    operation (load-source + write-target + tombstone-source + bump-revision);
    it never decomposes into a public put + delete.

    A read-only backend implements only ObjectReaderBackend -- it lacks these
    write methods, so it does not satisfy this Protocol and cannot be a
    primary. Read-only-ness is structural (which methods exist), not a flag."""

    async def raw_put_checked(
        self,
        key: StorageKey,
        content: bytes,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> StoredObject: ...

    async def raw_delete_checked(
        self,
        key: StorageKey,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> None: ...

    async def raw_move_checked(
        self,
        source: StorageKey,
        target: StorageKey,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> StoredObject: ...


@runtime_checkable
class TransactionalObjectBackend(Protocol):
    """Multi-object transaction capability. A backend WITHOUT this Protocol
    (e.g. the filesystem backend, which is process-local checked-write only)
    refuses multi-object transactions rather than faking atomicity.

    ``transaction()`` yields a transaction-bound ``ObjectWriterBackend``: the
    child owns all transaction-local state (staged writes, session, revision).
    The reusable parent backend remains free of active session or staged
    mutation state. Reads through the child see staged writes (read-your-
    writes); writes through the child are committed atomically on clean
    context exit and dropped on exception."""

    def transaction(
        self,
    ) -> "AbstractAsyncContextManager[ObjectWriterBackend]":
        ...


@runtime_checkable
class VersionedObjectBackend(Protocol):
    """Per-key history capability. Backends that keep prior versions implement
    these raw history reads; a backend without history (directory reader) omits
    the Protocol entirely."""

    async def raw_get_version(self, key: StorageKey, version: int) -> "StoredObject | None": ...

    async def raw_get_at_revision(self, key: StorageKey, revision: int) -> "StoredObject | None": ...

    async def raw_list_versions(
        self,
        key: StorageKey,
        *,
        limit: int,
        cursor: "str | None",
    ) -> ObjectVersionPage: ...

    async def raw_list_at_revision(self, prefix: StorageKey, revision: int) -> "tuple[ObjectInfo, ...]": ...
