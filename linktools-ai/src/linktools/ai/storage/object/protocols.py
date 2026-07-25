#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The storage kernel's object-store Protocols.

A backend implements ``ObjectReader`` (read/list), optionally
``RevisionSource`` (a namespace revision token), and ``ObjectWriter``
(mutate with CAS + idempotency). A backend that supports per-key versioning
implements ``ObjectHistoryReader``. Domains depend on these narrow Protocols
-- never on a concrete backend or a SQLAlchemy session. ``RevisionSource`` is
deliberately OPTIONAL: a directory reader has no revision, so it simply does
not implement it."""

from __future__ import annotations

from typing import Protocol

from .models import Depth, ObjectInfo, ObjectPage, ObjectVersionPage, StorageKey, StoredObject, WriteOptions


class ObjectReader(Protocol):
    async def get(self, key: StorageKey) -> "StoredObject | None": ...

    async def stat(self, key: StorageKey) -> "ObjectInfo | None": ...

    async def list(
        self,
        prefix: StorageKey,
        *,
        depth: "Depth" = Depth.ONE,
        limit: int = 100,
        cursor: "str | None" = None,
    ) -> ObjectPage: ...


class RevisionSource(Protocol):
    async def revision(self) -> str: ...


class RevisionedObjectReader(ObjectReader, RevisionSource, Protocol):
    """A reader that ALSO exposes a namespace revision token. Backends without
    a revision (e.g. a plain directory reader) simply omit ``revision``."""


class ObjectWriter(ObjectReader, Protocol):
    async def put(
        self,
        key: StorageKey,
        content: bytes,
        *,
        options: WriteOptions = WriteOptions(),
    ) -> StoredObject: ...

    async def delete(
        self,
        key: StorageKey,
        *,
        options: WriteOptions = WriteOptions(),
    ) -> None: ...

    async def move(
        self,
        source: StorageKey,
        target: StorageKey,
        *,
        options: WriteOptions = WriteOptions(),
    ) -> StoredObject: ...


class ObjectHistoryReader(Protocol):
    async def get_version(self, key: StorageKey, version: int) -> "StoredObject | None": ...

    async def get_at_revision(
        self, key: StorageKey, revision: int
    ) -> "StoredObject | None": ...

    async def list_versions(
        self,
        key: StorageKey,
        *,
        limit: int = 100,
        cursor: "str | None" = None,
    ) -> ObjectVersionPage: ...

    async def list_at_revision(
        self, prefix: StorageKey, revision: int
    ) -> "tuple[ObjectInfo, ...]": ...
