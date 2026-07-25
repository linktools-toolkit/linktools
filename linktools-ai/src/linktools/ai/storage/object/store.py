#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The single-backend ObjectStore.

Wraps one ``ObjectWriterBackend`` with the public contract the storage kernel
exposes to domains: input validation (reject the namespace root for any content
operation), request-hash computation (so an idempotency key replayed with
different preconditions/content is detected as a conflict, not a silent
overwrite), and error semantics. The backend owns atomicity -- precondition +
idempotency + mutate run as ONE checked call -- so this layer adds no locking.

The multi-backend k-way merge + HMAC cursor + whiteout semantics live in
``overlay.py`` (OverlayObjectStore); this layer is single-backend only."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .backend import ObjectWriterBackend
from .errors import StorageObjectError
from .models import Depth, Found, ObjectInfo, ObjectPage, StorageKey, StoredObject, WriteOptions


def _request_hash(*parts: bytes) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return hasher.hexdigest()


def _require_persistable_key(key: StorageKey) -> None:
    """Reject the namespace root for any content operation (get/put/delete/
    move). The root is a synthetic directory, not a persistable object; only
    list/stat/revision accept it."""
    if key.is_root:
        raise StorageObjectError(
            "the namespace root (\"/\") is a synthetic directory, not a "
            "persistable object -- only list/stat/revision accept it"
        )


def _put_request_hash(key: StorageKey, content: bytes, options: WriteOptions) -> str:
    # The hash covers every input that changes the operation's meaning: two
    # PUTs with the same key/content but DIFFERENT preconditions (or
    # content_type/metadata) hash differently, so a replayed idempotency key
    # re-evaluates the differing inputs instead of returning the first call's
    # cached result.
    return _request_hash(
        b"put",
        key.value.encode(),
        content,
        (options.if_match or "").encode(),
        str(options.if_none_match).encode(),
        (options.idempotency_key or "").encode(),
        (options.content_type or "").encode(),
        json.dumps(dict(options.metadata or {}), sort_keys=True).encode(),
    )


def _delete_request_hash(key: StorageKey, options: WriteOptions) -> str:
    return _request_hash(
        b"delete",
        key.value.encode(),
        (options.if_match or "").encode(),
        (options.idempotency_key or "").encode(),
    )


def _move_request_hash(
    source: StorageKey, target: StorageKey, options: WriteOptions
) -> str:
    return _request_hash(
        b"move",
        source.value.encode(),
        target.value.encode(),
        (options.if_match or "").encode(),
        str(options.if_none_match).encode(),
        (options.idempotency_key or "").encode(),
    )


class ObjectStore:
    """Single-backend ObjectStore. The primary backend must be an
    ``ObjectWriterBackend`` (a read-only backend lacks the write methods and
    cannot be a primary). Read-only-ness is structural, not a flag."""

    def __init__(
        self,
        *,
        primary: ObjectWriterBackend,
        metrics: Any = None,
    ) -> None:
        if not isinstance(primary, ObjectWriterBackend):
            raise StorageObjectError(
                "the ObjectStore primary must be an ObjectWriterBackend; a "
                "read-only backend cannot be a primary"
            )
        primary.backend_id = "primary"
        self._primary = primary
        self._metrics = metrics

    @property
    def primary(self) -> ObjectWriterBackend:
        return self._primary

    # --- read ----------------------------------------------------------------

    async def get(self, key: StorageKey) -> "StoredObject | None":
        _require_persistable_key(key)
        lookup = await self._primary.raw_get(key)
        return lookup.object if isinstance(lookup, Found) else None

    async def stat(self, key: StorageKey) -> "ObjectInfo | None":
        if key.is_root:
            return None
        return await self._primary.raw_stat(key)

    async def list(
        self,
        prefix: StorageKey,
        *,
        depth: "Depth" = Depth.ONE,
        limit: int = 100,
        cursor: "str | None" = None,
    ) -> ObjectPage:
        return await self._primary.raw_list(prefix, depth=depth, limit=limit, cursor=cursor)

    async def revision(self) -> str:
        return await self._primary.revision()

    # --- write ---------------------------------------------------------------

    async def put(
        self,
        key: StorageKey,
        content: bytes,
        *,
        options: WriteOptions = WriteOptions(),
    ) -> StoredObject:
        _require_persistable_key(key)
        return await self._primary.raw_put_checked(
            key, content, options=options, request_hash=_put_request_hash(key, content, options)
        )

    async def delete(
        self, key: StorageKey, *, options: WriteOptions = WriteOptions()
    ) -> None:
        _require_persistable_key(key)
        await self._primary.raw_delete_checked(
            key, options=options, request_hash=_delete_request_hash(key, options)
        )

    async def move(
        self,
        source: StorageKey,
        target: StorageKey,
        *,
        options: WriteOptions = WriteOptions(),
    ) -> StoredObject:
        _require_persistable_key(source)
        _require_persistable_key(target)
        return await self._primary.raw_move_checked(
            source,
            target,
            options=options,
            request_hash=_move_request_hash(source, target, options),
        )
