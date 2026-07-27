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
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from .backend import ObjectWriterBackend, TransactionalObjectBackend
from .errors import StorageObjectError, StorageTransactionNotSupportedError
from .models import Depth, Found, ObjectInfo, ObjectPage, StorageKey, StoredObject, WriteOptions


def _request_hash(*parts: bytes) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return hasher.hexdigest()


def _content_cache_key(key: str, version: int, etag: str) -> str:
    """Versioned content-cache key. Embedding both ``version`` and ``etag``
    guarantees a new version can never be satisfied by an old cache entry —
    write invalidation is automatic, no explicit eviction needed."""
    return f"object:{hashlib.sha256(key.encode('utf-8')).hexdigest()}:v{version}:{etag}"


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
    cannot be a primary). Read-only-ness is structural, not a flag.

    An optional ``cache`` (ContentCache) + ``index`` (RevisionedObjectIndex)
    pair wires a read-through cache: ``stat`` serves from the revision-gated
    index (0 SQL on hot path after first sync), and ``get`` checks the
    content cache keyed by version+etag (0 SQL on hit). Write invalidation
    is automatic — a new version produces a new cache key, so stale entries
    are never served. Transaction-bound stores (from ``transaction()``)
    bypass cache+index to preserve read-your-writes within the tx."""

    def __init__(
        self,
        *,
        primary: ObjectWriterBackend,
        metrics: Any = None,
        cache: Any = None,
        index: Any = None,
    ) -> None:
        if not isinstance(primary, ObjectWriterBackend):
            raise StorageObjectError(
                "the ObjectStore primary must be an ObjectWriterBackend; a "
                "read-only backend cannot be a primary"
            )
        self._primary = primary
        self._metrics = metrics
        self._cache = cache
        self._index = index

    @classmethod
    def _from_transaction_backend(
        cls,
        backend: ObjectWriterBackend,
        *,
        metrics: Any = None,
    ) -> "ObjectStore":
        """Build a lightweight ObjectStore wrapper around a transaction-bound
        child backend. The child is NOT re-validated (it was already validated
        by the parent's constructor); ``backend_id`` is whatever the child
        declares. Cache+index are NOT copied: a transaction must see its own
        staged writes (read-your-writes), which a cache layer would bypass."""
        instance = object.__new__(cls)
        instance._primary = backend
        instance._metrics = metrics
        instance._cache = None
        instance._index = None
        return instance

    @property
    def primary(self) -> ObjectWriterBackend:
        return self._primary

    @property
    def supports_optimistic_concurrency(self) -> bool:
        """True iff the primary backend enforces CAS preconditions
        atomically (``raw_put_checked``/``raw_delete_checked``/``raw_move_
        checked`` apply if_match / if_none_match inside the same atomic step
        as the mutate). Every backend that implements ObjectWriterBackend
        does this by contract; the property exists so a caller can read the
        capability without resorting to isinstance against a concrete class
        (P4 capability self-consistency)."""
        return isinstance(self._primary, ObjectWriterBackend)

    @property
    def transaction_scope(self) -> "TransactionScope":
        """The multi-object transaction range the backing backend actually
        provides: DATABASE when the primary implements TransactionalObjectBackend
        (real transaction child backend), NONE otherwise (single-key checked
        writes only)."""
        from ...storage.features import TransactionScope

        if isinstance(self._primary, TransactionalObjectBackend):
            return TransactionScope.DATABASE
        return TransactionScope.NONE

    @property
    def capabilities(self) -> "ComponentCapabilities":
        """Per-component capabilities derived from what the backing backend
        ACTUALLY does (P4 capability self-consistency). The object store
        participates in a transaction when its backend implements
        TransactionalObjectBackend; it offers optimistic concurrency iff it
        implements ObjectWriterBackend (CAS enforced atomically); it always
        supports idempotency keys via WriteOptions; it is not append-only
        (objects can be overwritten)."""
        from ...storage.features import ComponentCapabilities

        return ComponentCapabilities(
            transaction_participation=isinstance(self._primary, TransactionalObjectBackend),
            optimistic_concurrency=isinstance(self._primary, ObjectWriterBackend),
            idempotency=True,
            append_only=False,
        )

    # --- read ----------------------------------------------------------------

    async def get(self, key: StorageKey) -> "StoredObject | None":
        _require_persistable_key(key)
        if self._cache is not None and self._index is not None:
            info = await self._index.stat(key)
            if info is not None:
                cache_key = _content_cache_key(key.value, info.version, info.etag)
                cached = await self._cache.get(cache_key)
                if cached is not None:
                    return StoredObject(info=info, content=cached)
            # index miss or cache miss → fall through to origin
        lookup = await self._primary.raw_get(key)
        obj = lookup.object if isinstance(lookup, Found) else None
        if obj is not None and self._cache is not None:
            cache_key = _content_cache_key(
                key.value, obj.info.version, obj.info.etag
            )
            await self._cache.put(cache_key, obj.content)
        return obj

    async def stat(self, key: StorageKey) -> "ObjectInfo | None":
        if key.is_root:
            return None
        if self._index is not None:
            return await self._index.stat(key)
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

    # --- transaction ---------------------------------------------------------

    @asynccontextmanager
    async def transaction(self) -> "AsyncIterator[ObjectStore]":
        """Yield a transaction-bound ObjectStore. Reads through the child see
        staged writes (read-your-writes); writes through the child commit
        atomically on clean context exit and roll back on exception.

        The parent store remains stateless and safe for concurrent use. A
        backend without ``TransactionalObjectBackend`` raises
        ``StorageTransactionNotSupportedError``."""
        if not isinstance(self._primary, TransactionalObjectBackend):
            raise StorageTransactionNotSupportedError(
                f"{type(self._primary).__name__} does not support object transactions"
            )
        async with self._primary.transaction() as tx_backend:
            yield ObjectStore._from_transaction_backend(
                tx_backend,
                metrics=self._metrics,
            )
