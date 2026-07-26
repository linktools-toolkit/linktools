#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The in-memory object backend -- the storage kernel's reference implementation.

Implements every capability (revision, history, transaction, idempotency, CAS,
move) so the storage.object contract suite has one backend that exercises the
FULL behavior, parameterized over the same contract the production backends
must satisfy. Not durable -- process-local state only. Production deployments
use the filesystem or SQLAlchemy backend; this one is for tests + as the
behavioral reference other backends are checked against.

Transaction model: ``transaction()`` yields a transaction-bound child backend
(``_MemoryTransactionBackend``) that stages writes in a per-transaction state
object. The child provides read-your-writes (reads see parent + staged),
one-transaction-one-revision, and idempotency that rolls back with the
transaction. The parent backend has NO ambient transaction state."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import AsyncIterator

from ...object.errors import (
    StorageIdempotencyConflictError,
    StorageObjectNotFoundError,
    StoragePreconditionFailedError,
    StorageTransactionClosedError,
)
from ...object.models import (
    Depth,
    Found,
    LookupResult,
    Masked,
    Missing,
    ObjectInfo,
    ObjectPage,
    ObjectVersionPage,
    StorageKey,
    StoredObject,
    WriteOptions,
)


def _etag(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class _ReentrantAsyncLock:
    """A task-owner-aware re-entrant async lock. A transaction acquires it once
    for its whole duration; the checked ops it drives run in the SAME task and
    re-acquire it (a no-op bump of the count) rather than deadlocking on a
    non-re-entrant ``asyncio.Lock``. A concurrent writer in a DIFFERENT task
    blocks on the underlying lock until the transaction releases."""

    def __init__(self) -> None:
        self._gate = asyncio.Lock()
        self._owner: "asyncio.Task | None" = None
        self._depth = 0

    async def __aenter__(self) -> "_ReentrantAsyncLock":
        task = asyncio.current_task()
        if self._owner is task:
            self._depth += 1
            return self
        await self._gate.acquire()
        self._owner = task
        self._depth = 1
        return self

    async def __aexit__(self, *exc) -> None:
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
            self._gate.release()


@dataclass
class _Record:
    """One version of a key: a live object (content set) or a tombstone
    (content None). Appended immutably to a key's history; the last record
    is the current state."""

    info: ObjectInfo
    content: "bytes | None"
    tombstone: bool = False


@dataclass
class _IdempotencyEntry:
    request_hash: str
    result: "StoredObject | None"  # None for a deleted key's idempotent delete
    op: str


@dataclass
class _MemoryTransactionState:
    """Per-transaction staged state. Lives on the child backend only; the
    parent never sees it until commit."""

    base_revision: int
    commit_revision: "int | None" = None  # base+1, set on first mutation
    staged_history: "dict[str, list[_Record]]" = field(default_factory=dict)
    staged_idempotency: "dict[str, _IdempotencyEntry]" = field(default_factory=dict)
    mutated: bool = False
    closed: bool = False


def _visible_history(
    parent_history: "list[_Record] | None",
    staged_history: "list[_Record] | None",
) -> "tuple[_Record, ...]":
    """Unified view: parent committed records + staged records, in order."""
    return tuple(parent_history or ()) + tuple(staged_history or ())


def _live_from_history(history: "tuple[_Record, ...]") -> "_Record | None":
    if not history:
        return None
    last = history[-1]
    return None if last.tombstone else last


def _next_version_from_history(history: "tuple[_Record, ...]") -> int:
    return (history[-1].info.version + 1) if history else 1


class MemoryObjectBackend:
    """In-memory reference backend. All writes are serialized by an asyncio
    lock so the checked operations' precondition + idempotency + mutate are
    atomic within the event loop (the TOCTOU guarantee the Protocol requires).

    The parent backend has NO ambient transaction state. ``transaction()``
    yields a ``_MemoryTransactionBackend`` child that stages writes and reads
    through a unified (parent + staged) view."""

    backend_id: str = "primary"

    def __init__(self) -> None:
        self._records: "dict[str, list[_Record]]" = {}
        self._idempotency: "dict[str, _IdempotencyEntry]" = {}
        self._revision: int = 0
        self._lock = _ReentrantAsyncLock()

    # --- helpers (parent-level, no tx) --------------------------------------

    def _live(self, key_value: str) -> "_Record | None":
        history = self._records.get(key_value)
        if not history:
            return None
        last = history[-1]
        return None if last.tombstone else last

    def _next_version(self, key_value: str) -> int:
        history = self._records.get(key_value)
        return (history[-1].info.version + 1) if history else 1

    def _check_preconditions(
        self, key_value: str, options: WriteOptions, live: "_Record | None"
    ) -> None:
        if options.if_none_match:
            if live is not None:
                raise StoragePreconditionFailedError(
                    f"if_none_match failed: {key_value!r} already exists"
                )
        if options.if_match is not None:
            if live is None or live.info.etag != options.if_match:
                raise StoragePreconditionFailedError(
                    f"if_match failed: {key_value!r} etag mismatch"
                )

    def _replay_or_conflict(
        self, request_hash: str, options: WriteOptions
    ) -> "_IdempotencyEntry | None":
        key = options.idempotency_key
        if key is None:
            return None
        existing = self._idempotency.get(key)
        if existing is None:
            return None
        if existing.request_hash != request_hash:
            raise StorageIdempotencyConflictError(
                f"idempotency key {key!r} replayed with a different request"
            )
        return existing

    # --- ObjectReaderBackend -------------------------------------------------

    async def raw_get(self, key: StorageKey, *, include_content: bool = True):
        async with self._lock:
            history = self._records.get(key.value)
            if not history:
                return Missing
            last = history[-1]
            if last.tombstone:
                return Masked(
                    key=key,
                    version=last.info.version,
                    commit_revision=last.info.commit_revision,
                )
            return Found(
                StoredObject(
                    info=last.info,
                    content=last.content if include_content else b"",
                )
            )

    async def raw_stat(self, key: StorageKey) -> "ObjectInfo | None":
        async with self._lock:
            rec = self._live(key.value)
            return None if rec is None else rec.info

    async def raw_list(
        self,
        prefix: StorageKey,
        *,
        depth: "Depth",
        limit: int,
        cursor: "str | None",
    ) -> ObjectPage:
        async with self._lock:
            items: "list[ObjectInfo]" = []
            for key_value in sorted(self._records):
                if not _under(prefix, key_value):
                    continue
                rec = self._live(key_value)
                if rec is None:
                    continue
                if not _matches_depth(prefix.value, key_value, depth):
                    continue
                items.append(rec.info)
            start = 0 if cursor is None else int(cursor)
            page = items[start : start + limit]
            next_start = start + len(page)
            next_cursor = None if next_start >= len(items) else str(next_start)
            return ObjectPage(items=tuple(page), next_cursor=next_cursor)

    async def revision(self) -> str:
        return str(self._revision)

    # --- ObjectWriterBackend -------------------------------------------------

    async def raw_put_checked(
        self,
        key: StorageKey,
        content: bytes,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> StoredObject:
        async with self._lock:
            replay = self._replay_or_conflict(request_hash, options)
            if replay is not None:
                if replay.result is None:
                    raise StorageObjectNotFoundError(key.value)
                return replay.result
            live = self._live(key.value)
            self._check_preconditions(key.value, options, live)
            new_revision = self._revision + 1
            version = self._next_version(key.value)
            info = ObjectInfo(
                key=key,
                etag=_etag(content),
                version=version,
                commit_revision=new_revision,
                content_type=options.content_type,
                size=len(content),
                modified_at=datetime.now(timezone.utc),
                metadata=MappingProxyType(dict(options.metadata or {})),
            )
            record = _Record(info=info, content=content)
            self._records.setdefault(key.value, []).append(record)
            self._revision = new_revision
            result = StoredObject(info=info, content=content)
            if options.idempotency_key is not None:
                self._idempotency[options.idempotency_key] = _IdempotencyEntry(
                    request_hash=request_hash, result=result, op="put"
                )
            return result

    async def raw_delete_checked(
        self,
        key: StorageKey,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> None:
        async with self._lock:
            replay = self._replay_or_conflict(request_hash, options)
            if replay is not None:
                return None
            live = self._live(key.value)
            self._check_preconditions(key.value, options, live)
            if live is None:
                return None
            new_revision = self._revision + 1
            tombstone = _Record(
                info=ObjectInfo(
                    key=key,
                    etag="",
                    version=self._next_version(key.value),
                    commit_revision=new_revision,
                    content_type=None,
                    size=0,
                    modified_at=datetime.now(timezone.utc),
                    metadata=MappingProxyType({}),
                ),
                content=None,
                tombstone=True,
            )
            self._records.setdefault(key.value, []).append(tombstone)
            self._revision = new_revision
            if options.idempotency_key is not None:
                self._idempotency[options.idempotency_key] = _IdempotencyEntry(
                    request_hash=request_hash, result=None, op="delete"
                )

    async def raw_move_checked(
        self,
        source: StorageKey,
        target: StorageKey,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> StoredObject:
        async with self._lock:
            replay = self._replay_or_conflict(request_hash, options)
            if replay is not None:
                if replay.result is None:
                    raise StorageObjectNotFoundError(source.value)
                return replay.result
            src = self._live(source.value)
            if src is None:
                raise StorageObjectNotFoundError(source.value)
            target_live = self._live(target.value)
            self._check_preconditions(target.value, options, target_live)
            content = src.content or b""
            new_revision = self._revision + 1
            # Target: new live record.
            target_info = ObjectInfo(
                key=target,
                etag=_etag(content),
                version=self._next_version(target.value),
                commit_revision=new_revision,
                content_type=src.info.content_type,
                size=len(content),
                modified_at=datetime.now(timezone.utc),
                metadata=MappingProxyType(dict(src.info.metadata)),
            )
            self._records.setdefault(target.value, []).append(
                _Record(info=target_info, content=content)
            )
            # Source: tombstone in the SAME revision.
            self._records.setdefault(source.value, []).append(
                _Record(
                    info=ObjectInfo(
                        key=source,
                        etag="",
                        version=self._next_version(source.value),
                        commit_revision=new_revision,
                        content_type=None,
                        size=0,
                        modified_at=datetime.now(timezone.utc),
                        metadata=MappingProxyType({}),
                    ),
                    content=None,
                    tombstone=True,
                )
            )
            self._revision = new_revision
            result = StoredObject(info=target_info, content=content)
            if options.idempotency_key is not None:
                self._idempotency[options.idempotency_key] = _IdempotencyEntry(
                    request_hash=request_hash, result=result, op="move"
                )
            return result

    # --- TransactionalObjectBackend -----------------------------------------

    @contextlib.asynccontextmanager
    async def transaction(self) -> "AsyncIterator[_MemoryTransactionBackend]":
        async with self._lock:
            state = _MemoryTransactionState(base_revision=self._revision)
            child = _MemoryTransactionBackend(self, state)
            try:
                yield child
            except BaseException:
                state.closed = True
                raise
            else:
                self._commit_transaction(state)
                state.closed = True

    def _commit_transaction(self, state: _MemoryTransactionState) -> None:
        """Apply staged records + idempotency + revision atomically."""
        for key_value, records in state.staged_history.items():
            self._records.setdefault(key_value, []).extend(records)
        for key, entry in state.staged_idempotency.items():
            self._idempotency[key] = entry
        if state.mutated:
            self._revision = state.commit_revision  # type: ignore[assignment]

    # --- VersionedObjectBackend ---------------------------------------------

    async def raw_get_version(self, key: StorageKey, version: int) -> "StoredObject | None":
        async with self._lock:
            history = self._records.get(key.value, [])
            for rec in history:
                if rec.info.version == version:
                    if rec.tombstone:
                        return None
                    return StoredObject(info=rec.info, content=rec.content or b"")
            return None

    async def raw_get_at_revision(self, key: StorageKey, revision: int) -> "StoredObject | None":
        async with self._lock:
            history = self._records.get(key.value, [])
            live: "_Record | None" = None
            for rec in history:
                if rec.info.commit_revision is not None and rec.info.commit_revision <= revision:
                    live = None if rec.tombstone else rec
                else:
                    break
            return None if live is None else StoredObject(info=live.info, content=live.content or b"")

    async def raw_list_versions(
        self,
        key: StorageKey,
        *,
        limit: int = 100,
        cursor: "str | None" = None,
    ) -> ObjectVersionPage:
        async with self._lock:
            history = self._records.get(key.value, [])
            items = [rec.info for rec in history]
            start = 0 if cursor is None else int(cursor)
            page = items[start : start + limit]
            next_start = start + len(page)
            next_cursor = None if next_start >= len(items) else str(next_start)
            return ObjectVersionPage(items=tuple(page), next_cursor=next_cursor)

    async def raw_list_at_revision(self, prefix: StorageKey, revision: int) -> "tuple[ObjectInfo, ...]":
        async with self._lock:
            out: "list[ObjectInfo]" = []
            for key_value in sorted(self._records):
                if not _under(prefix, key_value):
                    continue
                history = self._records[key_value]
                live: "_Record | None" = None
                for rec in history:
                    if rec.info.commit_revision is not None and rec.info.commit_revision <= revision:
                        live = None if rec.tombstone else rec
                    else:
                        break
                if live is not None:
                    out.append(live.info)
            return tuple(out)


class _MemoryTransactionBackend:
    """Transaction-bound child backend. Stages writes in its own state;
    reads see a unified (parent committed + staged) view. After the context
    exits, any call raises ``StorageTransactionClosedError``."""

    backend_id: str = "primary"

    def __init__(self, parent: MemoryObjectBackend, state: _MemoryTransactionState) -> None:
        self._parent = parent
        self._state = state

    # --- lifecycle guard ----------------------------------------------------

    def _check_open(self) -> None:
        if self._state.closed:
            raise StorageTransactionClosedError(
                "transaction-bound backend used after context exit"
            )

    # --- unified-view helpers -----------------------------------------------

    def _visible_history(self, key_value: str) -> "tuple[_Record, ...]":
        parent_history = self._parent._records.get(key_value)
        staged = self._state.staged_history.get(key_value)
        return _visible_history(parent_history, staged)

    def _visible_live(self, key_value: str) -> "_Record | None":
        return _live_from_history(self._visible_history(key_value))

    def _visible_next_version(self, key_value: str) -> int:
        return _next_version_from_history(self._visible_history(key_value))

    def _ensure_commit_revision(self) -> int:
        if self._state.commit_revision is None:
            self._state.commit_revision = self._state.base_revision + 1
            self._state.mutated = True
        return self._state.commit_revision

    def _replay_or_conflict_tx(
        self, request_hash: str, options: WriteOptions
    ) -> "_IdempotencyEntry | None":
        """Check staged idempotency FIRST, then parent committed."""
        key = options.idempotency_key
        if key is None:
            return None
        # 1. staged
        staged = self._state.staged_idempotency.get(key)
        if staged is not None:
            if staged.request_hash != request_hash:
                raise StorageIdempotencyConflictError(
                    f"idempotency key {key!r} replayed with a different request"
                )
            return staged
        # 2. parent committed
        return self._parent._replay_or_conflict(request_hash, options)

    # --- ObjectReaderBackend -------------------------------------------------

    async def raw_get(self, key: StorageKey, *, include_content: bool = True):
        self._check_open()
        async with self._parent._lock:
            history = self._visible_history(key.value)
            if not history:
                return Missing
            last = history[-1]
            if last.tombstone:
                return Masked(
                    key=key,
                    version=last.info.version,
                    commit_revision=last.info.commit_revision,
                )
            return Found(
                StoredObject(
                    info=last.info,
                    content=last.content if include_content else b"",
                )
            )

    async def raw_stat(self, key: StorageKey) -> "ObjectInfo | None":
        self._check_open()
        async with self._parent._lock:
            rec = self._visible_live(key.value)
            return None if rec is None else rec.info

    async def raw_list(
        self,
        prefix: StorageKey,
        *,
        depth: "Depth",
        limit: int,
        cursor: "str | None",
    ) -> ObjectPage:
        self._check_open()
        async with self._parent._lock:
            # Merge parent + staged keys for a unified listing.
            all_keys = sorted(
                set(self._parent._records.keys()) | set(self._state.staged_history.keys())
            )
            items: "list[ObjectInfo]" = []
            for key_value in all_keys:
                if not _under(prefix, key_value):
                    continue
                rec = self._visible_live(key_value)
                if rec is None:
                    continue
                if not _matches_depth(prefix.value, key_value, depth):
                    continue
                items.append(rec.info)
            start = 0 if cursor is None else int(cursor)
            page = items[start : start + limit]
            next_start = start + len(page)
            next_cursor = None if next_start >= len(items) else str(next_start)
            return ObjectPage(items=tuple(page), next_cursor=next_cursor)

    async def revision(self) -> str:
        self._check_open()
        # Inside a tx: if mutated, report the commit revision; otherwise base.
        if self._state.mutated and self._state.commit_revision is not None:
            return str(self._state.commit_revision)
        return str(self._state.base_revision)

    # --- ObjectWriterBackend -------------------------------------------------

    async def raw_put_checked(
        self,
        key: StorageKey,
        content: bytes,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> StoredObject:
        self._check_open()
        async with self._parent._lock:
            replay = self._replay_or_conflict_tx(request_hash, options)
            if replay is not None:
                if replay.result is None:
                    raise StorageObjectNotFoundError(key.value)
                return replay.result
            live = self._visible_live(key.value)
            self._parent._check_preconditions(key.value, options, live)
            commit_revision = self._ensure_commit_revision()
            version = self._visible_next_version(key.value)
            info = ObjectInfo(
                key=key,
                etag=_etag(content),
                version=version,
                commit_revision=commit_revision,
                content_type=options.content_type,
                size=len(content),
                modified_at=datetime.now(timezone.utc),
                metadata=MappingProxyType(dict(options.metadata or {})),
            )
            record = _Record(info=info, content=content)
            self._state.staged_history.setdefault(key.value, []).append(record)
            result = StoredObject(info=info, content=content)
            if options.idempotency_key is not None:
                self._state.staged_idempotency[options.idempotency_key] = _IdempotencyEntry(
                    request_hash=request_hash, result=result, op="put"
                )
            return result

    async def raw_delete_checked(
        self,
        key: StorageKey,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> None:
        self._check_open()
        async with self._parent._lock:
            replay = self._replay_or_conflict_tx(request_hash, options)
            if replay is not None:
                return None
            live = self._visible_live(key.value)
            self._parent._check_preconditions(key.value, options, live)
            if live is None:
                return None
            commit_revision = self._ensure_commit_revision()
            tombstone = _Record(
                info=ObjectInfo(
                    key=key,
                    etag="",
                    version=self._visible_next_version(key.value),
                    commit_revision=commit_revision,
                    content_type=None,
                    size=0,
                    modified_at=datetime.now(timezone.utc),
                    metadata=MappingProxyType({}),
                ),
                content=None,
                tombstone=True,
            )
            self._state.staged_history.setdefault(key.value, []).append(tombstone)
            if options.idempotency_key is not None:
                self._state.staged_idempotency[options.idempotency_key] = _IdempotencyEntry(
                    request_hash=request_hash, result=None, op="delete"
                )

    async def raw_move_checked(
        self,
        source: StorageKey,
        target: StorageKey,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> StoredObject:
        self._check_open()
        async with self._parent._lock:
            replay = self._replay_or_conflict_tx(request_hash, options)
            if replay is not None:
                if replay.result is None:
                    raise StorageObjectNotFoundError(source.value)
                return replay.result
            src = self._visible_live(source.value)
            if src is None:
                raise StorageObjectNotFoundError(source.value)
            target_live = self._visible_live(target.value)
            self._parent._check_preconditions(target.value, options, target_live)
            commit_revision = self._ensure_commit_revision()
            content = src.content or b""
            target_info = ObjectInfo(
                key=target,
                etag=_etag(content),
                version=self._visible_next_version(target.value),
                commit_revision=commit_revision,
                content_type=src.info.content_type,
                size=len(content),
                modified_at=datetime.now(timezone.utc),
                metadata=MappingProxyType(dict(src.info.metadata)),
            )
            self._state.staged_history.setdefault(target.value, []).append(
                _Record(info=target_info, content=content)
            )
            self._state.staged_history.setdefault(source.value, []).append(
                _Record(
                    info=ObjectInfo(
                        key=source,
                        etag="",
                        version=self._visible_next_version(source.value),
                        commit_revision=commit_revision,
                        content_type=None,
                        size=0,
                        modified_at=datetime.now(timezone.utc),
                        metadata=MappingProxyType({}),
                    ),
                    content=None,
                    tombstone=True,
                )
            )
            result = StoredObject(info=target_info, content=content)
            if options.idempotency_key is not None:
                self._state.staged_idempotency[options.idempotency_key] = _IdempotencyEntry(
                    request_hash=request_hash, result=result, op="move"
                )
            return result


class MemoryObjectStore:
    """Convenience: an ObjectStore pre-wired to a fresh MemoryObjectBackend
    (the common test + reference-impl entry point)."""

    def __init__(self) -> None:
        from ...object.store import ObjectStore

        self._backend = MemoryObjectBackend()
        self._store = ObjectStore(primary=self._backend)

    @property
    def backend(self) -> MemoryObjectBackend:
        return self._backend

    @contextlib.asynccontextmanager
    async def transaction(self):
        async with self._store.transaction() as tx:
            yield tx

    # Delegate the ObjectStore surface to the wrapped store.
    async def get(self, key: StorageKey) -> "StoredObject | None":
        return await self._store.get(key)

    async def stat(self, key: StorageKey) -> "ObjectInfo | None":
        return await self._store.stat(key)

    async def list(self, prefix: StorageKey, **kwargs) -> ObjectPage:
        return await self._store.list(prefix, **kwargs)

    async def revision(self) -> str:
        return await self._store.revision()

    async def put(self, key: StorageKey, content: bytes, **kwargs) -> StoredObject:
        return await self._store.put(key, content, **kwargs)

    async def delete(self, key: StorageKey, **kwargs) -> None:
        await self._store.delete(key, **kwargs)

    async def move(self, source: StorageKey, target: StorageKey, **kwargs) -> StoredObject:
        return await self._store.move(source, target, **kwargs)

    # History surface (delegates to the backend's VersionedObjectBackend).
    async def get_version(self, key: StorageKey, version: int) -> "StoredObject | None":
        return await self._backend.raw_get_version(key, version)

    async def list_versions(self, key: StorageKey, *, limit: int = 100, cursor: "str | None" = None) -> ObjectVersionPage:
        return await self._backend.raw_list_versions(key, limit=limit, cursor=cursor)


def _under(prefix: StorageKey, key_value: str) -> bool:
    if prefix.is_root:
        return True
    return key_value == prefix.value or key_value.startswith(prefix.value + "/")


def _matches_depth(prefix_value: str, key_value: str, depth: "Depth") -> bool:
    if depth is Depth.ZERO:
        return True
    if depth is Depth.INFINITY:
        return True
    # Depth.ONE: exactly one path component below the prefix.
    remainder = key_value[len(prefix_value) :].lstrip("/") if prefix_value != "/" else key_value.lstrip("/")
    return "/" not in remainder
