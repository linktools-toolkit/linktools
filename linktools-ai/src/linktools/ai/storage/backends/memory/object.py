#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The in-memory object backend -- the storage kernel's reference implementation.

Implements every capability (revision, history, transaction, idempotency, CAS,
move) so the storage.object contract suite has one backend that exercises the
FULL behavior, parameterized over the same contract the production backends
must satisfy. Not durable -- process-local state only. Production deployments
use the filesystem or SQLAlchemy backend; this one is for tests + as the
behavioral reference other backends are checked against."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType

from ...object.errors import (
    StorageIdempotencyConflictError,
    StorageObjectNotFoundError,
    StoragePreconditionFailedError,
)
from ...object.models import (
    Depth,
    Found,
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
class _PendingTx:
    """Staged mutations inside a transaction, applied atomically on commit."""

    puts: "dict[str, _Record]" = field(default_factory=dict)
    deletes: "dict[str, _Record]" = field(default_factory=dict)
    bumps_revision: bool = False


class MemoryObjectBackend:
    """In-memory reference backend. All writes are serialized by an asyncio
    lock so the checked operations' precondition + idempotency + mutate are
    atomic within the event loop (the TOCTOU guarantee the Protocol requires)."""

    backend_id: str = "primary"

    def __init__(self) -> None:
        self._records: "dict[str, list[_Record]]" = {}
        self._idempotency: "dict[str, _IdempotencyEntry]" = {}
        self._revision: int = 0
        self._lock = _ReentrantAsyncLock()
        self._tx: "_PendingTx | None" = None

    # --- helpers -------------------------------------------------------------

    def _live(self, key_value: str) -> "_Record | None":
        history = self._records.get(key_value)
        if not history:
            return None
        last = history[-1]
        return None if last.tombstone else last

    def _next_version(self, key_value: str) -> int:
        history = self._records.get(key_value)
        return (history[-1].info.version + 1) if history else 1

    def _bump_revision(self) -> None:
        # Inside a transaction the bump happens once at commit (a single bump
        # for the whole tx); outside, bump immediately.
        if self._tx is not None:
            self._tx.bumps_revision = True
        else:
            self._revision += 1

    def _apply_record(self, key_value: str, record: _Record) -> None:
        target = self._tx.puts if self._tx is not None else self._records
        if self._tx is not None and key_value in self._tx.deletes:
            del self._tx.deletes[key_value]
        target.setdefault(key_value, []).append(record)

    def _apply_tombstone(self, key_value: str, record: _Record) -> None:
        if self._tx is not None:
            # A delete inside a tx stages the tombstone; drop any staged put.
            self._tx.puts.pop(key_value, None)
            self._tx.deletes[key_value] = record
        else:
            self._records.setdefault(key_value, []).append(record)

    def _resolve(self, key_value: str) -> "_Record | None":
        """Current live record, accounting for staged transaction state."""
        if self._tx is not None:
            if key_value in self._tx.deletes:
                return None
            if key_value in self._tx.puts:
                return self._tx.puts[key_value][-1]
        return self._live(key_value)

    def _check_preconditions(self, key_value: str, options: WriteOptions) -> None:
        live = self._resolve(key_value)
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
        """Return a cached entry to replay, raise on conflict, or None to
        perform the op fresh."""
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
                # Idempotent replay: return the cached result, no version bump.
                if replay.result is None:
                    raise StorageObjectNotFoundError(key.value)
                return replay.result
            self._check_preconditions(key.value, options)
            version = self._next_version(key.value)
            info = ObjectInfo(
                key=key,
                etag=_etag(content),
                version=version,
                commit_revision=self._revision_if_txn(),
                content_type=options.content_type,
                size=len(content),
                modified_at=datetime.now(timezone.utc),
                metadata=MappingProxyType(dict(options.metadata or {})),
            )
            record = _Record(info=info, content=content)
            self._apply_record(key.value, record)
            self._bump_revision()
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
            self._check_preconditions(key.value, options)
            if self._resolve(key.value) is None:
                # Deleting a missing key is a no-op (no tombstone, no bump).
                return None
            tombstone = _Record(
                info=ObjectInfo(
                    key=key,
                    etag="",
                    version=self._next_version(key.value),
                    commit_revision=self._revision_if_txn(),
                    content_type=None,
                    size=0,
                    modified_at=datetime.now(timezone.utc),
                    metadata=MappingProxyType({}),
                ),
                content=None,
                tombstone=True,
            )
            self._apply_tombstone(key.value, tombstone)
            self._bump_revision()
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
            src = self._resolve(source.value)
            if src is None:
                raise StorageObjectNotFoundError(source.value)
            self._check_preconditions(target.value, options)
            content = src.content or b""
            version = self._next_version(target.value)
            info = ObjectInfo(
                key=target,
                etag=_etag(content),
                version=version,
                commit_revision=self._revision_if_txn(),
                content_type=src.info.content_type,
                size=len(content),
                modified_at=datetime.now(timezone.utc),
                metadata=MappingProxyType(dict(src.info.metadata)),
            )
            # ONE revision bump for the whole move (tombstone + create).
            self._apply_record(target.value, _Record(info=info, content=content))
            self._apply_tombstone(
                source.value,
                _Record(
                    info=ObjectInfo(
                        key=source,
                        etag="",
                        version=self._next_version(source.value),
                        commit_revision=self._revision_if_txn(),
                        content_type=None,
                        size=0,
                        modified_at=datetime.now(timezone.utc),
                        metadata=MappingProxyType({}),
                    ),
                    content=None,
                    tombstone=True,
                ),
            )
            self._bump_revision()
            result = StoredObject(info=info, content=content)
            if options.idempotency_key is not None:
                self._idempotency[options.idempotency_key] = _IdempotencyEntry(
                    request_hash=request_hash, result=result, op="move"
                )
            return result

    def _revision_if_txn(self) -> "int | None":
        # commit_revision is the namespace watermark; inside a tx it is the
        # post-commit value, outside it is the current revision.
        return self._revision

    # --- TransactionalObjectBackend -----------------------------------------

    @contextlib.asynccontextmanager
    async def transaction(self):
        async with self._lock:
            if self._tx is not None:
                raise RuntimeError("nested transactions are not supported")
            self._tx = _PendingTx()
            try:
                yield self
            except BaseException:
                # Rollback: drop the staged tx state entirely -- the canonical
                # records + revision are untouched.
                self._tx = None
                raise
            else:
                tx = self._tx
                self._tx = None
                # Commit: flush staged puts + tombstones to the canonical history.
                for key_value, records in tx.puts.items():
                    self._records.setdefault(key_value, []).extend(records)
                for key_value, record in tx.deletes.items():
                    self._records.setdefault(key_value, []).append(record)
                if tx.bumps_revision:
                    self._revision += 1

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


class MemoryObjectStore:
    """Convenience: an ObjectStore pre-wired to a fresh MemoryObjectBackend
    (the common test + reference-impl entry point)."""

    def __init__(self) -> None:
        from ...object.store import ObjectStore

        self._backend = MemoryObjectBackend()
        self._store = ObjectStore(primary=self._backend)

    @property
    def backend(self) -> MemoryObjectBackend:
        """The raw backend (for composing into an OverlayObjectStore)."""
        return self._backend

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

    def transaction(self):
        return self._backend.transaction()

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
        return key_value == prefix_value
    rel = 0 if key_value == prefix_value else key_value[len(prefix_value) :].count("/") if prefix_value == "/" else key_value[len(prefix_value) + 1 :].count("/") + 1
    if depth is Depth.ONE:
        return rel <= 1
    return True
