#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime object-store routing and lifecycle ownership."""

from ...storage import StorageRetentionMode

import asyncio
import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic
from pathlib import Path

from linktools.core import environ

from ...errors import AIError, ErrorCode
from ...storage import (
    CacheBackend,
    FilesystemObjectStore,
    InMemoryObjectStore,
    ObjectRef,
    ObjectStore,
    SqlObjectStore,
    SqlStorageContext,
    StorageReadCache,
    TransientObjectStore,
    read_object,
)
from ._plan import RuntimeDomain, RuntimeStoragePlan, runtime_domain_uses_object_store

_logger = environ.get_logger("ai.runtime.state.materializer")
_OBJECT_MAX_AGE_SECONDS = 3.0
_source_reads: ContextVar[bool] = ContextVar("runtime_object_source_reads", default=False)


@contextmanager
def runtime_object_source_reads() -> Iterator[None]:
    """Require current source membership without holding a storage transaction."""
    token = _source_reads.set(True)
    try:
        yield
    finally:
        _source_reads.reset(token)


class _RuntimeObjectRouter:
    def __init__(
        self,
        stores: Mapping[RuntimeDomain, ObjectStore],
        *,
        close_guard_stores: Sequence[ObjectStore],
        cache: CacheBackend | None = None,
        domain_caches: Mapping[RuntimeDomain, CacheBackend | None] | None = None,
        cache_domains: Sequence[RuntimeDomain] = (),
        namespace: str = "",
        tenant_id: str = "",
    ) -> None:
        self._stores = dict(stores)
        self._close_guard_stores = tuple(close_guard_stores)
        selected = dict(domain_caches or {})
        self._caches = {
            domain: candidate if isinstance(candidate := selected.get(domain, cache), CacheBackend) else None
            for domain in stores
        }
        self._hot = {
            domain: StorageReadCache(candidate, max_age_seconds=_OBJECT_MAX_AGE_SECONDS)
            for domain, candidate in self._caches.items()
        }
        self._cache_domains = frozenset(cache_domains)
        self._cache_scope = (namespace, tenant_id)
        self._closing = False
        self._pending_reads: set[asyncio.Task[bytes]] = set()
        self._inflight: dict[tuple[RuntimeDomain, ObjectRef], tuple[float, asyncio.Task[bytes]]] = {}

    async def read_object(self, domain: RuntimeDomain, reference: ObjectRef) -> bytes:
        """Read verified bytes with bounded source-membership staleness."""
        if self._closing:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        store = self.object_store(domain)
        task = asyncio.create_task(self._read_accepted(domain, store, reference))
        self._pending_reads.add(task)
        task.add_done_callback(self._read_finished)
        if self._caches[domain] is None or domain not in self._cache_domains:
            # Unshared reads retain request-owned cancellation and stream close.
            return await task
        return await asyncio.shield(task)

    async def _read_accepted(
        self, domain: RuntimeDomain, store: ObjectStore, reference: ObjectRef
    ) -> bytes:
        if self._caches[domain] is None or domain not in self._cache_domains:
            return await self._read_origin(store, reference)
        identity = (domain, reference)
        flight = self._inflight.get(identity)
        if (flight is None or flight[1].done()
                or monotonic() >= flight[0] + _OBJECT_MAX_AGE_SECONDS):
            task = asyncio.create_task(self._read_cached(domain, store, reference))
            self._inflight[identity] = (monotonic(), task)
            task.add_done_callback(
                lambda finished: self._flight_finished(identity, finished)
            )
        else:
            task = flight[1]
            if domain in {RuntimeDomain.RECOVERY, RuntimeDomain.CHECKPOINT} or _source_reads.get():
                await self._validate_source(store, reference)
        return await asyncio.shield(task)

    async def _read_cached(
        self, domain: RuntimeDomain, store: ObjectStore, reference: ObjectRef
    ) -> bytes:
        key = self._cache_key(domain, reference)

        loaded = False

        async def load() -> bytes:
            nonlocal loaded
            loaded = True
            return await self._read_origin(store, reference)

        content = await self._hot[domain].get(key, load)
        if (isinstance(content, bytes) and len(content) == reference.size
                and hashlib.sha256(content).hexdigest() == reference.digest):
            if not loaded and (domain in {RuntimeDomain.RECOVERY, RuntimeDomain.CHECKPOINT} or _source_reads.get()):
                await self._validate_source(store, reference)
            return content
        await self._hot[domain].invalidate(key)
        return await load()

    def _cache_key(self, domain: RuntimeDomain, reference: ObjectRef) -> str:
        identity = (
            *self._cache_scope,
            domain.value,
            reference.store_id,
            reference.key,
            reference.digest,
            reference.size,
        )
        encoded = json.dumps(identity, ensure_ascii=True, separators=(",", ":"))
        # A shared cache/Runtime namespace is bound to one logical source.
        # The hot envelope bounds membership age independently of backend TTL.
        return "runtime-object:v1:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    async def _validate_source(self, store: ObjectStore, reference: ObjectRef) -> None:
        current = await store.stat(reference.key)
        if current is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if (
            current.key != reference.key
            or current.size != reference.size
            or current.digest != reference.digest
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _read_origin(self, store: ObjectStore, reference: ObjectRef) -> bytes:
        return await read_object(
            store,
            reference.key,
            expected_digest=reference.digest,
            expected_size=reference.size,
        )

    def _read_finished(self, task: asyncio.Task[bytes]) -> None:
        self._pending_reads.discard(task)
        if not task.cancelled():
            task.exception()

    def _flight_finished(
        self, identity: tuple[RuntimeDomain, ObjectRef], task: asyncio.Task[bytes]
    ) -> None:
        flight = self._inflight.get(identity)
        if flight is not None and flight[1] is task:
            del self._inflight[identity]
        if not task.cancelled():
            task.exception()

    def begin_close(self) -> None:
        """Stop admitting new reads before draining owned work."""
        self._closing = True

    async def drain_reads(self) -> None:
        """Wait for all accepted reads, including abandoned caller waiters."""
        self.begin_close()
        while self._pending_reads:
            await asyncio.wait(tuple(self._pending_reads))
        for hot in self._hot.values():
            await hot.close()

    def object_store(self, domain: RuntimeDomain) -> ObjectStore:
        try:
            return self._stores[domain]
        except KeyError as error:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY) from error

    def local_paths(self) -> tuple[Path, ...]:
        paths: list[Path] = []
        for store in self._stores.values():
            paths.extend(store.local_paths())
        return tuple(dict.fromkeys(paths))

    def working_object_store(
        self, domain: RuntimeDomain, *, owner_scope: str
    ) -> ObjectStore:
        store = self.object_store(domain)
        if isinstance(store, TransientObjectStore):
            return store.scoped(f"runtime:{domain.value}:{owner_scope}")
        return store

    async def release_object_scope(
        self, domain: RuntimeDomain, *, owner_scope: str
    ) -> None:
        store = self.object_store(domain)
        if isinstance(store, TransientObjectStore):
            await store.release_scope(f"runtime:{domain.value}:{owner_scope}")

    async def clear_transient(self) -> None:
        seen: set[int] = set()
        for store in self._close_guard_stores:
            if isinstance(store, TransientObjectStore) and id(store) not in seen:
                store.clear()
                seen.add(id(store))

    async def preflight_close(self) -> None:
        pending: dict[int, asyncio.Task[object]] = {}
        seen: set[int] = set()
        for store in self._close_guard_stores:
            if id(store) in seen:
                continue
            seen.add(id(store))
            if not isinstance(store, (FilesystemObjectStore, SqlObjectStore)):
                continue
            for task in store.pending_background_tasks:
                pending[id(task)] = task
        if pending:
            _logger.warning(
                "runtime object preflight found pending background work: tasks=%s",
                len(pending),
            )
            raise AIError(
                ErrorCode.STORAGE_RECOVERY_REQUIRED,
                safe_details={
                    "phase": "object_preflight_close",
                    "pending_tasks": len(pending),
                },
            )


def build_runtime_object_router(
    plan: RuntimeStoragePlan,
    external: ObjectStore | None,
    stores: Mapping[RuntimeDomain, object],
    contexts: Mapping[RuntimeDomain, SqlStorageContext],
    *,
    namespace: str = "",
    tenant_id: str = "",
) -> _RuntimeObjectRouter:
    values: dict[RuntimeDomain, ObjectStore] = {}
    close_guard_stores: list[ObjectStore] = []
    filesystem_objects: dict[Path, FilesystemObjectStore] = {}
    sql_objects: dict[int, SqlObjectStore] = {}
    for domain in RuntimeDomain:
        if not runtime_domain_uses_object_store(domain):
            continue
        route = plan.route(domain)
        if route.retention is StorageRetentionMode.DURABLE and external is not None:
            values[domain] = external
        elif route.retention is StorageRetentionMode.VOLATILE:
            store = InMemoryObjectStore()
            values[domain] = store
            close_guard_stores.append(store)
        elif route.retention is StorageRetentionMode.TRANSIENT:
            store = TransientObjectStore()
            values[domain] = store
            close_guard_stores.append(store)
        elif route.kind == "filesystem" and route.path is not None:
            object_root = (route.path / "objects").resolve()
            store = filesystem_objects.get(object_root)
            if store is None:
                store = FilesystemObjectStore(object_root)
                filesystem_objects[object_root] = store
                close_guard_stores.append(store)
            values[domain] = store
        elif route.kind in {"sqlite", "sql"} and domain in contexts:
            context = contexts[domain]
            context_key = id(context)
            store = sql_objects.get(context_key)
            if store is None:
                store = SqlObjectStore.from_context(context)
                sql_objects[context_key] = store
            values[domain] = store
            close_guard_stores.append(store)
        else:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
    return _RuntimeObjectRouter(
        values,
        close_guard_stores=close_guard_stores,
        domain_caches={domain: plan.route(domain).cache for domain in values},
        cache_domains=tuple(
            domain for domain in values
            if plan.route(domain).retention is StorageRetentionMode.DURABLE
        ),
        namespace=namespace,
        tenant_id=tenant_id,
    )


__all__ = ["_RuntimeObjectRouter", "build_runtime_object_router", "runtime_object_source_reads"]
