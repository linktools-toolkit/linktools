#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime object-store routing and lifecycle ownership."""

import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
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
    TransientObjectStore,
    read_object,
)
from ._plan import (
    RuntimeDomain,
    RuntimeRetentionMode,
    RuntimeStoragePlan,
    runtime_domain_uses_object_store,
)

_logger = environ.get_logger("ai.runtime.state.materializer")


class _RuntimeObjectRouter:
    def __init__(
        self,
        stores: Mapping[RuntimeDomain, ObjectStore],
        *,
        close_guard_stores: Sequence[ObjectStore],
        cache_backend: CacheBackend | None = None,
        cache_domains: Sequence[RuntimeDomain] = (),
        namespace: str = "",
        tenant_id: str = "",
    ) -> None:
        self._stores = dict(stores)
        self._close_guard_stores = tuple(close_guard_stores)
        self._cache_backend = cache_backend
        self._cache_domains = frozenset(cache_domains)
        self._cache_scope = (namespace, tenant_id)
        self._closing = False
        self._pending_reads: set[asyncio.Task[bytes]] = set()
        self._inflight: dict[tuple[RuntimeDomain, ObjectRef], asyncio.Task[bytes]] = {}

    async def read_object(self, domain: RuntimeDomain, reference: ObjectRef) -> bytes:
        """Read immutable bytes without making the cache a retention authority."""
        if self._closing:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        store = self.resolve_object(domain, reference)
        task = asyncio.create_task(self._read_accepted(domain, store, reference))
        self._pending_reads.add(task)
        task.add_done_callback(self._read_finished)
        if self._cache_backend is None or domain not in self._cache_domains:
            # Unshared reads retain request-owned cancellation and stream close.
            return await task
        return await asyncio.shield(task)

    async def _read_accepted(
        self, domain: RuntimeDomain, store: ObjectStore, reference: ObjectRef
    ) -> bytes:
        if self._cache_backend is None or domain not in self._cache_domains:
            return await self._read_origin(store, reference)
        identity = (domain, reference)
        task = self._inflight.get(identity)
        if task is None or task.done():
            task = asyncio.create_task(self._read_cached(domain, store, reference))
            self._inflight[identity] = task
            task.add_done_callback(
                lambda finished: self._flight_finished(identity, finished)
            )
        else:
            # A load may have opened the source before this borrower arrived.
            # Membership must therefore be established for each joining read.
            await self._validate_source(store, reference)
        return await asyncio.shield(task)

    async def _read_cached(
        self, domain: RuntimeDomain, store: ObjectStore, reference: ObjectRef
    ) -> bytes:
        cache = self._cache_backend
        assert cache is not None
        key = self._cache_key(domain, reference)
        try:
            cached = await cache.get(key)
        except Exception:  # noqa: BLE001
            cached = None
        if cached is not None:
            if (
                isinstance(cached, bytes)
                and len(cached) == reference.size
                and hashlib.sha256(cached).hexdigest() == reference.digest
            ):
                await self._validate_source(store, reference)
                return cached
            try:
                await cache.delete(key)
            except Exception:  # noqa: BLE001, S110
                pass
        content = await self._read_origin(store, reference)
        try:
            await cache.put(key, content)
        except Exception:  # noqa: BLE001, S110
            pass
        return content

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
        # Store IDs are not globally unique. Equal keys may only reuse verified
        # bytes; a hit still checks membership in this router's actual source.
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
        if self._inflight.get(identity) is task:
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

    def resolve_object(
        self, domain: RuntimeDomain, reference: ObjectRef
    ) -> ObjectStore:
        """Resolve an object by its durable Runtime domain."""
        return self.object_store(domain)

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
    cache_backend: CacheBackend | None = None,
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
        if route.retention is RuntimeRetentionMode.DURABLE and external is not None:
            values[domain] = external
        elif route.retention is RuntimeRetentionMode.VOLATILE:
            store = InMemoryObjectStore()
            values[domain] = store
            close_guard_stores.append(store)
        elif route.retention is RuntimeRetentionMode.TRANSIENT:
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
        cache_backend=cache_backend,
        cache_domains=tuple(
            domain for domain in values
            if plan.route(domain).retention is RuntimeRetentionMode.DURABLE
        ),
        namespace=namespace,
        tenant_id=tenant_id,
    )


__all__ = ["_RuntimeObjectRouter", "build_runtime_object_router"]
