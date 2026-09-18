#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime object-store routing and lifecycle ownership."""

import asyncio
from collections.abc import Mapping, Sequence

from linktools.core import environ

from ...errors import AIError, ErrorCode
from ...storage import (
    FilesystemObjectStore,
    InMemoryObjectStore,
    ObjectRef,
    ObjectStore,
    SqlObjectStore,
    SqlStorageContext,
    TransientObjectStore,
)
from ._plan import (
    RuntimeDomain,
    RuntimeRetentionMode,
    RuntimeStatePlan,
    runtime_domain_uses_object_store,
)

_logger = environ.get_logger("ai.runtime.state.materializer")


class _RuntimeObjectRouter:
    def __init__(
        self,
        stores: Mapping[RuntimeDomain, ObjectStore],
        *,
        close_guard_stores: Sequence[ObjectStore],
    ) -> None:
        self._stores = dict(stores)
        self._close_guard_stores = tuple(close_guard_stores)

    def object_store(self, domain: RuntimeDomain) -> ObjectStore:
        try:
            return self._stores[domain]
        except KeyError as error:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY) from error

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
    plan: RuntimeStatePlan,
    external: ObjectStore | None,
    stores: Mapping[RuntimeDomain, object],
    contexts: Mapping[RuntimeDomain, SqlStorageContext],
) -> _RuntimeObjectRouter:
    values: dict[RuntimeDomain, ObjectStore] = {}
    close_guard_stores: list[ObjectStore] = []
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
            store = FilesystemObjectStore(route.path / "objects")
            values[domain] = store
            close_guard_stores.append(store)
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
    )


__all__ = ["_RuntimeObjectRouter", "build_runtime_object_router"]
