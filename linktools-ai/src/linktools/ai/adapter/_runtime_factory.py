#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapter-owned materialization of declarative Runtime storage targets."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.core import environ
from ..errors import AIError, ErrorCode
from ..runtime import RuntimeDomain, RuntimeRetention, RuntimeStorage, RuntimeStores
from ..storage import ObjectRef, ObjectStore, SqlStorageContext, TransientObjectStore, create_sql_storage_context, provision_sql, validate_sql
from ._persistence import (
    build_filesystem_runtime,
    build_in_memory_runtime,
)
from ._retention import _RuntimeRetention
from ._schema import build_runtime_sql_metadata
from ._sql_runtime import build_sql_runtime
from ._step import (
    FilesystemStepArchive,
    InMemoryStepArchive,
    RuntimeStepPersistence,
    SqlStepArchive,
    StagingStepStore,
)

_logger = environ.get_logger("ai.adapter.runtime_factory")

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


@dataclass(frozen=True, slots=True)
class RuntimePersistence:
    domain: RuntimeStores
    steps: RuntimeStepPersistence
    retention: _RuntimeRetention
    close_callback: Callable[[], Awaitable[None]]

    async def close(self) -> None:
        callback = self.close_callback
        await callback()

@asynccontextmanager
async def open_runtime_persistence(
    storage: RuntimeStorage,
    *,
    namespace: str,
    tenant_id: str,
) -> AsyncIterator[RuntimePersistence]:
    """Materialize one fixed namespace/tenant Runtime persistence boundary."""

    durable = frozenset(
        domain
        for domain in RuntimeDomain
        if storage.plan.route(domain).retention is RuntimeRetention.DURABLE
    )
    target_kind = storage.target_kind
    owned_context = False
    sql_context: "SqlStorageContext | None" = None
    if target_kind in {"sqlite", "sql"}:
        from sqlalchemy.ext.asyncio import create_async_engine

        if target_kind == "sqlite":
            path = storage.target_path
            if path is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
            sql_context = create_sql_storage_context(engine, owns_engine=True)
        else:
            engine = storage.target_engine
            if engine is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            sql_context = create_sql_storage_context(engine)
        owned_context = True
    runtime = None
    steps: RuntimeStepPersistence | None = None
    try:
        if target_kind in {"sqlite", "sql"}:
            if sql_context is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            metadata = build_runtime_sql_metadata(storage.plan)
            if target_kind == "sqlite":
                await provision_sql(sql_context.engine, metadata)
            else:
                await validate_sql(sql_context.engine, metadata)
            await sql_context.initialize()
        if target_kind == "memory":
            runtime = build_in_memory_runtime(namespace=namespace)
        elif target_kind == "filesystem":
            path = storage.target_path
            if path is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            runtime = build_filesystem_runtime(str(path), namespace=namespace, persist=durable)
            from ._persistence import RuntimeObjectRouter

            runtime.persistence = replace(
                runtime.persistence,
                object_router=RuntimeObjectRouter(_working_object_stores(storage, runtime.persistence)),
            )
        elif target_kind in {"sqlite", "sql"} and sql_context is not None:
            runtime = build_sql_runtime(
                sql_context,
                namespace=namespace,
                tenant_id=tenant_id,
                plan=storage.plan,
            )
            from ._persistence import RuntimeObjectRouter

            runtime.persistence = replace(
                runtime.persistence,
                object_router=RuntimeObjectRouter(_working_object_stores(storage, runtime.persistence)),
            )
        else:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        await runtime.initialize()
        archives = {}
        if target_kind == "filesystem":
            path = storage.target_path
            if path is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            for domain in (RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY):
                if domain in durable:
                    archives[domain] = FilesystemStepArchive._runtime(
                        path,
                        namespace=namespace,
                        tenant_id=tenant_id,
                        runtime_domain=domain,
                        object_store=runtime.persistence.object_store(domain),
                        writer_lock=runtime.writer_lock,
                    )
                elif storage.plan.route(domain).retention is RuntimeRetention.VOLATILE or (storage.plan.route(domain).retention is RuntimeRetention.TRANSIENT and domain is RuntimeDomain.CONVERSATION):
                    archives[domain] = InMemoryStepArchive(domain)
        elif target_kind in {"sqlite", "sql"}:
            if sql_context is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            for domain in (RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY):
                if domain in durable:
                    archives[domain] = SqlStepArchive._runtime(
                        sql_context.engine,
                        namespace=namespace,
                        tenant_id=tenant_id,
                        runtime_domain=domain,
                        object_store=runtime.persistence.object_store(domain),
                        context=sql_context,
                    )
                elif storage.plan.route(domain).retention is RuntimeRetention.VOLATILE or (storage.plan.route(domain).retention is RuntimeRetention.TRANSIENT and domain is RuntimeDomain.CONVERSATION):
                    archives[domain] = InMemoryStepArchive(domain)
        elif target_kind == "memory":
            from ._persistence import RuntimeObjectRouter

            runtime.persistence = replace(
                runtime.persistence,
                object_router=RuntimeObjectRouter(_working_object_stores(storage, runtime.persistence)),
            )
            for domain in (RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY):
                if storage.plan.route(domain).retention is RuntimeRetention.VOLATILE or (storage.plan.route(domain).retention is RuntimeRetention.TRANSIENT and domain is RuntimeDomain.CONVERSATION):
                    archives[domain] = InMemoryStepArchive(domain)
        steps = RuntimeStepPersistence(
            StagingStepStore(),
            conversation_archive=archives[RuntimeDomain.CONVERSATION],
            execution_archive=archives.get(RuntimeDomain.EXECUTION),
            recovery_archive=archives.get(RuntimeDomain.RECOVERY),
            conversation_retention=storage.plan.route(RuntimeDomain.CONVERSATION).retention,
            execution_retention=storage.plan.route(RuntimeDomain.EXECUTION).retention,
            recovery_retention=storage.plan.route(RuntimeDomain.RECOVERY).retention,
        )
        await steps.initialize()
    except BaseException:
        if steps is not None:
            try:
                await steps.preflight_close()
                await steps.close()
            except BaseException:
                _logger.error("runtime step cleanup failed during setup", exc_info=environ.debug)
        if runtime is not None:
            try:
                await runtime.close()
            except BaseException:
                _logger.error("runtime cleanup failed during setup", exc_info=environ.debug)
        if owned_context and sql_context is not None:
            try:
                await sql_context.close()
            except BaseException:
                _logger.error("SQL context cleanup failed during setup", exc_info=environ.debug)
        raise
    if steps is None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    retention = _RuntimeRetention(
        runtime.persistence,
        steps,
        storage.plan,
        namespace=namespace,
    )
    close_phase = {"value": 0}

    async def close_persistence() -> None:
        phases: tuple[tuple[Callable[[], Awaitable[None]] | None, str], ...] = (
            (steps.preflight_close, "step preflight"),
            (retention.close, "retention"),
            (steps.close, "steps"),
            (runtime.close, "runtime stores"),
            (sql_context.close if owned_context and sql_context is not None else None, "SQL context"),
        )
        while close_phase["value"] < len(phases):
            callback, label = phases[close_phase["value"]]
            if callback is None:
                close_phase["value"] += 1
                continue
            try:
                await callback()
            except BaseException:
                _logger.error("runtime persistence close phase failed: phase=%s", label, exc_info=environ.debug)
                raise
            close_phase["value"] += 1

    persistence = RuntimePersistence(runtime.persistence, steps, retention, close_persistence)
    try:
        yield persistence
    finally:
        await persistence.close()


__all__ = ["RuntimePersistence", "open_runtime_persistence"]


def runtime_storage_kind(storage: RuntimeStorage) -> str:
    return storage.target_kind


def runtime_storage_path(storage: RuntimeStorage) -> Path | None:
    return storage.target_path


def runtime_storage_engine(storage: RuntimeStorage) -> "AsyncEngine | None":
    return storage.target_engine


def runtime_durable_domains(storage: RuntimeStorage) -> frozenset[RuntimeDomain]:
    return frozenset(domain for domain in RuntimeDomain if storage.plan.route(domain).retention is RuntimeRetention.DURABLE)


__all__ += ["runtime_durable_domains", "runtime_storage_engine", "runtime_storage_kind", "runtime_storage_path"]


def _working_object_stores(storage: RuntimeStorage, persistence: RuntimeStores) -> dict[RuntimeDomain, ObjectStore]:
    stores: dict[RuntimeDomain, ObjectStore] = {}
    for domain in RuntimeDomain:
        route = storage.plan.route(domain)
        if route.object_store is not None:
            stores[domain] = route.object_store
        elif route.retention is RuntimeRetention.TRANSIENT:
            stores[domain] = TransientObjectStore()
        else:
            stores[domain] = persistence.object_store(domain)
    return stores
