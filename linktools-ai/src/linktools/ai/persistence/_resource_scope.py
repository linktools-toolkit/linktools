#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physical resource groups and ownership for requested state stores."""

import asyncio
import os
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..errors import AIError, ErrorCode
from ..observe.stream import await_stream_cleanup
from ..storage import (
    FilesystemMutationLock,
    SqlStorageContext,
    StorageRoute,
    create_sql_storage_context,
    create_sqlite_storage_context,
    sync_directory,
)
from ._memory import InMemoryStateStorageGroup
from ._sql import SqlStateStorageGroup

if TYPE_CHECKING:
    from sqlalchemy import MetaData


@dataclass(frozen=True, slots=True)
class StorageResourceRequest:
    name: str
    route: StorageRoute
    configure_schema: Callable[["MetaData"], None] | None = None
    uses_archive: bool = False


@dataclass(frozen=True, slots=True)
class MemoryResource:
    group: InMemoryStateStorageGroup


@dataclass(frozen=True, slots=True)
class SqlResource:
    group: SqlStateStorageGroup
    context: SqlStorageContext
    metadata: "MetaData"


class StorageResourceScope:
    def __init__(self) -> None:
        self.resources: tuple[MemoryResource | SqlResource, ...] = ()
        self._close_actions: list[Callable[[], Awaitable[None]]] = []
        self._close_lock = asyncio.Lock()

    @classmethod
    async def open(
        cls,
        requests: tuple[StorageResourceRequest, ...],
        *,
        namespace: str,
        tenant_id: str,
        read_only: bool = False,
    ) -> "StorageResourceScope":
        scope = cls()
        try:
            await scope.initialize(requests, namespace=namespace, tenant_id=tenant_id,
                                   read_only=read_only)
        except BaseException as primary_error:
            try:
                await await_stream_cleanup(scope.close(), None)
            except BaseException as cleanup_error:
                raise primary_error from cleanup_error
            raise
        return scope

    async def initialize(
        self,
        requests: tuple[StorageResourceRequest, ...],
        *,
        namespace: str,
        tenant_id: str,
        read_only: bool = False,
    ) -> None:
        if len({request.name for request in requests}) != len(requests):
            raise ValueError("storage resource request names must be unique")
        resources: dict[int, MemoryResource | SqlResource] = {}
        memory = MemoryResource(InMemoryStateStorageGroup(read_only=read_only))
        self._close_actions.append(memory.group.close)
        sql_groups: dict[tuple[str, object], list[int]] = {}
        for index, request in enumerate(requests):
            route = request.route
            if route.kind == "memory":
                resources[index] = memory
            elif route.kind in {"sqlite", "sql"}:
                if route.kind == "sqlite":
                    if route.path is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    key = ("sqlite", route.path)
                else:
                    if route.engine is None:
                        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                    key = ("sql", route.engine)
                sql_groups.setdefault(key, []).append(index)
            else:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        for key, indices in sql_groups.items():
            if key[0] == "sqlite" and len({requests[index].route.exclusive_writer for index in indices}) != 1:
                raise ValueError("SQLite routes for the same path must agree on exclusive_writer")
        for key, indices in sql_groups.items():
            from sqlalchemy import MetaData

            route = requests[indices[-1]].route
            metadata = MetaData()
            for index in indices:
                configure = requests[index].configure_schema
                if configure is not None:
                    configure(metadata)
            if key[0] == "sqlite":
                if route.path is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if read_only and not route.path.exists():
                    raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                context = create_sqlite_storage_context(route.path)
                self._close_actions.append(context.close)
                if not read_only:
                    await asyncio.to_thread(route.path.parent.mkdir, parents=True, exist_ok=True)
                    async with FilesystemMutationLock(route.path.with_name(route.path.name + ".init.lock")):
                        if not route.path.exists():
                            local = route.exclusive_writer
                            if local:
                                await context.acquire_local_sqlite(route.path)
                            await _provision_sqlite_database(
                                route.path, metadata, register_cleanup=self._close_actions.append, local=local,
                            )
                        elif route.exclusive_writer:
                            if not await context.is_local_sqlite_source(route.path):
                                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED,
                                    "Shared SQLite cannot be upgraded online; restore offline into a new exclusive root")
                            await context.acquire_local_sqlite(route.path)
            else:
                if route.engine is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                context = create_sql_storage_context(route.engine)
            group = SqlStateStorageGroup(
                context, metadata, read_only=read_only,
                uses_archive=any(requests[index].uses_archive for index in indices),
            )
            self._close_actions.append(group.close)
            resource = SqlResource(group, context, metadata)
            for index in indices:
                resources[index] = resource
        self.resources = tuple(resources[index] for index in range(len(requests)))

    async def close(self) -> None:
        async with self._close_lock:
            while self._close_actions:
                await self._close_actions[-1]()
                self._close_actions.pop()


async def _provision_sqlite_database(
    path: Path, metadata: "MetaData", *,
    register_cleanup: Callable[[Callable[[], Awaitable[None]]], None],
    local: bool = False,
) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.init-",
        suffix=".db",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(name)
    context: SqlStorageContext | None = None
    async def cleanup() -> None:
        if context is not None:
            await context.close()
        await asyncio.to_thread(_cleanup_sqlite_temporary, temporary)

    register_cleanup(cleanup)
    try:
        context = create_sqlite_storage_context(temporary)
        await context.initialize()
        async with context.engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
            if local:
                await connection.exec_driver_sql(f"PRAGMA main.application_id={SqlStorageContext.LOCAL_SQLITE_APPLICATION_ID}")
        await context.close()
        _ensure_sqlite_database_is_self_contained(temporary)
        await await_stream_cleanup(asyncio.to_thread(_sync_file, temporary), None)

        def publish() -> None:
            # A writer using another entry point need not hold our init lock.
            os.link(temporary, path)
            try:
                sync_directory(path.parent)
            except BaseException as error:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error

        await await_stream_cleanup(asyncio.to_thread(publish), None)
    except BaseException as primary_error:
        try:
            await await_stream_cleanup(cleanup(), None)
        except BaseException as cleanup_error:
            raise primary_error from cleanup_error
        raise
    else:
        await await_stream_cleanup(cleanup(), None)


def _ensure_sqlite_database_is_self_contained(path: Path) -> None:
    sidecars = tuple(
        candidate
        for candidate in _sqlite_sidecars(path)
        if candidate.exists()
    )
    if sidecars:
        raise AIError(
            ErrorCode.STORAGE_RECOVERY_REQUIRED,
            safe_details={
                "phase": "sqlite_initial_publish",
                "reason": "sidecar_present",
            },
        )


def _sqlite_sidecars(path: Path) -> tuple[Path, ...]:
    return tuple(
        Path(str(path) + suffix)
        for suffix in ("-journal", "-wal", "-shm")
    )


def _cleanup_sqlite_temporary(path: Path) -> None:
    for candidate in (path, *_sqlite_sidecars(path)):
        candidate.unlink(missing_ok=True)


def _sync_file(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


__all__ = ["StorageResourceRequest", "StorageResourceScope", "MemoryResource", "SqlResource"]
