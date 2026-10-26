#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physical resource groups and ownership for requested state stores."""

import asyncio
import hashlib
import json
import os
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..core import canonical_sha256
from ..errors import AIError, ErrorCode
from ..observe.stream import await_stream_cleanup
from ..storage import (
    FilesystemMutationLock,
    SqlStorageContext,
    StorageRoute,
    create_sql_storage_context,
    sync_directory,
)
from ._filesystem import FilesystemStateStorageGroup
from ._memory import InMemoryStateStorageGroup
from ._sql import SqlStateStorageGroup

if TYPE_CHECKING:
    from sqlalchemy import MetaData


@dataclass(frozen=True, slots=True)
class StorageResourceRequest:
    name: str
    route: StorageRoute
    member_path: Path | None = None
    configure_schema: Callable[["MetaData"], None] | None = None
    uses_archive: bool = False


@dataclass(frozen=True, slots=True)
class MemoryResource:
    group: InMemoryStateStorageGroup


@dataclass(frozen=True, slots=True)
class FilesystemResource:
    group: FilesystemStateStorageGroup
    member_path: Path


@dataclass(frozen=True, slots=True)
class SqlResource:
    group: SqlStateStorageGroup
    context: SqlStorageContext
    metadata: "MetaData"


class StorageResourceScope:
    def __init__(self) -> None:
        self.resources: tuple[MemoryResource | FilesystemResource | SqlResource, ...] = ()
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
            await scope.initialize(requests, namespace=namespace, tenant_id=tenant_id, read_only=read_only)
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
        resources: dict[int, MemoryResource | FilesystemResource | SqlResource] = {}
        memory = MemoryResource(InMemoryStateStorageGroup(read_only=read_only))
        self._close_actions.append(memory.group.close)
        filesystem_groups: dict[Path, list[int]] = {}
        sql_groups: dict[tuple[str, object], list[int]] = {}
        member_roots: dict[str, Path] = {}
        for index, request in enumerate(requests):
            route = request.route
            if route.kind == "memory":
                resources[index] = memory
            elif route.kind == "filesystem":
                if route.path is None or request.member_path is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                root = route.transaction_root or route.path
                filesystem_groups.setdefault(root, []).append(index)
                member_roots[request.name] = request.member_path
            elif route.kind in {"sqlite", "sql"}:
                if route.kind == "sqlite":
                    if route.path is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if read_only and not route.path.exists():
                        raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                    key = ("sqlite", route.path)
                else:
                    if route.engine is None:
                        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                    key = ("sql", route.engine)
                sql_groups.setdefault(key, []).append(index)
            else:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        _validate_filesystem_member_roots(member_roots, tuple(filesystem_groups))
        for root, indices in filesystem_groups.items():
            route = requests[indices[-1]].route
            members = {requests[index].name: member_roots[requests[index].name] for index in indices}
            if read_only and any(not path.is_dir() for path in members.values()):
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            group = FilesystemStateStorageGroup(
                root,
                namespace=namespace,
                tenant_id=tenant_id,
                scope_digest=_filesystem_group_scope(namespace, tenant_id),
                read_only=read_only,
            )
            self._close_actions.append(group.close)
            for index in indices:
                resources[index] = FilesystemResource(group, members[requests[index].name])

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
                if not read_only:
                    await asyncio.to_thread(route.path.parent.mkdir, parents=True, exist_ok=True)
                    async with FilesystemMutationLock(route.path.with_name(route.path.name + ".init.lock")):
                        if not route.path.exists():
                            await _provision_sqlite_database(
                                route.path, metadata, register_cleanup=self._close_actions.append,
                            )
                from sqlalchemy import URL
                from sqlalchemy.ext.asyncio import create_async_engine

                engine = create_async_engine(URL.create("sqlite+aiosqlite", database=str(route.path)))
                context = create_sql_storage_context(engine, owns_engine=True)
                self._close_actions.append(context.close)
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
) -> None:
    from sqlalchemy import URL
    from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.init-",
        suffix=".db",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(name)
    engine: AsyncEngine | None = None
    context: SqlStorageContext | None = None
    published = False

    async def cleanup() -> None:
        if context is not None:
            await context.close()
        elif engine is not None:
            await engine.dispose()
        if not published:
            await asyncio.to_thread(_cleanup_sqlite_temporary, temporary)

    register_cleanup(cleanup)
    try:
        engine = create_async_engine(URL.create("sqlite+aiosqlite", database=str(temporary)))
        context = create_sql_storage_context(engine, owns_engine=True)
        await context.initialize()
        async with context.engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
        await context.close()
        _ensure_sqlite_database_is_self_contained(temporary)
        await asyncio.to_thread(_sync_file, temporary)
        try:
            await asyncio.to_thread(os.replace, temporary, path)
            published = True
            await asyncio.to_thread(sync_directory, path.parent)
        except BaseException as error:
            if published:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
            raise
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
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _validate_filesystem_member_roots(
    member_roots: Mapping[str, Path],
    group_roots: tuple[Path, ...],
) -> None:
    ordered = tuple(
        sorted(
            member_roots.items(),
            key=lambda item: (item[1].as_posix(), item[0]),
        )
    )
    for index, (left_name, left) in enumerate(ordered):
        for right_name, right in ordered[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(
                    "filesystem StorageRoute member paths overlap: "
                    f"{left_name}={left} {right_name}={right}"
                )
    reserved_roots = tuple(
        root / reserved
        for root in dict.fromkeys(group_roots)
        for reserved in (".state-groups", ".state-archive", ".state-archive-objects", ".state-partitions")
    )
    for name, member in ordered:
        for reserved in reserved_roots:
            if (
                member == reserved
                or member in reserved.parents
                or reserved in member.parents
            ):
                raise ValueError(
                    "filesystem StorageRoute member path overlaps "
                    f"coordination storage: {name}={member}"
                )


def _filesystem_group_scope(namespace: str, tenant_id: str) -> str:
    # The physical transaction root already scopes this metadata locator.
    # Opening fewer domain routes must not create a new persistent group.
    return canonical_sha256({"namespace": namespace, "tenant_id": tenant_id})[:32]


__all__ = ["reject_filesystem_capability_partitions", "StorageResourceRequest", "StorageResourceScope", "MemoryResource", "FilesystemResource", "SqlResource"]

def reject_filesystem_capability_partitions(routes: Sequence[StorageRoute], namespace: str, tenant_id: str) -> None:
    """Recognize existing group membership before opening a reduced resource set."""
    roots = {route.transaction_root or route.path for route in routes if route.kind == "filesystem"}
    namespace_digest = hashlib.sha256(namespace.encode()).hexdigest()
    tenant_digest = hashlib.sha256(tenant_id.encode()).hexdigest()
    for root in roots:
        if root is None:
            continue
        for path in (root / ".state-groups").glob("*/manifest.json"):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if not isinstance(manifest, dict):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if manifest.get("namespace_digest") != namespace_digest or manifest.get("tenant_digest") != tenant_digest:
                continue
            members = manifest.get("members")
            if (manifest.get("format") != "linktools-ai-state-group" or manifest.get("version") != 1
                    or not isinstance(members, list)
                    or any(not isinstance(member, dict) or not isinstance(member.get("runtime_domain"), str) for member in members)):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if any(member["runtime_domain"].startswith("capability:") for member in members):
                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED, "Capability persistence partitions are not supported by portable snapshots")
