#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable, content-addressed ObjectStore implementations."""

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Protocol,
    runtime_checkable,
)

from linktools.core import environ

from ..errors import AIError, ErrorCode
from ._database import create_sql_storage_context
from ._dialects import (
    dialect_for_name,
    sql_audit_columns,
    sql_audit_indexes,
    sql_blob,
    sql_id_column,
    sql_sha256,
    sql_table_options,
    sql_text_key,
    sql_unique,
)
from ._files import sync_directory, write_json_atomic
from ._lock import FilesystemMutationLock

if TYPE_CHECKING:
    from sqlalchemy import MetaData
    from sqlalchemy.ext.asyncio import AsyncEngine

    from ._database import SqlStorageContext

_logger = environ.get_logger("ai.storage.object")
_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ObjectRef:
    store_id: str
    key: str
    digest: str
    size: int

    def __post_init__(self) -> None:
        _validate_store_id(self.store_id)
        _validate_key(self.key)
        _validate_digest(self.digest)
        if not isinstance(self.size, int) or self.size < 0:
            raise ValueError("object size must be non-negative")


@dataclass(frozen=True, slots=True)
class ObjectStat:
    key: str
    digest: str
    size: int


class ObjectStore(Protocol):
    @property
    def store_id(self) -> str: ...

    async def put(
        self,
        key: str,
        chunks: AsyncIterator[bytes],
        *,
        expected_size: int,
        expected_digest: str,
    ) -> ObjectStat: ...

    async def stat(self, key: str) -> ObjectStat | None: ...

    async def validate_integrity(self) -> None: ...

    def open(self, key: str) -> AsyncIterator[bytes]: ...


def runtime_object_key(
    *,
    namespace_digest: str,
    tenant_digest: str,
    stored_digest: str,
) -> str:
    """Build the tenant-scoped physical key for immutable Runtime bytes."""
    for value in (namespace_digest, tenant_digest, stored_digest):
        _validate_digest(value)
    return f"v2/runtime/{namespace_digest}/{tenant_digest}/{stored_digest}"


@runtime_checkable
class ObjectStoreInspection(Protocol):
    def list_objects(self) -> AsyncIterator[ObjectStat]: ...


@runtime_checkable
class ObjectStoreMaintenance(ObjectStoreInspection, Protocol):
    async def delete_object(self, key: str, *, expected_digest: str) -> bool: ...

    def offline_exclusivity(self) -> AbstractAsyncContextManager[None]: ...


class InMemoryObjectStore:
    def __init__(self, store_id: str = "memory") -> None:
        _validate_store_id(store_id)
        self._store_id = store_id
        self._objects: dict[str, bytes] = {}

    @property
    def store_id(self) -> str:
        return self._store_id

    async def put(
        self,
        key: str,
        chunks: AsyncIterator[bytes],
        *,
        expected_size: int,
        expected_digest: str,
    ) -> ObjectStat:
        data, digest = await _spool_memory(chunks, expected_size)
        if len(data) != expected_size or digest != expected_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        current = self._objects.get(key)
        if current is not None and current != data:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._objects[key] = data
        return ObjectStat(key, digest, len(data))

    async def stat(self, key: str) -> ObjectStat | None:
        _validate_key(key)
        value = self._objects.get(key)
        return None if value is None else ObjectStat(key, _digest(value), len(value))

    async def validate_integrity(self) -> None:
        return None

    @asynccontextmanager
    async def offline_exclusivity(self) -> AsyncIterator[None]:
        yield

    async def _list_objects(self) -> AsyncIterator[ObjectStat]:
        for key, value in self._objects.items():
            yield ObjectStat(key, _digest(value), len(value))

    def list_objects(self) -> AsyncIterator[ObjectStat]:
        return self._list_objects()

    async def delete_object(self, key: str, *, expected_digest: str) -> bool:
        _validate_key(key)
        value = self._objects.get(key)
        if value is None:
            return False
        if _digest(value) != expected_digest:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        del self._objects[key]
        return True

    async def _open(self, key: str) -> AsyncIterator[bytes]:
        _validate_key(key)
        value = self._objects.get(key)
        if value is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        for offset in range(0, len(value), _CHUNK_SIZE):
            yield value[offset : offset + _CHUNK_SIZE]

    def open(self, key: str) -> AsyncIterator[bytes]:
        return self._open(key)


class TransientObjectStore(InMemoryObjectStore):
    def __init__(self) -> None:
        super().__init__("transient")
        self._scopes: dict[str, set[str]] = {}

    def scoped(self, scope: str) -> ObjectStore:
        if not scope or "/" in scope:
            raise ValueError("transient object scope is invalid")
        return _ScopedObjectStore(self, scope)

    async def release_scope(self, scope: str) -> None:
        for key in self._scopes.pop(scope, set()):
            self._objects.pop(key, None)

    def clear(self) -> None:
        self._objects.clear()
        self._scopes.clear()


class _ScopedObjectStore:
    def __init__(self, parent: TransientObjectStore, scope: str) -> None:
        self._parent = parent
        self._scope = scope

    @property
    def store_id(self) -> str:
        return self._parent.store_id

    async def put(
        self, key: str, chunks: AsyncIterator[bytes], *, expected_size: int, expected_digest: str
    ) -> ObjectStat:
        physical_key = self._physical_key(key)
        value = await self._parent.put(
            physical_key,
            chunks,
            expected_size=expected_size,
            expected_digest=expected_digest,
        )
        self._parent._scopes.setdefault(self._scope, set()).add(physical_key)
        return ObjectStat(key, value.digest, value.size)

    async def stat(self, key: str) -> ObjectStat | None:
        value = await self._parent.stat(self._physical_key(key))
        return None if value is None else ObjectStat(key, value.digest, value.size)

    async def validate_integrity(self) -> None:
        await self._parent.validate_integrity()

    def offline_exclusivity(self) -> AbstractAsyncContextManager[None]:
        return self._parent.offline_exclusivity()

    async def _list_objects(self) -> AsyncIterator[ObjectStat]:
        values = [value async for value in self._parent.list_objects()]
        prefix = f"{self._scope}/"
        for value in values:
            if value.key.startswith(prefix):
                yield ObjectStat(value.key[len(prefix) :], value.digest, value.size)

    def list_objects(self) -> AsyncIterator[ObjectStat]:
        return self._list_objects()

    async def delete_object(self, key: str, *, expected_digest: str) -> bool:
        return await self._parent.delete_object(
            self._physical_key(key),
            expected_digest=expected_digest,
        )

    def open(self, key: str) -> AsyncIterator[bytes]:
        return self._parent.open(self._physical_key(key))

    def _physical_key(self, key: str) -> str:
        _validate_key(key)
        return f"{self._scope}/{key}"


class FilesystemObjectStore:
    def __init__(self, root: str | Path, *, store_id: str = "builtin") -> None:
        _validate_store_id(store_id)
        self._root = Path(root).expanduser().resolve()
        self._store_id = store_id
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._offline_owner: asyncio.Task[Any] | None = None
        self._offline_tasks: set[asyncio.Task[Any]] = set()

    @property
    def store_id(self) -> str:
        return self._store_id

    def _paths(self, key: str) -> tuple[Path, Path]:
        digest = _key_digest(self.store_id, key).hex()
        root = self._root / "objects" / digest[:2]
        return root / f"{digest}.bin", root / f"{digest}.json"

    async def put(
        self, key: str, chunks: AsyncIterator[bytes], *, expected_size: int, expected_digest: str
    ) -> ObjectStat:
        _validate_put(key, expected_size, expected_digest)
        owner = asyncio.current_task()
        offline_owned = owner is not None and owner is self._offline_owner
        task = asyncio.create_task(
            self._put_owned(
                key,
                chunks,
                expected_size=expected_size,
                expected_digest=expected_digest,
                acquire_lock=not offline_owned,
            ),
            name=f"filesystem-object-put-{_key_digest(self.store_id, key).hex()[:12]}",
        )
        _track_object_task(self._background_tasks, task, "filesystem object put")
        if offline_owned:
            self._offline_tasks.add(task)
            task.add_done_callback(self._offline_tasks.discard)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as error:
            if task.done():
                task.result()
                raise
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from error

    async def _put_owned(
        self,
        key: str,
        chunks: AsyncIterator[bytes],
        *,
        expected_size: int,
        expected_digest: str,
        acquire_lock: bool,
    ) -> ObjectStat:
        temporary_root = self._root / ".tmp"
        await asyncio.to_thread(temporary_root.mkdir, parents=True, exist_ok=True)
        name = await asyncio.to_thread(_create_temp_file, temporary_root)
        try:
            size, digest = await _spool_file(chunks, name, expected_size)
            if digest != expected_digest or size != expected_size:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            destination, metadata = self._paths(key)

            async def publish() -> None:
                nonlocal name
                duplicate = await asyncio.to_thread(
                    _publish_filesystem_object,
                    name,
                    destination,
                    metadata,
                    key,
                    expected_size,
                    expected_digest,
                )
                if duplicate:
                    _logger.debug(
                        "filesystem object duplicate accepted by metadata: key=%s",
                        key,
                    )
                else:
                    name = None

            if acquire_lock:
                async with FilesystemMutationLock(self._root / "object.lock"):
                    await publish()
            else:
                await publish()
            return ObjectStat(key, expected_digest, expected_size)
        finally:
            if name is not None:
                await asyncio.to_thread(name.unlink, missing_ok=True)

    async def stat(self, key: str) -> ObjectStat | None:
        _validate_key(key)
        destination, metadata = self._paths(key)
        return await asyncio.to_thread(
            _stat_filesystem_object,
            metadata,
            destination,
            key,
        )

    async def validate_integrity(self) -> None:
        await asyncio.to_thread(
            _validate_filesystem_objects,
            self._root,
            self._store_id,
        )

    @asynccontextmanager
    async def offline_exclusivity(self) -> AsyncIterator[None]:
        lock = FilesystemMutationLock(self._root / "object.lock")
        await lock.__aenter__()
        owner = asyncio.current_task()
        if owner is None:
            await lock.__aexit__(None, None, None)
            raise RuntimeError("offline ObjectStore exclusivity requires an asyncio task")
        self._offline_owner = owner
        try:
            yield
        finally:
            self._offline_owner = None
            pending = tuple(task for task in self._offline_tasks if not task.done())
            if pending:
                cleanup = asyncio.create_task(
                    _release_offline_lock(lock, pending),
                    name="filesystem-object-offline-release",
                )
                _track_object_task(
                    self._background_tasks,
                    cleanup,
                    "filesystem object offline lock release",
                )
            else:
                await lock.__aexit__(None, None, None)

    async def _list_objects(self) -> AsyncIterator[ObjectStat]:
        values = await asyncio.to_thread(
            _list_filesystem_objects,
            self._root,
            self._store_id,
        )
        for value in values:
            yield value

    def list_objects(self) -> AsyncIterator[ObjectStat]:
        return self._list_objects()

    async def delete_object(self, key: str, *, expected_digest: str) -> bool:
        _validate_key(key)
        owner = asyncio.current_task()
        offline_owned = owner is not None and owner is self._offline_owner
        task = asyncio.create_task(
            self._delete_owned(
                key,
                expected_digest=expected_digest,
                acquire_lock=not offline_owned,
            ),
            name=f"filesystem-object-delete-{_key_digest(self.store_id, key).hex()[:12]}",
        )
        _track_object_task(self._background_tasks, task, "filesystem object delete")
        if offline_owned:
            self._offline_tasks.add(task)
            task.add_done_callback(self._offline_tasks.discard)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as error:
            if task.done():
                task.result()
                raise
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from error

    async def _delete_owned(
        self,
        key: str,
        *,
        expected_digest: str,
        acquire_lock: bool,
    ) -> bool:
        destination, metadata = self._paths(key)

        async def delete() -> bool:
            current = await asyncio.to_thread(
                _stat_filesystem_object,
                metadata,
                destination,
                key,
            )
            if current is None:
                return False
            if current.digest != expected_digest:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return await asyncio.to_thread(
                _delete_filesystem_object,
                destination,
                metadata,
            )

        if acquire_lock:
            async with FilesystemMutationLock(self._root / "object.lock"):
                return await delete()
        return await delete()

    async def _open(self, key: str) -> AsyncIterator[bytes]:
        _validate_key(key)
        destination, metadata = self._paths(key)
        expected = await asyncio.to_thread(
            _read_filesystem_metadata,
            metadata,
            destination,
            key,
        )
        digest = hashlib.sha256()
        size = 0
        open_task = asyncio.create_task(asyncio.to_thread(destination.open, "rb"))
        try:
            handle: BinaryIO | None = await asyncio.shield(open_task)
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(
                _close_after_open(open_task),
                name="filesystem-object-open-cleanup",
            )
            _track_object_task(
                self._background_tasks,
                cleanup,
                "filesystem object open cleanup",
            )
            raise
        try:
            while True:
                if handle is None:
                    raise RuntimeError("filesystem object handle ownership was transferred")
                read_task = asyncio.create_task(
                    asyncio.to_thread(handle.read, _CHUNK_SIZE)
                )
                try:
                    chunk = await asyncio.shield(read_task)
                except asyncio.CancelledError:
                    cleanup = asyncio.create_task(
                        _close_after_read(read_task, handle),
                        name="filesystem-object-read-cleanup",
                    )
                    _track_object_task(
                        self._background_tasks,
                        cleanup,
                        "filesystem object read cleanup",
                    )
                    handle = None
                    raise
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                yield chunk
            if expected.size != size or expected.digest != digest.hexdigest():
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        finally:
            if handle is not None:
                close_task = asyncio.create_task(asyncio.to_thread(handle.close))
                try:
                    await asyncio.shield(close_task)
                except asyncio.CancelledError:
                    _track_object_task(
                        self._background_tasks,
                        close_task,
                        "filesystem object close cleanup",
                    )
                    raise

    def open(self, key: str) -> AsyncIterator[bytes]:
        return self._open(key)


class SqlObjectStore:
    def __init__(
        self,
        engine: "AsyncEngine",
        *,
        store_id: str = "builtin",
        context: "SqlStorageContext | None" = None,
    ) -> None:
        from sqlalchemy.ext.asyncio import AsyncEngine

        _validate_store_id(store_id)
        if not isinstance(engine, AsyncEngine):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if context is not None and context.engine is not engine:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        dialect_for_name(engine.dialect.name)
        self._store_id = store_id
        self._context = context or create_sql_storage_context(engine)
        self._metadata = build_object_sql_metadata()
        self._background_tasks: set[asyncio.Task[Any]] = set()

    @classmethod
    def from_context(cls, context: "SqlStorageContext", *, store_id: str = "builtin") -> "SqlObjectStore":
        return cls(context.engine, store_id=store_id, context=context)

    @property
    def store_id(self) -> str:
        return self._store_id

    async def put(
        self, key: str, chunks: AsyncIterator[bytes], *, expected_size: int, expected_digest: str
    ) -> ObjectStat:
        _validate_put(key, expected_size, expected_digest)
        task = asyncio.create_task(
            self._put_owned(
                key,
                chunks,
                expected_size=expected_size,
                expected_digest=expected_digest,
            ),
            name=f"sql-object-put-{_key_digest(self.store_id, key).hex()[:12]}",
        )
        _track_object_task(self._background_tasks, task, "SQL object put")
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as error:
            if task.done():
                task.result()
                raise
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from error

    async def _put_owned(
        self,
        key: str,
        chunks: AsyncIterator[bytes],
        *,
        expected_size: int,
        expected_digest: str,
    ) -> ObjectStat:
        from sqlalchemy.exc import IntegrityError

        temporary_root = await asyncio.to_thread(
            lambda: Path(tempfile.mkdtemp(prefix="linktools-object-"))
        )
        temporary = temporary_root / "payload"
        try:
            size, digest = await _spool_file(chunks, temporary, expected_size)
            if size != expected_size or digest != expected_digest:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                await self._insert(key, temporary, size, digest)
            except IntegrityError:
                existing = await self.stat(key)
                if existing is None or existing.digest != digest or existing.size != size:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
            return ObjectStat(key, digest, size)
        finally:
            await asyncio.to_thread(shutil.rmtree, temporary_root, ignore_errors=True)

    async def _insert(self, key: str, path: Path, size: int, digest: str) -> None:
        from sqlalchemy import insert

        table = self._metadata.tables["ai_objects"]
        chunks = self._metadata.tables["ai_object_chunks"]
        key_digest = _key_digest(self.store_id, key)
        async def execute(session) -> None:
            await session.execute(
                insert(table).values(
                    key_digest=key_digest.hex(),
                    store_id=self.store_id,
                    object_key=key,
                    content_digest=digest,
                    size=size,
                )
            )
            index = 0
            offset = 0
            while True:
                rows, offset = await asyncio.to_thread(
                    _read_payload_batch,
                    path,
                    offset,
                    index,
                    64,
                )
                if not rows:
                    break
                for row in rows:
                    row["key_digest"] = key_digest.hex()
                await session.execute(insert(chunks), rows)
                index += len(rows)
        await self._context.run_mutation(execute, domain="storage.object")

    async def stat(self, key: str) -> ObjectStat | None:
        _validate_key(key)
        from sqlalchemy import select

        table = self._metadata.tables["ai_objects"]
        session = self._context.sessions()
        try:
            row = (
                (
                    await session.execute(
                        select(table).where(table.c.key_digest == _key_digest(self.store_id, key).hex())
                    )
                )
                .mappings()
                .one_or_none()
            )
        finally:
            await session.close()
        if row is None:
            return None
        if row["store_id"] != self.store_id or row["object_key"] != key:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return ObjectStat(key, str(row["content_digest"]), int(row["size"]))

    async def validate_integrity(self) -> None:
        from sqlalchemy import select

        objects = self._metadata.tables["ai_objects"]
        chunks = self._metadata.tables["ai_object_chunks"]
        session = self._context.sessions()
        try:
            headers = (
                (
                    await session.execute(
                        select(
                            objects.c.key_digest,
                            objects.c.store_id,
                            objects.c.object_key,
                            objects.c.size,
                            objects.c.content_digest,
                        ).where(objects.c.store_id == self.store_id)
                    )
                )
                .mappings()
                .all()
            )
            rows = (
                (await session.execute(select(chunks.c.key_digest, chunks.c.chunk_index, chunks.c.content)))
                .mappings()
                .all()
            )
            all_object_keys = {str(value) for value in await session.scalars(select(objects.c.key_digest))}
        finally:
            await session.close()
        header_keys = {str(row["key_digest"]) for row in headers}
        if any(str(row["key_digest"]) not in all_object_keys for row in rows):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        grouped: dict[str, list[Mapping[str, object]]] = {}
        for row in rows:
            key = str(row["key_digest"])
            if key not in all_object_keys:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if key not in header_keys:
                continue
            grouped.setdefault(key, []).append(row)
        for header in headers:
            key = str(header["key_digest"])
            if _key_digest(str(header["store_id"]), str(header["object_key"])).hex() != key:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            values = sorted(grouped.get(key, ()), key=lambda row: int(row["chunk_index"]))
            if [int(row["chunk_index"]) for row in values] != list(range(len(values))):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            digest = hashlib.sha256()
            size = 0
            for row in values:
                content = bytes(row["content"])
                digest.update(content)
                size += len(content)
            if size != int(header["size"]) or digest.hexdigest() != str(header["content_digest"]):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _list_objects(self) -> AsyncIterator[ObjectStat]:
        from sqlalchemy import select

        table = self._metadata.tables["ai_objects"]
        session = self._context.sessions()
        try:
            rows = (
                (await session.execute(select(table).where(table.c.store_id == self.store_id)))
                .mappings()
                .all()
            )
        finally:
            await session.close()
        for row in rows:
            yield ObjectStat(
                str(row["object_key"]),
                str(row["content_digest"]),
                int(row["size"]),
            )

    def list_objects(self) -> AsyncIterator[ObjectStat]:
        return self._list_objects()

    async def delete_object(self, key: str, *, expected_digest: str) -> bool:
        _validate_key(key)
        from sqlalchemy import delete, select

        key_digest = _key_digest(self.store_id, key).hex()

        async def execute(session) -> bool:
            table = self._metadata.tables["ai_objects"]
            current = (
                await session.execute(
                    select(table.c.content_digest).where(
                        table.c.key_digest == key_digest,
                        table.c.store_id == self.store_id,
                    )
                )
            ).scalar_one_or_none()
            if current is None:
                return False
            if str(current) != expected_digest:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await session.execute(
                delete(self._metadata.tables["ai_object_chunks"]).where(
                    self._metadata.tables["ai_object_chunks"].c.key_digest == key_digest
                )
            )
            result = await session.execute(
                delete(table).where(
                    self._metadata.tables["ai_objects"].c.key_digest == key_digest,
                    self._metadata.tables["ai_objects"].c.store_id == self.store_id,
                )
            )
            return result.rowcount == 1

        return await self._context.run_mutation(execute, domain="storage.object")

    async def _open(self, key: str) -> AsyncIterator[bytes]:
        _validate_key(key)
        from sqlalchemy import select

        table = self._metadata.tables["ai_objects"]
        chunks = self._metadata.tables["ai_object_chunks"]
        session = self._context.sessions()
        try:
            header = (
                (
                    await session.execute(
                        select(table).where(table.c.key_digest == _key_digest(self.store_id, key).hex())
                    )
                )
                .mappings()
                .one_or_none()
            )
            if header is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            if header["store_id"] != self.store_id or header["object_key"] != key:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result = await session.stream(
                select(chunks).where(chunks.c.key_digest == header["key_digest"]).order_by(chunks.c.chunk_index)
            )
            digest = hashlib.sha256()
            size = 0
            expected_index = 0
            async for row in result.mappings():
                if int(row["chunk_index"]) != expected_index:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                value = bytes(row["content"])
                expected_index += 1
                digest.update(value)
                size += len(value)
                yield value
            if size != int(header["size"]) or digest.hexdigest() != str(header["content_digest"]):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        finally:
            await session.close()

    def open(self, key: str) -> AsyncIterator[bytes]:
        return self._open(key)


def build_object_sql_metadata(metadata: "MetaData | None" = None) -> "MetaData":
    from sqlalchemy import BigInteger, Column, MetaData, Table, Text

    if metadata is None:
        metadata = MetaData()
    if "ai_objects" in metadata.tables:
        return metadata
    digest = sql_sha256()
    objects = Table(
        "ai_objects",
        metadata,
        sql_id_column(),
        Column(
            "key_digest",
            digest,
            nullable=False,
            comment="Canonical SHA-256 identity of the ObjectStore store identifier and object key.",
        ),
        Column(
            "store_id",
            sql_text_key(64),
            nullable=False,
            comment="Logical ObjectStore identifier that namespaces object keys.",
        ),
        Column(
            "object_key", Text, nullable=False, comment="Original opaque object key exposed by the ObjectStore API."
        ),
        Column("content_digest", digest, nullable=False, comment="SHA-256 digest of the immutable object bytes."),
        Column("size", BigInteger, nullable=False, comment="Exact immutable object size in bytes."),
        *sql_audit_columns(),
        comment="Immutable ObjectStore headers containing canonical object identity, content digest, and size.",
        **sql_table_options(),
    )
    sql_unique(objects, "key_digest")
    sql_audit_indexes(objects)
    chunks = Table(
        "ai_object_chunks",
        metadata,
        sql_id_column(),
        Column(
            "key_digest",
            digest,
            nullable=False,
            comment="Canonical SHA-256 identity of the immutable ObjectStore object owning this chunk.",
        ),
        Column(
            "chunk_index",
            BigInteger,
            nullable=False,
            comment="Zero-based ordered chunk position within the immutable object.",
        ),
        Column(
            "content",
            sql_blob(),
            nullable=False,
            comment="Binary content bytes for this immutable object chunk.",
        ),
        *sql_audit_columns(),
        comment="Ordered binary chunks that compose immutable ObjectStore content.",
        **sql_table_options(),
    )
    sql_unique(chunks, "key_digest", "chunk_index")
    sql_audit_indexes(chunks)
    return metadata


async def read_object(store: ObjectStore, key: str, *, expected_digest: str, expected_size: int) -> bytes:
    digest = hashlib.sha256()
    size = 0
    data = bytearray()
    async for chunk in store.open(key):
        data.extend(chunk)
        digest.update(chunk)
        size += len(chunk)
    if size != expected_size or digest.hexdigest() != expected_digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return bytes(data)


async def _spool_memory(chunks: AsyncIterator[bytes], expected_size: int) -> tuple[bytes, str]:
    data = bytearray()
    digest = hashlib.sha256()
    async for chunk in chunks:
        if not isinstance(chunk, bytes) or not chunk:
            raise ValueError("object chunks must be non-empty bytes")
        data.extend(chunk)
        digest.update(chunk)
        if len(data) > expected_size:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return bytes(data), digest.hexdigest()


async def _spool_file(chunks: AsyncIterator[bytes], path: Path, expected_size: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    buffer = bytearray()
    handle = await asyncio.to_thread(path.open, "wb")
    try:
        async for chunk in chunks:
            if not isinstance(chunk, bytes) or not chunk:
                raise ValueError("object chunks must be non-empty bytes")
            size += len(chunk)
            if size > expected_size:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            digest.update(chunk)
            for offset in range(0, len(chunk), _CHUNK_SIZE):
                buffer.extend(chunk[offset : offset + _CHUNK_SIZE])
                while len(buffer) >= _CHUNK_SIZE:
                    value = bytes(buffer[:_CHUNK_SIZE])
                    del buffer[:_CHUNK_SIZE]
                    await asyncio.to_thread(handle.write, value)
        if buffer:
            value = bytes(buffer)
            await asyncio.to_thread(handle.write, value)
        await asyncio.to_thread(_flush_file, handle)
    finally:
        await asyncio.to_thread(handle.close)
    return size, digest.hexdigest()


def _create_temp_file(root: Path) -> Path:
    descriptor, name = tempfile.mkstemp(prefix="object-", dir=root)
    os.close(descriptor)
    return Path(name)


def _flush_file(handle: BinaryIO) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def _stat_filesystem_object(metadata: Path, destination: Path, key: str) -> ObjectStat | None:
    if not metadata.is_file() and not destination.is_file():
        return None
    return _read_filesystem_metadata(metadata, destination, key)


def _publish_filesystem_object(
    temporary: Path,
    destination: Path,
    metadata: Path,
    key: str,
    size: int,
    digest: str,
) -> bool:
    if destination.exists() or metadata.exists():
        current = _read_filesystem_metadata(metadata, destination, key)
        if current.digest != digest or current.size != size:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, destination)
    write_json_atomic(
        metadata,
        {"key": key, "digest": digest, "size": size},
        fsync=True,
    )
    sync_directory(destination.parent)
    return False


def _validate_filesystem_objects(root: Path, store_id: str) -> None:
    expected_files: set[Path] = set()
    for metadata in (root / "objects").glob("*/*.json"):
        try:
            value = json.loads(metadata.read_text(encoding="utf-8"))
            key = value["key"]
            expected_digest = value["digest"]
            expected_size = int(value["size"])
            _validate_key(str(key))
            digest = _key_digest(store_id, str(key)).hex()
            destination = metadata.with_name(f"{digest}.bin")
            expected_metadata = metadata.with_name(f"{digest}.json")
            if metadata != expected_metadata:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stat = _read_filesystem_metadata(metadata, destination, str(key))
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if stat.digest != expected_digest or stat.size != expected_size:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        actual_size, actual_digest = _hash_file(destination)
        if actual_size != expected_size or actual_digest != expected_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        expected_files.update({metadata, destination})
    actual_files = {path for path in (root / "objects").rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _list_filesystem_objects(root: Path, store_id: str) -> tuple[ObjectStat, ...]:
    values: list[ObjectStat] = []
    for metadata in (root / "objects").glob("*/*.json"):
        try:
            value = json.loads(metadata.read_text(encoding="utf-8"))
            key = value["key"]
            destination = metadata.with_suffix(".bin")
            stat = _read_filesystem_metadata(metadata, destination, str(key))
            if _key_digest(store_id, str(key)).hex() != metadata.stem:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        values.append(stat)
    return tuple(values)


def _delete_filesystem_object(destination: Path, metadata: Path) -> bool:
    present = destination.exists() or metadata.exists()
    if not present:
        return False
    destination.unlink(missing_ok=True)
    metadata.unlink(missing_ok=True)
    sync_directory(destination.parent)
    return True


def _read_payload_batch(
    path: Path,
    offset: int,
    index: int,
    limit: int,
) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    with path.open("rb") as handle:
        handle.seek(offset)
        while len(rows) < limit:
            chunk = handle.read(_CHUNK_SIZE)
            if not chunk:
                break
            rows.append({"chunk_index": index + len(rows), "content": chunk})
            offset += len(chunk)
    return rows, offset


def _track_object_task(
    tasks: set[asyncio.Task[Any]],
    task: asyncio.Task[Any],
    label: str,
) -> None:
    tasks.add(task)

    def consume(done: asyncio.Task[Any]) -> None:
        try:
            done.result()
        except asyncio.CancelledError:
            pass
        except BaseException:  # noqa: BLE001
            _logger.exception("background %s failed", label)
        finally:
            tasks.discard(done)

    task.add_done_callback(consume)


async def _release_offline_lock(
    lock: FilesystemMutationLock,
    tasks: tuple[asyncio.Task[Any], ...],
) -> None:
    await asyncio.gather(*tasks, return_exceptions=True)
    await lock.__aexit__(None, None, None)


async def _close_after_open(task: asyncio.Task[BinaryIO]) -> None:
    handle = await task
    await asyncio.to_thread(handle.close)


async def _close_after_read(
    task: asyncio.Task[bytes],
    handle: BinaryIO,
) -> None:
    try:
        await task
    finally:
        await asyncio.to_thread(handle.close)


def _key_digest(store_id: str, key: str) -> bytes:
    return hashlib.sha256(store_id.encode("utf-8") + b"\0" + key.encode("utf-8")).digest()


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_filesystem_metadata(metadata: Path, destination: Path, key: str) -> ObjectStat:
    if not metadata.is_file() or not destination.is_file():
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        value = json.loads(metadata.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("object metadata must be an object")  # noqa: TRY004
        if value.get("key") != key:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        digest = str(value["digest"])
        size = int(value["size"])
    except (OSError, TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    _validate_digest(digest)
    if size < 0:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return ObjectStat(key, digest, size)


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _validate_store_id(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-" for character in value
        )
    ):
        raise ValueError("object store id is invalid")


def _validate_key(value: str) -> None:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("object key is invalid")


def _validate_digest(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("object digest is invalid")


def _validate_put(key: str, size: int, digest: str) -> None:
    _validate_key(key)
    _validate_digest(digest)
    if not isinstance(size, int) or size < 0:
        raise ValueError("object size is invalid")


__all__ = [
    "FilesystemObjectStore",
    "InMemoryObjectStore",
    "ObjectRef",
    "ObjectStat",
    "ObjectStore",
    "ObjectStoreInspection",
    "ObjectStoreMaintenance",
    "SqlObjectStore",
    "TransientObjectStore",
    "build_object_sql_metadata",
    "read_object",
    "runtime_object_key",
]
