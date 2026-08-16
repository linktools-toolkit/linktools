#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic immutable content-addressed object storage primitives."""

import asyncio
import hashlib
import os
import re
import tempfile
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from linktools.core import environ

from ..errors import AIError, ErrorCode
from ._database import (
    dialect_for_name,
    sql_audit_columns,
    sql_blob,
    sql_digest,
    sql_id_column,
    sql_integer_id,
    sql_query_index,
    sql_table_options,
    sql_text_key,
    sql_unique,
)
from ._files import sync_directory
from ._lock import FilesystemMutationLock

if TYPE_CHECKING:
    from sqlalchemy import MetaData
    from sqlalchemy.ext.asyncio import AsyncEngine

    from ._database import SqlStorageContext


_logger = environ.get_logger("ai.storage.object")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_STORE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_KEY_SEGMENT = re.compile(r"[A-Za-z0-9._-]+\Z")
_RESERVED_STORE_IDS = frozenset({"builtin", "memory", "transient"})
_SQL_CHUNK_SIZE = 1024 * 1024
_SQL_BATCH_KEY_LIMIT = 256


@dataclass(frozen=True, slots=True)
class _SqlLoadedObject:
    key: str
    digest: str
    size: int
    chunks: tuple[bytes, ...]


class _ObjectInsertConflict(Exception):
    def __init__(self, error: BaseException) -> None:
        super().__init__(str(error))
        self.error = error


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

    def __post_init__(self) -> None:
        _validate_key(self.key)
        _validate_digest(self.digest)
        if not isinstance(self.size, int) or self.size < 0:
            raise ValueError("object size must be non-negative")


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

    def open(self, key: str) -> AsyncIterator[bytes]: ...


class _MemoryObjectStore:
    def __init__(self, store_id: str) -> None:
        _validate_store_id(store_id)
        self._store_id = store_id
        self._objects: dict[str, bytes] = {}
        self._stats: dict[str, ObjectStat] = {}
        self._lock = asyncio.Lock()

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
        _validate_put(key, expected_size, expected_digest)
        data = await _collect(chunks, expected_size, expected_digest)
        value = ObjectStat(key, expected_digest, expected_size)
        async with self._lock:
            self._put_data(key, data, value)
        _logger.debug(
            "object stored: store=%s object_key_digest=%s size=%s",
            self.store_id,
            _key_digest(key),
            expected_size,
        )
        return value

    def _put_data(self, key: str, data: bytes, value: ObjectStat) -> None:
        current = self._objects.get(key)
        if current is not None and current != data:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._objects[key] = data
        self._stats[key] = value

    async def stat(self, key: str) -> ObjectStat | None:
        _validate_key(key)
        return self._stats.get(key)

    async def _open(self, key: str) -> AsyncIterator[bytes]:
        _validate_key(key)
        data = self._objects.get(key)
        expected = self._stats.get(key)
        if data is None or expected is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        digest = hashlib.sha256()
        size = 0
        for offset in range(0, len(data), 64 * 1024):
            chunk = data[offset : offset + 64 * 1024]
            digest.update(chunk)
            size += len(chunk)
            yield chunk
        if size != expected.size or digest.hexdigest() != expected.digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def open(self, key: str) -> AsyncIterator[bytes]:
        return self._open(key)

    def clear(self) -> None:
        self._objects.clear()
        self._stats.clear()


class InMemoryObjectStore(_MemoryObjectStore):
    def __init__(self, store_id: str = "memory") -> None:
        if store_id in _RESERVED_STORE_IDS and store_id != "memory":
            raise ValueError("reserved object store id belongs to another built-in store")
        super().__init__(store_id)


class TransientObjectStore(_MemoryObjectStore):
    def __init__(self) -> None:
        super().__init__("transient")
        self._scope_keys: dict[str, set[str]] = {}
        self._key_scopes: dict[str, set[str]] = {}

    def scoped(self, scope: str) -> ObjectStore:
        if not scope or "/" in scope:
            raise ValueError("transient object scope is invalid")
        return _ScopedTransientObjectStore(self, scope)

    async def release_scope(self, scope: str) -> None:
        if not scope or "/" in scope:
            raise ValueError("transient object scope is invalid")
        async with self._lock:
            keys = self._scope_keys.pop(scope, set())
            for key in keys:
                scopes = self._key_scopes.get(key)
                if scopes is None:
                    continue
                scopes.discard(scope)
                if not scopes:
                    self._key_scopes.pop(key, None)
                    self._objects.pop(key, None)
                    self._stats.pop(key, None)
        _logger.debug("transient object scope released: scope=%s keys=%s", scope, len(keys))

    def clear(self) -> None:
        super().clear()
        self._scope_keys.clear()
        self._key_scopes.clear()


class _ScopedTransientObjectStore:
    def __init__(self, parent: TransientObjectStore, scope: str) -> None:
        self._parent = parent
        self._scope = scope

    @property
    def store_id(self) -> str:
        return self._parent.store_id

    async def put(
        self,
        key: str,
        chunks: AsyncIterator[bytes],
        *,
        expected_size: int,
        expected_digest: str,
    ) -> ObjectStat:
        _validate_put(key, expected_size, expected_digest)
        data = await _collect(chunks, expected_size, expected_digest)
        value = ObjectStat(key, expected_digest, expected_size)
        async with self._parent._lock:
            self._parent._put_data(key, data, value)
            self._parent._scope_keys.setdefault(self._scope, set()).add(key)
            self._parent._key_scopes.setdefault(key, set()).add(self._scope)
        _logger.debug(
            "transient scoped object stored: scope=%s object_key_digest=%s size=%s",
            self._scope,
            _key_digest(key),
            expected_size,
        )
        return value

    async def stat(self, key: str) -> ObjectStat | None:
        _validate_key(key)
        return await self._parent.stat(key)

    def open(self, key: str) -> AsyncIterator[bytes]:
        return self._parent.open(key)


class FilesystemObjectStore:
    def __init__(self, root: str | Path, *, store_id: str = "builtin") -> None:
        if store_id in _RESERVED_STORE_IDS and store_id != "builtin":
            raise ValueError("reserved object store id belongs to another built-in store")
        _validate_store_id(store_id)
        self._root = Path(root).expanduser().resolve()
        self._store_id = store_id

    @property
    def store_id(self) -> str:
        return self._store_id

    def _path(self, key: str) -> Path:
        _validate_key(key)
        return self._root.joinpath(*key.split("/"))

    def _lock(self, key: str) -> FilesystemMutationLock:
        return FilesystemMutationLock(self._root / ".locks" / f"{_key_digest(key)}.lock")

    async def put(
        self,
        key: str,
        chunks: AsyncIterator[bytes],
        *,
        expected_size: int,
        expected_digest: str,
    ) -> ObjectStat:
        _validate_put(key, expected_size, expected_digest)
        data = await _collect(chunks, expected_size, expected_digest)
        path = self._path(key)
        temporary_root = self._root / ".tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix="object-", dir=temporary_root)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            async with self._lock(key):
                if path.is_file():
                    existing = await asyncio.to_thread(path.read_bytes)
                    if existing != data:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(temporary, path)
                    sync_directory(path.parent)
                    temporary = ""
        except OSError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        finally:
            if temporary:
                Path(temporary).unlink(missing_ok=True)
        _logger.debug(
            "filesystem object stored: store=%s object_key_digest=%s size=%s",
            self.store_id,
            _key_digest(key),
            expected_size,
        )
        return ObjectStat(key, expected_digest, expected_size)

    async def stat(self, key: str) -> ObjectStat | None:
        path = self._path(key)
        try:
            if not path.is_file():
                return None
            data = await asyncio.to_thread(path.read_bytes)
        except OSError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        return ObjectStat(key, hashlib.sha256(data).hexdigest(), len(data))

    async def _open(self, key: str) -> AsyncIterator[bytes]:
        path = self._path(key)
        if not path.is_file():
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        digest = hashlib.sha256()
        size = 0
        try:
            with path.open("rb") as stream:
                while chunk := stream.read(64 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                    yield chunk
        except OSError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        stat = await self.stat(key)
        if stat is None or stat.size != size or stat.digest != digest.hexdigest():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def open(self, key: str) -> AsyncIterator[bytes]:
        return self._open(key)


class SqlObjectStore:
    """Immutable SQL object store backed by the two generic storage tables."""

    def __init__(self, engine: "AsyncEngine", *, store_id: str = "builtin") -> None:
        from sqlalchemy.ext.asyncio import AsyncEngine

        dialect_for_name(engine.dialect.name)
        if not isinstance(engine, AsyncEngine):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if store_id in _RESERVED_STORE_IDS and store_id != "builtin":
            raise ValueError("reserved object store id belongs to another built-in store")
        _validate_store_id(store_id)
        self._engine = engine
        self._store_id = store_id
        self._metadata = build_object_sql_metadata()
        self._lock = asyncio.Lock()
        self._context: "SqlStorageContext | None" = None

    @classmethod
    def from_context(cls, context: "SqlStorageContext", *, store_id: str = "builtin") -> "SqlObjectStore":
        store = cls(context.engine, store_id=store_id)
        store._context = context
        return store

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
        _validate_put(key, expected_size, expected_digest)
        data = await _collect(chunks, expected_size, expected_digest)
        objects = self._metadata.tables["ai_storage_objects"]
        chunks_table = self._metadata.tables["ai_storage_object_chunks"]
        from sqlalchemy import insert, select
        from sqlalchemy.exc import IntegrityError

        async def _write() -> ObjectStat:
            object_key_digest = _key_digest(key)
            try:
                async with self._begin() as connection:
                    try:
                        result = await connection.execute(
                            insert(objects).values(
                                object_key=key,
                                object_key_digest=object_key_digest,
                                digest=expected_digest,
                                size=expected_size,
                            )
                        )
                    except IntegrityError as error:
                        raise _ObjectInsertConflict(error) from error
                    object_id = int(result.inserted_primary_key[0])
                    rows = [
                        {
                            "object_id": object_id,
                            "chunk_index": index,
                            "content": data[offset : offset + _SQL_CHUNK_SIZE],
                        }
                        for index, offset in enumerate(range(0, len(data), _SQL_CHUNK_SIZE))
                    ]
                    if rows:
                        try:
                            await connection.execute(insert(chunks_table), rows)
                        except IntegrityError as error:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            except _ObjectInsertConflict as conflict:
                dialect = dialect_for_name(self._engine.dialect.name)
                if dialect.classify_integrity_error(conflict.error).value != "unique_conflict":
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from conflict.error
                async with self._connect() as connection:
                    rows = (
                        await connection.execute(
                            select(
                                objects.c.object_key,
                                objects.c.object_key_digest,
                                objects.c.digest,
                                objects.c.size,
                            ).where(objects.c.object_key_digest == object_key_digest)
                        )
                    ).mappings().all()
                if not rows:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                if len(rows) != 1:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                row = rows[0]
                stored_key = str(row["object_key"])
                stored_key_digest = str(row["object_key_digest"])
                if _key_digest(stored_key) != stored_key_digest:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if stored_key != key:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                existing = _object_stat(
                    stored_key,
                    str(row["digest"]),
                    int(row["size"]),
                )
                if existing.digest == expected_digest and existing.size == expected_size:
                    return existing
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return ObjectStat(key, expected_digest, expected_size)

        if self._engine.dialect.name == "sqlite":
            async with self._lock:
                result = await _write()
        else:
            result = await _write()
        _logger.debug(
            "SQL object stored: store=%s object_key_digest=%s size=%s",
            self.store_id,
            _key_digest(key),
            expected_size,
        )
        return result

    async def stat(self, key: str) -> ObjectStat | None:
        rows = await self.stat_many((key,))
        if key not in rows:
            return None
        return rows[key]

    async def stat_many(self, keys: Sequence[str]) -> dict[str, ObjectStat]:
        requested = _unique_object_keys(keys)
        if not requested:
            return {}
        loaded = await self._load_headers_many(requested)
        return {
            key: _object_stat(value.key, value.digest, value.size)
            for key, value in loaded.items()
        }

    async def read_many(self, refs: Sequence[ObjectRef]) -> dict[str, bytes]:
        requested: dict[str, ObjectRef] = {}
        for ref in refs:
            if ref.store_id != self.store_id:
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            current = requested.get(ref.key)
            if current is not None and (
                current.digest != ref.digest or current.size != ref.size
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            requested[ref.key] = ref
        if not requested:
            return {}
        loaded = await self._load_many(tuple(requested))
        if len(loaded) != len(requested):
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        result: dict[str, bytes] = {}
        for key, ref in requested.items():
            value = loaded[key]
            if value.digest != ref.digest or value.size != ref.size:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result[key] = b"".join(value.chunks)
        return result

    async def _load_many(self, keys: Sequence[str]) -> dict[str, _SqlLoadedObject]:
        requested = _unique_object_keys(keys)
        if not requested:
            return {}
        digest_keys = _object_digest_keys(requested)
        objects = self._metadata.tables["ai_storage_objects"]
        chunks_table = self._metadata.tables["ai_storage_object_chunks"]
        from sqlalchemy import select

        loaded: dict[str, _SqlLoadedObject] = {}
        async with self._connect() as connection:
            for batch in _batches(digest_keys):
                statement = (
                    select(
                        objects.c.id.label("object_id"),
                        objects.c.object_key,
                        objects.c.object_key_digest,
                        objects.c.digest,
                        objects.c.size,
                        chunks_table.c.chunk_index,
                        chunks_table.c.content,
                    )
                    .select_from(
                        objects.outerjoin(
                            chunks_table,
                            chunks_table.c.object_id == objects.c.id,
                        )
                    )
                    .where(objects.c.object_key_digest.in_(batch))
                    .order_by(objects.c.id, chunks_table.c.chunk_index)
                )
                rows = (await connection.execute(statement)).mappings().all()
                groups: dict[int, list[object]] = {}
                for row in rows:
                    object_id = int(row["object_id"])
                    groups.setdefault(object_id, []).append(row)
                for object_rows in groups.values():
                    row = object_rows[0]
                    stored_key = str(row["object_key"])
                    stored_key_digest = str(row["object_key_digest"])
                    stored_digest = str(row["digest"])
                    if _key_digest(stored_key) != stored_key_digest:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if stored_key not in requested:
                        if stored_key_digest in digest_keys:
                            raise AIError(ErrorCode.STORAGE_CONFLICT)
                        continue
                    if stored_key_digest != _key_digest(stored_key):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if stored_key in loaded:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    try:
                        stored_size = int(row["size"])
                    except (TypeError, ValueError) as error:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                    stat = _object_stat(stored_key, stored_digest, stored_size)
                    chunks = _validate_sql_chunks(
                        object_rows,
                        stat.digest,
                        stat.size,
                    )
                    loaded[stored_key] = _SqlLoadedObject(
                        stored_key,
                        stat.digest,
                        stat.size,
                        chunks,
                    )
        return loaded

    async def _load_headers_many(self, keys: Sequence[str]) -> dict[str, _SqlLoadedObject]:
        requested = _unique_object_keys(keys)
        if not requested:
            return {}
        digest_keys = _object_digest_keys(requested)
        objects = self._metadata.tables["ai_storage_objects"]
        from sqlalchemy import select

        loaded: dict[str, _SqlLoadedObject] = {}
        async with self._connect() as connection:
            for batch in _batches(digest_keys):
                rows = (
                    await connection.execute(
                        select(
                            objects.c.object_key,
                            objects.c.object_key_digest,
                            objects.c.digest,
                            objects.c.size,
                        )
                        .where(objects.c.object_key_digest.in_(batch))
                        .order_by(objects.c.id)
                    )
                ).mappings().all()
                for row in rows:
                    stored_key = str(row["object_key"])
                    stored_key_digest = str(row["object_key_digest"])
                    if _key_digest(stored_key) != stored_key_digest:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if stored_key not in requested:
                        if stored_key_digest in digest_keys:
                            raise AIError(ErrorCode.STORAGE_CONFLICT)
                        continue
                    if stored_key in loaded:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    try:
                        stat = _object_stat(
                            stored_key,
                            str(row["digest"]),
                            int(row["size"]),
                        )
                    except (TypeError, ValueError) as error:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                    loaded[stored_key] = _SqlLoadedObject(
                        stored_key,
                        stat.digest,
                        stat.size,
                        (),
                    )
        return loaded

    async def _open(self, key: str) -> AsyncIterator[bytes]:
        loaded = await self._load_many((key,))
        value = loaded.get(key)
        if value is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        for chunk in value.chunks:
            yield chunk

    def open(self, key: str) -> AsyncIterator[bytes]:
        return self._open(key)

    @asynccontextmanager
    async def _begin(self) -> AsyncIterator[object]:
        if self._context is None:
            async with self._engine.begin() as connection:
                yield connection
        else:
            async with self._context.sessions.begin() as session:
                yield session

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[object]:
        if self._context is None:
            async with self._engine.connect() as connection:
                yield connection
        else:
            async with self._context.sessions() as session:
                yield session


def build_object_sql_metadata(metadata: "MetaData | None" = None) -> "MetaData":
    from sqlalchemy import Column, MetaData, Table

    if metadata is None:
        metadata = MetaData()
    _objects = Table(
        "ai_storage_objects",
        metadata,
        sql_id_column(),
        Column("object_key", sql_text_key(255), nullable=False, comment="Object key"),
        Column(
            "object_key_digest",
            sql_digest(),
            nullable=False,
            comment="Object key SHA-256 digest",
        ),
        Column("digest", sql_digest(), nullable=False, comment="Object content SHA-256 digest"),
        Column("size", sql_integer_id(), nullable=False, comment="Size in bytes"),
        *sql_audit_columns(),
        comment="ObjectStore objects",
        **sql_table_options(),
    )
    sql_unique(_objects, "object_key_digest")
    sql_query_index(_objects, "updated_at")
    sql_query_index(_objects, "created_at")
    chunks = Table(
        "ai_storage_object_chunks",
        metadata,
        sql_id_column(),
        Column("object_id", sql_integer_id(), nullable=False, comment="Object row identifier"),
        Column("chunk_index", sql_integer_id(), nullable=False, comment="Chunk index"),
        Column("content", sql_blob(), nullable=False, comment="Chunk content"),
        *sql_audit_columns(),
        comment="ObjectStore object chunks",
        **sql_table_options(),
    )
    sql_unique(chunks, "object_id", "chunk_index")
    sql_query_index(chunks, "updated_at")
    sql_query_index(chunks, "created_at")
    return metadata


async def read_object(store: ObjectStore, key: str, *, expected_digest: str, expected_size: int) -> bytes:
    _validate_put(key, expected_size, expected_digest)
    data = bytearray()
    digest = hashlib.sha256()
    async for chunk in store.open(key):
        if not isinstance(chunk, bytes):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        data.extend(chunk)
        digest.update(chunk)
    if len(data) != expected_size or digest.hexdigest() != expected_digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return bytes(data)


async def _collect(chunks: AsyncIterator[bytes], expected_size: int, expected_digest: str) -> bytes:
    data = bytearray()
    digest = hashlib.sha256()
    async for chunk in chunks:
        if not isinstance(chunk, bytes) or not chunk:
            raise ValueError("object chunks must be non-empty bytes")
        data.extend(chunk)
        if len(data) > expected_size:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        digest.update(chunk)
    if len(data) != expected_size or digest.hexdigest() != expected_digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return bytes(data)


def _unique_object_keys(keys: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for key in keys:
        _validate_key(key)
        if key not in seen:
            seen.add(key)
            result.append(key)
    return tuple(result)


def _object_digest_keys(keys: Sequence[str]) -> tuple[str, ...]:
    digests: dict[str, str] = {}
    for key in keys:
        digest = _key_digest(key)
        previous = digests.get(digest)
        if previous is not None and previous != key:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        digests[digest] = key
    return tuple(digests)


def _batches(values: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(values[offset : offset + _SQL_BATCH_KEY_LIMIT])
        for offset in range(0, len(values), _SQL_BATCH_KEY_LIMIT)
    )


def _validate_sql_chunks(
    rows: Sequence[object],
    expected_digest: str,
    expected_size: int,
) -> tuple[bytes, ...]:
    digest = hashlib.sha256()
    size = 0
    chunks: list[bytes] = []
    sentinel = False
    for row in rows:
        chunk_index = row["chunk_index"]
        content = row["content"]
        if chunk_index is None or content is None:
            if chunk_index is not None or content is not None or sentinel or len(rows) != 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            sentinel = True
            continue
        if sentinel:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            index = int(chunk_index)
            value = bytes(content)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if index != len(chunks):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        chunks.append(value)
        digest.update(value)
        size += len(value)
    expected_count = 0 if expected_size == 0 else (expected_size + _SQL_CHUNK_SIZE - 1) // _SQL_CHUNK_SIZE
    if len(chunks) != expected_count or size != expected_size:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if digest.hexdigest() != expected_digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return tuple(chunks)


def _object_stat(key: str, digest: str, size: int) -> ObjectStat:
    try:
        return ObjectStat(key, digest, size)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _validate_store_id(value: str) -> None:
    if not isinstance(value, str) or _STORE_ID.fullmatch(value) is None:
        raise ValueError("invalid object store id")


def _validate_key(value: str) -> None:
    if not isinstance(value, str) or not 0 < len(value) <= 255 or "\\" in value or "\x00" in value or value.startswith("/"):
        raise ValueError("invalid object key")
    parts = value.split("/")
    if any(not part or part in {".", ".."} or _KEY_SEGMENT.fullmatch(part) is None for part in parts):
        raise ValueError("invalid object key")


def _validate_digest(value: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _validate_put(key: str, expected_size: int, expected_digest: str) -> None:
    _validate_key(key)
    if not isinstance(expected_size, int) or expected_size < 0:
        raise ValueError("object size must be non-negative")
    _validate_digest(expected_digest)


def _key_digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


__all__ = [
    "FilesystemObjectStore",
    "InMemoryObjectStore",
    "ObjectRef",
    "ObjectStat",
    "ObjectStore",
    "SqlObjectStore",
    "TransientObjectStore",
    "build_object_sql_metadata",
    "read_object",
]
