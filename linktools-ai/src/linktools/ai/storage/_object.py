#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable, content-addressed ObjectStore implementations."""

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from linktools.core import environ

from ..errors import AIError, ErrorCode
from ._database import (
    create_sql_storage_context,
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
        temporary_root = self._root / ".tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix="object-", dir=temporary_root)
        try:
            size, digest = await _spool_file(chunks, descriptor, name, expected_size)
            if digest != expected_digest or size != expected_size:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            destination, metadata = self._paths(key)
            async with FilesystemMutationLock(self._root / "object.lock"):
                if destination.exists() or metadata.exists():
                    current = await asyncio_to_thread(
                        lambda: _read_filesystem_metadata(metadata, destination, key)
                    )
                    if current.digest != expected_digest or current.size != expected_size:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                    _logger.debug("filesystem object duplicate accepted by metadata: key=%s", key)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(name, destination)
                    await asyncio_to_thread(
                        lambda: write_json_atomic(
                            metadata,
                            {"key": key, "digest": expected_digest, "size": expected_size},
                            fsync=True,
                        )
                    )
                    sync_directory(destination.parent)
                    name = ""
            return ObjectStat(key, expected_digest, expected_size)
        finally:
            if name:
                Path(name).unlink(missing_ok=True)

    async def stat(self, key: str) -> ObjectStat | None:
        _validate_key(key)
        destination, metadata = self._paths(key)
        if not destination.exists() and not metadata.exists():
            return None
        return await asyncio_to_thread(lambda: _read_filesystem_metadata(metadata, destination, key))

    async def validate_integrity(self) -> None:
        expected_files: set[Path] = set()
        for metadata in self._root.glob("objects/*/*.json"):
            try:
                value = json.loads(await asyncio_to_thread(lambda: metadata.read_text(encoding="utf-8")))
                key = value["key"]
                expected_digest = value["digest"]
                expected_size = int(value["size"])
                _validate_key(str(key))
                destination, expected_metadata = self._paths(str(key))
                if metadata != expected_metadata:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                stat = await self.stat(str(key))
            except (OSError, TypeError, ValueError, KeyError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if stat is None or stat.digest != expected_digest or stat.size != expected_size:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            actual_size, actual_digest = await asyncio_to_thread(lambda: _hash_file(destination))
            if actual_size != expected_size or actual_digest != expected_digest:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            expected_files.update({metadata, destination})
        actual_files = {path for path in (self._root / "objects").rglob("*") if path.is_file()}
        if actual_files != expected_files:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _open(self, key: str) -> AsyncIterator[bytes]:
        _validate_key(key)
        destination, metadata = self._paths(key)
        expected = await asyncio_to_thread(
            lambda: _read_filesystem_metadata(metadata, destination, key)
        )
        digest = hashlib.sha256()
        size = 0
        with destination.open("rb") as handle:
            while chunk := handle.read(_CHUNK_SIZE):
                digest.update(chunk)
                size += len(chunk)
                yield chunk
        if expected.size != size or expected.digest != digest.hexdigest():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def open(self, key: str) -> AsyncIterator[bytes]:
        return self._open(key)


class SqlObjectStore:
    def __init__(self, engine: "AsyncEngine", *, store_id: str = "builtin") -> None:
        from sqlalchemy.ext.asyncio import AsyncEngine

        _validate_store_id(store_id)
        if not isinstance(engine, AsyncEngine):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        dialect_for_name(engine.dialect.name)
        self._store_id = store_id
        self._context = create_sql_storage_context(engine)
        self._metadata = build_object_sql_metadata()

    @classmethod
    def from_context(cls, context: "SqlStorageContext", *, store_id: str = "builtin") -> "SqlObjectStore":
        store = cls(context.engine, store_id=store_id)
        store._context = context
        return store

    @property
    def store_id(self) -> str:
        return self._store_id

    async def put(
        self, key: str, chunks: AsyncIterator[bytes], *, expected_size: int, expected_digest: str
    ) -> ObjectStat:
        from sqlalchemy.exc import IntegrityError

        _validate_put(key, expected_size, expected_digest)
        temporary_root = Path(tempfile.mkdtemp(prefix="linktools-object-"))
        temporary = temporary_root / "payload"
        try:
            size, digest = await _spool_file(chunks, -1, str(temporary), expected_size)
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
            shutil.rmtree(temporary_root, ignore_errors=True)

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
                rows = []
                with path.open("rb") as handle:
                    index = 0
                    while chunk := handle.read(_CHUNK_SIZE):
                        rows.append({"key_digest": key_digest.hex(), "chunk_index": index, "content": chunk})
                        index += 1
                        if len(rows) == 64:
                            await session.execute(insert(chunks), rows)
                            rows.clear()
                if rows:
                    await session.execute(insert(chunks), rows)
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
        *sql_audit_columns(
            "Database audit time of the physical immutable object row; normally equal to creation time.",
            "Database audit time when the immutable object header was created.",
        ),
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
        *sql_audit_columns(
            "Database audit time of the physical immutable chunk row; normally equal to creation time.",
            "Database audit time when the immutable chunk row was inserted.",
        ),
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


async def _spool_file(chunks: AsyncIterator[bytes], descriptor: int, name: str, expected_size: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    handle = os.fdopen(descriptor, "wb") if descriptor >= 0 else Path(name).open("wb")
    try:
        async for chunk in chunks:
            if not isinstance(chunk, bytes) or not chunk:
                raise ValueError("object chunks must be non-empty bytes")
            size += len(chunk)
            if size > expected_size:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            digest.update(chunk)
            handle.write(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    return size, digest.hexdigest()


async def asyncio_to_thread(fn):
    import asyncio

    return await asyncio.to_thread(fn)


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
            raise ValueError("object metadata must be an object")
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
    "SqlObjectStore",
    "TransientObjectStore",
    "build_object_sql_metadata",
    "read_object",
]
