#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL ObjectStore implementation and schema metadata."""

import asyncio
import hashlib
import shutil
import tempfile
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from ._object import (
    ObjectStat,
    _CHUNK_SIZE,
    _finish_owned_task,
    _key_digest,
    _spool_file,
    _track_object_task,
    _validate_key,
    _validate_put,
    _validate_store_id,
)

if TYPE_CHECKING:
    from sqlalchemy import MetaData
    from sqlalchemy.ext.asyncio import AsyncEngine

    from ._database import SqlStorageContext


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
    def from_context(
        cls,
        context: "SqlStorageContext",
        *,
        store_id: str = "builtin",
    ) -> "SqlObjectStore":
        return cls(context.engine, store_id=store_id, context=context)

    @property
    def store_id(self) -> str:
        return self._store_id

    def local_paths(self) -> tuple[Path, ...]:
        return ()

    @property
    def pending_background_tasks(self) -> tuple[asyncio.Task[Any], ...]:
        return tuple(task for task in self._background_tasks if not task.done())

    async def put(
        self,
        key: str,
        chunks: AsyncIterator[bytes],
        *,
        expected_size: int,
        expected_digest: str,
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
        return await _finish_owned_task(task)

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
                        select(table).where(
                            table.c.key_digest == _key_digest(self.store_id, key).hex()
                        )
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
                (
                    await session.execute(
                        select(
                            chunks.c.key_digest,
                            chunks.c.chunk_index,
                            chunks.c.content,
                        )
                    )
                )
                .mappings()
                .all()
            )
            all_object_keys = {
                str(value)
                for value in await session.scalars(select(objects.c.key_digest))
            }
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
            if (
                _key_digest(
                    str(header["store_id"]),
                    str(header["object_key"]),
                ).hex()
                != key
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            values = sorted(
                grouped.get(key, ()),
                key=lambda row: int(row["chunk_index"]),
            )
            if [int(row["chunk_index"]) for row in values] != list(
                range(len(values))
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            digest = hashlib.sha256()
            size = 0
            for row in values:
                content = bytes(row["content"])
                digest.update(content)
                size += len(content)
            if (
                size != int(header["size"])
                or digest.hexdigest() != str(header["content_digest"])
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _list_objects(self) -> AsyncIterator[ObjectStat]:
        from sqlalchemy import select

        table = self._metadata.tables["ai_objects"]
        session = self._context.sessions()
        try:
            rows = (
                (
                    await session.execute(
                        select(table).where(table.c.store_id == self.store_id)
                    )
                )
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
        task = asyncio.create_task(
            self._delete_owned(key, expected_digest=expected_digest),
            name=f"sql-object-delete-{_key_digest(self.store_id, key).hex()[:12]}",
        )
        _track_object_task(self._background_tasks, task, "SQL object delete")
        return await _finish_owned_task(task)

    async def _delete_owned(self, key: str, *, expected_digest: str) -> bool:
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
                    self._metadata.tables["ai_object_chunks"].c.key_digest
                    == key_digest
                )
            )
            result = await session.execute(
                delete(table).where(
                    table.c.key_digest == key_digest,
                    table.c.store_id == self.store_id,
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
                        select(table).where(
                            table.c.key_digest == _key_digest(self.store_id, key).hex()
                        )
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
                select(chunks)
                .where(chunks.c.key_digest == header["key_digest"])
                .order_by(chunks.c.chunk_index)
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
            if (
                size != int(header["size"])
                or digest.hexdigest() != str(header["content_digest"])
            ):
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
            "object_key",
            Text,
            nullable=False,
            comment="Original opaque object key exposed by the ObjectStore API.",
        ),
        Column(
            "content_digest",
            digest,
            nullable=False,
            comment="SHA-256 digest of the immutable object bytes.",
        ),
        Column(
            "size",
            BigInteger,
            nullable=False,
            comment="Exact immutable object size in bytes.",
        ),
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


__all__ = ["SqlObjectStore", "build_object_sql_metadata"]
