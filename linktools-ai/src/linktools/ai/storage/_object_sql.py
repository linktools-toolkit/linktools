#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL ObjectStore implementation and schema metadata."""

import asyncio
import hashlib
import shutil
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

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
    _OBJECT_KEY_MAX_BYTES,
    _STORE_ID_MAX_LENGTH,
    _finish_owned_task,
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
            name="sql-object-put",
        )
        _track_object_task(self._background_tasks, task)
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

        async def execute(session) -> None:
            result = await self._context.run_statement(session.execute(
                insert(table).values(
                    store_id=self.store_id,
                    object_key=key.encode("utf-8"),
                    object_locator=_object_locator(key),
                    content_digest=digest,
                    size=size,
                )
            ))
            object_id = result.inserted_primary_key[0]
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
                    row["object_id"] = object_id
                await self._context.run_statement(session.execute(insert(chunks), rows))
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
                    await self._context.run_statement(session.execute(
                        select(table).where(
                            table.c.store_id == self.store_id,
                            table.c.object_locator == _object_locator(key)
                        )
                    ))
                )
                .mappings()
                .one_or_none()
            )
        finally:
            await session.close()
        if row is None:
            return None
        if bytes(row["object_key"]) != key.encode("utf-8"):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return ObjectStat(key, str(row["content_digest"]), int(row["size"]))

    async def validate_integrity(self) -> None:
        from sqlalchemy import select

        objects = self._metadata.tables["ai_objects"]
        chunks = self._metadata.tables["ai_object_chunks"]

        session = self._context.sessions()
        try:
            orphan = (
                await self._context.run_statement(session.execute(
                    select(chunks.c.object_id)
                    .outerjoin(
                        objects,
                        objects.c.id == chunks.c.object_id,
                    )
                    .where(objects.c.id.is_(None))
                    .limit(1)
                ))
            ).scalar_one_or_none()
        finally:
            await session.close()
        if orphan is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        cursor = 0
        while True:
            session = self._context.sessions()
            try:
                headers = (
                    (
                        await self._context.run_statement(session.execute(
                            select(
                                objects.c.id,
                                objects.c.object_key,
                                objects.c.object_locator,
                            )
                            .where(
                                objects.c.store_id == self.store_id,
                                objects.c.id > cursor,
                            )
                            .order_by(objects.c.id)
                            .limit(128)
                        ))
                    )
                    .mappings()
                    .all()
                )
            finally:
                await session.close()
            if not headers:
                return

            for header in headers:
                try:
                    object_key = bytes(header["object_key"]).decode("utf-8")
                    _validate_key(object_key)
                    if header["object_locator"] != _object_locator(object_key):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                except (UnicodeDecodeError, TypeError, ValueError) as error:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                stream = self._stream_open(object_key)
                try:
                    async for _chunk in stream:
                        pass
                finally:
                    await stream.aclose()
            cursor = int(headers[-1]["id"])


    async def _list_objects(self) -> AsyncIterator[ObjectStat]:
        from sqlalchemy import select

        table = self._metadata.tables["ai_objects"]
        session = self._context.sessions()
        try:
            rows = (
                (
                    await self._context.run_statement(session.execute(
                        select(table).where(table.c.store_id == self.store_id)
                    ))
                )
                .mappings()
                .all()
            )
        finally:
            await session.close()
        for row in rows:
            if row["object_locator"] != _object_locator(bytes(row["object_key"]).decode("utf-8")):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            yield ObjectStat(
                bytes(row["object_key"]).decode("utf-8"),
                str(row["content_digest"]),
                int(row["size"]),
            )

    def list_objects(self) -> AsyncIterator[ObjectStat]:
        return self._list_objects()

    async def delete_object(self, key: str, *, expected_digest: str) -> bool:
        _validate_key(key)
        task = asyncio.create_task(
            self._delete_owned(key, expected_digest=expected_digest),
            name="sql-object-delete",
        )
        _track_object_task(self._background_tasks, task)
        return await _finish_owned_task(task)

    async def _delete_owned(self, key: str, *, expected_digest: str) -> bool:
        from sqlalchemy import delete, select

        async def execute(session) -> bool:
            table = self._metadata.tables["ai_objects"]
            current = (
                await self._context.run_statement(session.execute(
                    select(table.c.id, table.c.object_key, table.c.content_digest).where(
                        table.c.object_locator == _object_locator(key),
                        table.c.store_id == self.store_id,
                    )
                ))
            ).one_or_none()
            if current is None:
                return False
            if bytes(current.object_key) != key.encode("utf-8"):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if str(current.content_digest) != expected_digest:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await self._context.run_statement(session.execute(
                delete(self._metadata.tables["ai_object_chunks"]).where(
                    self._metadata.tables["ai_object_chunks"].c.object_id
                    == current.id
                )
            ))
            result = await self._context.run_statement(session.execute(
                delete(table).where(
                    table.c.id == current.id,
                    table.c.object_locator == _object_locator(key),
                    table.c.store_id == self.store_id,
                )
            ))
            return result.rowcount == 1

        return await self._context.run_mutation(execute, domain="storage.object")

    async def _open(self, key: str) -> AsyncIterator[bytes]:
        _validate_key(key)
        if self._context.dialect.name != "sqlite":
            stream = self._stream_open(key)
            try:
                async for value in stream:
                    yield value
            finally:
                await stream.aclose()
            return

        temporary = await asyncio.to_thread(tempfile.TemporaryFile)
        try:
            await self._stage_open(key, temporary)
            await asyncio.to_thread(temporary.seek, 0)
            while True:
                value = await asyncio.to_thread(temporary.read, _CHUNK_SIZE)
                if not value:
                    break
                yield value
        finally:
            await asyncio.to_thread(temporary.close)

    async def _stream_open(self, key: str) -> AsyncIterator[bytes]:
        from sqlalchemy import case, or_, select

        table = self._metadata.tables["ai_objects"]
        chunks = self._metadata.tables["ai_object_chunks"]
        session = self._context.sessions()
        result = None
        try:
            first = or_(chunks.c.chunk_index == 0, chunks.c.chunk_index.is_(None))
            # Transmit header metadata once rather than with every content chunk.
            header_columns = tuple(
                case((first, column)).label(column.name)
                for column in (table.c.object_key, table.c.content_digest, table.c.size)
            )

            async def start_stream() -> None:
                nonlocal result
                result = await session.stream(
                    select(*header_columns, chunks.c.chunk_index, chunks.c.content)
                    .select_from(table.outerjoin(chunks, chunks.c.object_id == table.c.id))
                    .where(
                        table.c.store_id == self.store_id,
                        table.c.object_locator == _object_locator(key),
                    )
                    .order_by(chunks.c.chunk_index)
                )

            await self._context.run_statement(start_stream())
            digest = hashlib.sha256()
            size = 0
            expected_index = 0
            header = None
            rows = result.mappings()
            while True:
                try:
                    row = await self._context.run_statement(rows.__anext__())
                except StopAsyncIteration:
                    break
                if row["chunk_index"] is not None and int(row["chunk_index"]) != expected_index:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if header is None:
                    if bytes(row["object_key"]) != key.encode("utf-8"):
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                    header = ObjectStat(key, str(row["content_digest"]), int(row["size"]))
                if row["chunk_index"] is None:
                    continue
                value = bytes(row["content"])
                expected_index += 1
                digest.update(value)
                size += len(value)
                yield value
            if header is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            if (
                size != header.size
                or digest.hexdigest() != header.digest
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        finally:
            try:
                if result is not None:
                    await self._context.run_statement(result.close())
            finally:
                await session.close()

    async def _stage_open(self, key: str, handle: BinaryIO) -> None:
        stream = self._stream_open(key)
        try:
            async for value in stream:
                await asyncio.to_thread(handle.write, value)
        finally:
            await stream.aclose()

    def open(self, key: str) -> AsyncIterator[bytes]:
        return self._open(key)


def build_object_sql_metadata(metadata: "MetaData | None" = None) -> "MetaData":
    from sqlalchemy import BigInteger, Column, LargeBinary, MetaData, Table
    from sqlalchemy.dialects.mysql import VARBINARY

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
            "store_id",
            sql_text_key(_STORE_ID_MAX_LENGTH),
            nullable=False,
            comment="Logical ObjectStore identifier that namespaces object keys.",
        ),
        Column(
            "object_key",
            LargeBinary(_OBJECT_KEY_MAX_BYTES).with_variant(VARBINARY(_OBJECT_KEY_MAX_BYTES), "mysql"),
            nullable=False,
            comment="Exact UTF-8 bytes of the opaque ObjectStore key.",
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
        Column(
            "object_locator",
            sql_sha256(),
            nullable=False,
            comment="Single SHA-256 lookup projection of the complete object_key; raw identity must also match.",
        ),
        *sql_audit_columns(),
        comment="Immutable ObjectStore headers containing canonical object identity, content digest, and size.",
        **sql_table_options(),
    )
    sql_unique(objects, "store_id", "object_locator")
    sql_audit_indexes(objects)
    chunks = Table(
        "ai_object_chunks",
        metadata,
        sql_id_column(),
        Column(
            "object_id",
            BigInteger,
            nullable=False,
            comment="Surrogate identifier of the immutable ObjectStore object owning this chunk.",
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
    sql_unique(chunks, "object_id", "chunk_index")
    sql_audit_indexes(chunks)
    return metadata


def _object_locator(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


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
