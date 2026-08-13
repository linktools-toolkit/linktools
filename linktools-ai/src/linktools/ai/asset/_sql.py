#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL Asset backend and the Asset-owned SQL metadata builder."""

import asyncio
import hashlib
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from linktools.core import environ

from ..core import validate_asset_namespace
from ..errors import AIError, ErrorCode
from ..storage import (
    MetadataChange,
    MetadataLoad,
    MetadataLoadMode,
    ObjectStore,
    SqlObjectStore,
    StorageBatchResult,
    StorageChange,
    StorageDeleteResult,
    StorageEntryRevision,
    StorageEntryStatus,
    StoragePutResult,
    StorageResetResult,
    StorageRevision,
    VersionSummary,
    create_sql_context,
    provision_sql,
    read_object,
    sql_digest,
    sql_integer_id,
    sql_table_options,
    sql_text_key,
)
from ._domain import AssetInfo, AssetKey, AssetRoot
from ._object import AssetObjectKeyFactory

if TYPE_CHECKING:
    from sqlalchemy import MetaData
    from sqlalchemy.ext.asyncio import AsyncEngine


_logger = environ.get_logger("ai.asset.sql")
_EMPTY_DIGEST = hashlib.sha256(b"").hexdigest()


def build_asset_sql_metadata(*, object_store: ObjectStore | None = None) -> "MetaData":
    """Build only Asset tables, adding generic object tables for the built-in store."""

    from sqlalchemy import (
        Column,
        DateTime,
        Index,
        MetaData,
        String,
        Table,
        UniqueConstraint,
        func,
    )

    metadata = MetaData()
    digest = sql_digest()
    integer_id = sql_integer_id()
    _revision = Table(
        "asset_revision",
        metadata,
        Column("id", integer_id, primary_key=True, autoincrement=True),
        Column("namespace_key", digest, nullable=False),
        Column("store_revision", integer_id, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()),
        Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()),
        UniqueConstraint("namespace_key", name="uk_asset_revision_namespace_key"),
        **sql_table_options(),
    )
    columns = (
        Column("id", integer_id, primary_key=True, autoincrement=True),
        Column("namespace_key", digest, nullable=False),
        Column("asset_key_hash", digest, nullable=False),
        Column("asset_kind", sql_text_key(128), nullable=False),
        Column("asset_id", sql_text_key(512), nullable=False),
        Column("entry_revision", integer_id, nullable=False),
        Column("store_revision", integer_id, nullable=False),
        Column("etag", digest, nullable=False),
        Column("size", integer_id, nullable=False),
        Column("status", String(32), nullable=False),
        Column("object_store_id", sql_text_key(128), nullable=True),
        Column("object_key", sql_text_key(1024), nullable=True),
        Column("object_digest", digest, nullable=True),
        Column("object_size", integer_id, nullable=True),
        Column("modified_at", DateTime(timezone=True), nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()),
        Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()),
    )
    _entry = Table(
        "asset_entries",
        metadata,
        *columns,
        UniqueConstraint("namespace_key", "asset_key_hash", name="uk_asset_entries_identity"),
        Index("ix_asset_entries_revision", "namespace_key", "store_revision"),
        **sql_table_options(),
    )
    history_columns = [column.copy() for column in columns]
    _change = Table(
        "asset_changes",
        metadata,
        *history_columns,
        UniqueConstraint("namespace_key", "asset_key_hash", "entry_revision", name="uk_asset_changes_revision"),
        Index("ix_asset_changes_revision", "namespace_key", "store_revision"),
        Index("ix_asset_changes_history", "namespace_key", "asset_key_hash", "entry_revision"),
        **sql_table_options(),
    )
    if object_store is None:
        from ..storage import build_object_sql_metadata

        object_metadata = build_object_sql_metadata()
        for table in object_metadata.tables.values():
            table.to_metadata(metadata)
    return metadata


class SqlAssetBackend:
    """Direct SQL Asset backend with CAS and immutable change history."""

    def __init__(
        self,
        engine: "AsyncEngine",
        *,
        namespace: str,
        object_store: ObjectStore | None = None,
    ) -> None:
        from sqlalchemy.ext.asyncio import AsyncEngine

        if not isinstance(engine, AsyncEngine):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        try:
            validate_asset_namespace(namespace)
        except AIError as error:
            raise ValueError("SQL asset namespace is invalid") from error
        digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
        self._root = AssetRoot(f"sql:{digest[:16]}", "sql", namespace, digest)
        self._engine = engine
        self._namespace = namespace
        self._namespace_key = digest
        self._object_keys = AssetObjectKeyFactory(namespace)
        self._object_store = object_store if object_store is not None else SqlObjectStore(engine)
        self._metadata = build_asset_sql_metadata(object_store=object_store)
        self._context = create_sql_context(engine)
        self._lock = asyncio.Lock()
        self._ready = False

    @property
    def root(self) -> AssetRoot:
        return self._root

    @property
    def writable(self) -> bool:
        return True

    @property
    def atomic_batch(self) -> bool:
        return True

    async def initialize(self) -> None:
        await provision_sql(self._engine, self._metadata)
        await self._context.initialize()
        from sqlalchemy import insert, select
        from sqlalchemy.exc import IntegrityError

        table = self._metadata.tables["asset_revision"]
        try:
            async with self._begin() as connection:
                row = (await connection.execute(select(table.c.id).where(table.c.namespace_key == self._namespace_key))).first()
                if row is None:
                    await connection.execute(insert(table).values(namespace_key=self._namespace_key, store_revision=0))
        except IntegrityError:
            async with self._connect() as connection:
                row = await connection.scalar(select(table.c.id).where(table.c.namespace_key == self._namespace_key))
                if row is None:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
        await self._validate_persisted_objects()
        self._ready = True
        _logger.debug("SQL Asset backend initialized: namespace_key=%s", self._namespace_key)

    async def close(self) -> None:
        self._ready = False
        await self._context.close()

    async def head_revision(self) -> StorageRevision:
        await self._ensure_ready()
        from sqlalchemy import select
        table = self._metadata.tables["asset_revision"]
        async with self._connect() as connection:
            value = await connection.scalar(select(table.c.store_revision).where(table.c.namespace_key == self._namespace_key))
        if value is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return StorageRevision(str(int(value)))

    async def load_metadata(self, after_revision: StorageRevision | None) -> MetadataLoad[AssetKey, AssetInfo]:
        await self._ensure_ready()
        from sqlalchemy import select
        revision_table = self._metadata.tables["asset_revision"]
        entries_table = self._metadata.tables["asset_entries"]
        async with self._connect() as connection:
            value = await connection.scalar(select(revision_table.c.store_revision).where(revision_table.c.namespace_key == self._namespace_key))
            if value is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            current = int(value)
            if after_revision is not None and int(after_revision) == current:
                return MetadataLoad(MetadataLoadMode.PATCH, StorageRevision(str(current)), ())
            rows = (await connection.execute(select(entries_table).where(entries_table.c.namespace_key == self._namespace_key))).mappings().all()
        return MetadataLoad(MetadataLoadMode.REPLACE, StorageRevision(str(current)), tuple(MetadataChange(AssetKey(str(row["asset_kind"]), str(row["asset_id"])), self._info_from_row(row)) for row in rows))

    async def get(self, key: AssetKey) -> bytes | None:
        await self._ensure_ready()
        row = await self._current_row(key)
        if row is None:
            return None
        info = self._info_from_row(row)
        if info.status is not StorageEntryStatus.NORMAL:
            return None
        return await read_object(self._object_store, self._object_key_from_row(row), expected_digest=info.etag, expected_size=info.size)

    async def get_many(self, keys: Sequence[AssetKey]) -> dict[AssetKey, bytes]:
        await self._ensure_ready()
        result: dict[AssetKey, bytes] = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value
        return result

    async def stat(self, key: AssetKey) -> AssetInfo | None:
        await self._ensure_ready()
        row = await self._current_row(key)
        return None if row is None else self._info_from_row(row)

    async def put(self, key: AssetKey, value: bytes, *, expected_entry_revision: StorageEntryRevision | None = None) -> StoragePutResult[AssetInfo]:
        return await self._mutate("put", key, value, expected_entry_revision=expected_entry_revision)

    async def delete(self, key: AssetKey, *, expected_entry_revision: StorageEntryRevision | None = None) -> StorageDeleteResult[AssetKey]:
        return await self._mutate("delete", key, None, expected_entry_revision=expected_entry_revision)

    async def reset(self, key: AssetKey, *, expected_entry_revision: StorageEntryRevision | None = None) -> StorageResetResult[AssetKey]:
        return await self._mutate("reset", key, None, expected_entry_revision=expected_entry_revision)

    async def apply_batch(self, changes: Sequence[StorageChange[AssetKey, bytes]], *, expected_revision: StorageRevision | None = None) -> StorageBatchResult[AssetInfo, AssetKey]:
        await self._ensure_ready()
        async with self._lock:
            return await self._apply_batch_direct(changes, expected_revision=expected_revision)

    async def list_versions(self, key: AssetKey) -> tuple[VersionSummary, ...]:
        await self._ensure_ready()
        from sqlalchemy import select
        table = self._metadata.tables["asset_changes"]
        async with self._connect() as connection:
            rows = (await connection.execute(select(table).where(table.c.namespace_key == self._namespace_key, table.c.asset_key_hash == _asset_key_hash(key)).order_by(table.c.entry_revision))).mappings().all()
        summaries = []
        for row in rows:
            info = self._info_from_row(row)
            summaries.append(VersionSummary(info.revision, info.etag, info.size, info.modified_at, info.status))
        return tuple(summaries)

    async def get_at_revision(self, key: AssetKey, entry_revision: StorageEntryRevision) -> bytes | None:
        await self._ensure_ready()
        from sqlalchemy import select
        table = self._metadata.tables["asset_changes"]
        async with self._connect() as connection:
            row = (await connection.execute(select(table).where(table.c.namespace_key == self._namespace_key, table.c.asset_key_hash == _asset_key_hash(key), table.c.entry_revision == entry_revision.value))).mappings().first()
        if row is None:
            return None
        info = self._info_from_row(row)
        if info.status is not StorageEntryStatus.NORMAL:
            return None
        return await read_object(self._object_store, self._object_key_from_row(row), expected_digest=info.etag, expected_size=info.size)

    async def get_at_version(self, key: AssetKey, version: int) -> bytes | None:
        return await self.get_at_revision(key, StorageEntryRevision(version))

    async def _mutate(self, operation: str, key: AssetKey, value: bytes | None, *, expected_entry_revision: StorageEntryRevision | None) -> object:
        await self._ensure_ready()
        if operation == "put" and value is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        async with self._lock:
            from sqlalchemy import delete, insert, select, update

            revision_table = self._metadata.tables["asset_revision"]
            entry_table = self._metadata.tables["asset_entries"]
            change_table = self._metadata.tables["asset_changes"]
            content = bytes(value or b"")
            object_key = self._object_keys.key(_etag(content)) if operation == "put" else None
            if operation == "put":
                await self._object_store.put(object_key, _one(content), expected_size=len(content), expected_digest=_etag(content))
            async with self._begin() as connection:
                revision_row = (await connection.execute(select(revision_table.c.store_revision).where(revision_table.c.namespace_key == self._namespace_key).with_for_update())).first()
                if revision_row is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                current_row = (await connection.execute(select(entry_table).where(entry_table.c.namespace_key == self._namespace_key, entry_table.c.asset_key_hash == _asset_key_hash(key)).with_for_update())).mappings().first()
                current = None if current_row is None else self._info_from_row(current_row)
                _check_entry_revision(current, expected_entry_revision)
                if not _operation_mutates(operation, current, _etag(content)):
                    return _mutation_result(operation, key, current, StorageRevision(str(int(revision_row[0]))), changed=False)
                store_revision = StorageRevision(str(int(revision_row[0]) + 1))
                info = _next_info(self._root, key, content, current, operation, store_revision)
                row = _row_for_info(info, self._object_store.store_id, object_key)
                reservation = await connection.execute(
                    update(revision_table)
                    .where(
                        revision_table.c.namespace_key == self._namespace_key,
                        revision_table.c.store_revision == int(revision_row[0]),
                    )
                    .values(store_revision=int(store_revision.value), updated_at=datetime.now(timezone.utc))
                )
                if reservation.rowcount != 1:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                await connection.execute(insert(change_table).values(**row))
                await connection.execute(delete(entry_table).where(entry_table.c.namespace_key == self._namespace_key, entry_table.c.asset_key_hash == _asset_key_hash(key)))
                await connection.execute(insert(entry_table).values(**row))
            _logger.debug("SQL Asset mutation committed: namespace_key=%s revision=%s", self._namespace_key, store_revision)
            return _mutation_result(operation, key, info, store_revision, changed=True)

    async def _apply_batch_direct(self, changes: Sequence[StorageChange[AssetKey, bytes]], *, expected_revision: StorageRevision | None) -> StorageBatchResult[AssetInfo, AssetKey]:
        from sqlalchemy import delete, insert, select, update

        if len({change.key for change in changes}) != len(changes):
            raise AIError(ErrorCode.STORAGE_BATCH_DUPLICATE_KEY)
        revision_table = self._metadata.tables["asset_revision"]
        entry_table = self._metadata.tables["asset_entries"]
        change_table = self._metadata.tables["asset_changes"]
        prepared: dict[AssetKey, bytes] = {}
        for change in changes:
            if change.operation.name == "PUT":
                prepared[change.key] = bytes(change.value or b"")
                digest = _etag(prepared[change.key])
                await self._object_store.put(self._object_keys.key(digest), _one(prepared[change.key]), expected_size=len(prepared[change.key]), expected_digest=digest)
        async with self._begin() as connection:
            revision_row = (await connection.execute(select(revision_table.c.store_revision).where(revision_table.c.namespace_key == self._namespace_key).with_for_update())).first()
            if revision_row is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            current_revision = StorageRevision(str(int(revision_row[0])))
            if expected_revision is not None and expected_revision != current_revision:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            rows = (await connection.execute(select(entry_table).where(entry_table.c.namespace_key == self._namespace_key, entry_table.c.asset_key_hash.in_([_asset_key_hash(change.key) for change in changes])))).mappings().all()
            current = {AssetKey(str(row["asset_kind"]), str(row["asset_id"])): self._info_from_row(row) for row in rows}
            for change in changes:
                _check_entry_revision(current.get(change.key), change.expected_entry_revision)
            mutates = tuple(_operation_mutates(change.operation.name.lower(), current.get(change.key), _etag(prepared.get(change.key, b""))) for change in changes)
            if not any(mutates):
                return StorageBatchResult(current_revision, True, tuple(_mutation_result(change.operation.name.lower(), change.key, current.get(change.key), current_revision, changed=False) for change in changes))
            next_revision = StorageRevision(str(int(current_revision.value) + 1))
            reservation = await connection.execute(
                update(revision_table)
                .where(
                    revision_table.c.namespace_key == self._namespace_key,
                    revision_table.c.store_revision == int(current_revision.value),
                )
                .values(store_revision=int(next_revision.value), updated_at=datetime.now(timezone.utc))
            )
            if reservation.rowcount != 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            results = []
            for change, mutates_change in zip(changes, mutates, strict=True):
                previous = current.get(change.key)
                if not mutates_change:
                    results.append(_mutation_result(change.operation.name.lower(), change.key, previous, next_revision, changed=False))
                    continue
                content = prepared.get(change.key, b"")
                operation = change.operation.name.lower()
                info = _next_info(self._root, change.key, content, previous, operation, next_revision)
                row = _row_for_info(info, self._object_store.store_id, None if operation != "put" else self._object_keys.key(info.etag))
                await connection.execute(insert(change_table).values(**row))
                await connection.execute(delete(entry_table).where(entry_table.c.namespace_key == self._namespace_key, entry_table.c.asset_key_hash == _asset_key_hash(change.key)))
                await connection.execute(insert(entry_table).values(**row))
                results.append(_mutation_result(operation, change.key, info, next_revision, changed=True))
        _logger.debug("SQL Asset batch committed: namespace_key=%s revision=%s", self._namespace_key, next_revision)
        return StorageBatchResult(next_revision, True, tuple(results))

    async def _validate_persisted_objects(self) -> None:
        from sqlalchemy import select

        entry_table = self._metadata.tables["asset_entries"]
        change_table = self._metadata.tables["asset_changes"]
        async with self._connect() as connection:
            rows = list(
                (
                    await connection.execute(
                        select(entry_table).where(entry_table.c.namespace_key == self._namespace_key)
                    )
                ).mappings().all()
            )
            rows.extend(
                (
                    await connection.execute(
                        select(change_table).where(change_table.c.namespace_key == self._namespace_key)
                    )
                ).mappings().all()
            )
        for row in rows:
            info = self._info_from_row(row)
            if info.status is not StorageEntryStatus.NORMAL:
                continue
            object_key = self._object_key_from_row(row)
            stat = await self._object_store.stat(object_key)
            if stat is None or stat.digest != info.etag or stat.size != info.size:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _current_row(self, key: AssetKey) -> Mapping[str, object] | None:
        from sqlalchemy import select
        table = self._metadata.tables["asset_entries"]
        async with self._connect() as connection:
            return (await connection.execute(select(table).where(table.c.namespace_key == self._namespace_key, table.c.asset_key_hash == _asset_key_hash(key)))).mappings().first()

    def _info_from_row(self, row: Mapping[str, object]) -> AssetInfo:
        try:
            status = StorageEntryStatus(str(row["status"]))
            self._validate_object_columns(row, status)
            return AssetInfo(
                AssetKey(str(row["asset_kind"]), str(row["asset_id"])),
                StorageEntryRevision(int(row["entry_revision"])),
                StorageRevision(str(row["store_revision"])),
                str(row["etag"]),
                int(row["size"]),
                status,
                self._root.root_id,
                self._root.digest,
                _utc(row["modified_at"]),
            )
        except AIError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def _validate_object_columns(self, row: Mapping[str, object], status: StorageEntryStatus) -> None:
        fields = tuple(row.get(name) for name in ("object_store_id", "object_key", "object_digest", "object_size"))
        if status is StorageEntryStatus.NORMAL:
            if any(value is None for value in fields):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if str(row["object_store_id"]) != self._object_store.store_id:
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            if str(row["object_key"]) != self._object_keys.key(str(row["etag"])):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if str(row["object_digest"]) != str(row["etag"]) or int(row["object_size"]) != int(row["size"]):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        elif any(value is not None for value in fields):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _object_key_from_row(self, row: Mapping[str, object]) -> str:
        status = StorageEntryStatus(str(row["status"]))
        self._validate_object_columns(row, status)
        return str(row["object_key"])

    async def _ensure_ready(self) -> None:
        if not self._ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "asset store is not initialized")

    @asynccontextmanager
    async def _begin(self):
        async with self._context.sessions.begin() as session:
            yield session

    @asynccontextmanager
    async def _connect(self):
        async with self._context.sessions() as session:
            yield session

async def _one(value: bytes):
    yield value


def _check_entry_revision(current: AssetInfo | None, expected: StorageEntryRevision | None) -> None:
    if expected is not None and (current is None or current.revision != expected):
        raise AIError(ErrorCode.STORAGE_CONFLICT)


def _operation_mutates(operation: str, current: AssetInfo | None, digest: str) -> bool:
    if operation == "delete":
        return current is not None and current.status is not StorageEntryStatus.DELETED
    if operation == "reset":
        return current is not None and current.status is not StorageEntryStatus.RESET
    return current is None or current.status is not StorageEntryStatus.NORMAL or current.etag != digest


def _next_info(root: AssetRoot, key: AssetKey, content: bytes, previous: AssetInfo | None, operation: str, store_revision: StorageRevision) -> AssetInfo:
    status = StorageEntryStatus.NORMAL if operation == "put" else StorageEntryStatus.DELETED if operation == "delete" else StorageEntryStatus.RESET
    value = content if status is StorageEntryStatus.NORMAL else b""
    entry_revision = 1 if previous is None else previous.revision.value + 1
    return AssetInfo(key, StorageEntryRevision(entry_revision), store_revision, _etag(value), len(value), status, root.root_id, root.digest, datetime.now(timezone.utc))


def _row_for_info(info: AssetInfo, store_id: str, object_key: str | None) -> dict[str, object]:
    normal = info.status is StorageEntryStatus.NORMAL
    return {
        "namespace_key": info.root_digest,
        "asset_key_hash": _asset_key_hash(info.key),
        "asset_kind": info.key.kind,
        "asset_id": info.key.id,
        "entry_revision": info.revision.value,
        "store_revision": info.store_revision.value,
        "etag": info.etag,
        "size": info.size,
        "status": info.status.value,
        "object_store_id": store_id if normal else None,
        "object_key": object_key if normal else None,
        "object_digest": info.etag if normal else None,
        "object_size": info.size if normal else None,
        "modified_at": info.modified_at,
    }


def _mutation_result(operation: str, key: AssetKey, info: AssetInfo | None, store_revision: StorageRevision, *, changed: bool) -> object:
    if operation == "put":
        if info is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return StoragePutResult(info, info.revision, store_revision, changed)
    if operation == "delete":
        return StorageDeleteResult(key, changed, info.revision if changed and info is not None else None, store_revision)
    return StorageResetResult(key, changed, store_revision)


def _asset_key_hash(key: AssetKey) -> str:
    return hashlib.sha256(f"{key.kind}\0{key.id}".encode("utf-8")).hexdigest()


def _etag(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


__all__ = ["SqlAssetBackend", "build_asset_sql_metadata"]
