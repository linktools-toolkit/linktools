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

from ..core import JsonValue, canonical_identity_digest, validate_asset_namespace
from ..errors import AIError, ErrorCode
from ..storage import (
    MetadataChange,
    MetadataLoad,
    MetadataLoadMode,
    ObjectStore,
    SqlObjectStore,
    create_sql_storage_context,
    dialect_for_name,
    StorageBatchResult,
    StorageChange,
    StorageDeleteResult,
    StorageEntryRevision,
    StorageEntryStatus,
    StoragePutResult,
    StorageResetResult,
    StorageRevision,
    VersionSummary,
    normalize_storage_metadata,
    build_object_sql_metadata,
    read_object,
    sql_audit_columns,
    sql_digest,
    sql_id_column,
    sql_integer_id,
    sql_query_index,
    sql_table_options,
    sql_text_key,
    sql_unique,
)
from ._domain import AssetInfo, AssetKey, AssetRoot
from ._object import AssetObjectKeyFactory

if TYPE_CHECKING:
    from sqlalchemy import MetaData
    from sqlalchemy.ext.asyncio import AsyncEngine


_logger = environ.get_logger("ai.asset.sql")
_EMPTY_DIGEST = hashlib.sha256(b"").hexdigest()

def build_asset_sql_metadata(
    *,
    metadata: "MetaData | None" = None,
) -> "MetaData":
    """Build the Asset-owned tables in the supplied metadata collection."""

    from sqlalchemy import Column, DateTime, JSON, MetaData, Table

    if metadata is None:
        metadata = MetaData()
    digest = sql_digest()
    integer_id = sql_integer_id()
    revision = Table(
        "ai_asset_heads",
        metadata,
        sql_id_column(),
        Column("namespace_digest", digest, nullable=False, comment="Namespace SHA-256 digest"),
        Column("store_revision", integer_id, nullable=False, comment="Global storage revision"),
        *sql_audit_columns(),
        comment="Asset namespace heads",
        **sql_table_options(),
    )
    sql_unique(revision, "namespace_digest")
    sql_query_index(revision, "updated_at")
    sql_query_index(revision, "created_at")

    def asset_columns() -> tuple[Column, ...]:
        return (
            sql_id_column(),
            Column("namespace_digest", digest, nullable=False, comment="Namespace SHA-256 digest"),
            Column("asset_key_digest", digest, nullable=False, comment="Asset logical key SHA-256 digest"),
            Column("asset_kind", sql_text_key(128), nullable=False, comment="Asset kind"),
            Column("asset_id", sql_text_key(512), nullable=False, comment="Asset identifier"),
            Column("entry_revision", integer_id, nullable=False, comment="File revision"),
            Column("store_revision", integer_id, nullable=False, comment="Global storage revision"),
            Column("etag", digest, nullable=False, comment="File content SHA-256 ETag"),
            Column("size", integer_id, nullable=False, comment="Size in bytes"),
            Column("status", sql_text_key(32), nullable=False, comment="Status"),
            Column(
                "content_store_id",
                sql_text_key(128),
                nullable=True,
                comment="Content ObjectStore identifier",
            ),
            Column("content_key", sql_text_key(255), nullable=True, comment="Content object key"),
            Column("metadata", JSON(), nullable=False, comment="Extended metadata"),
            Column("modified_at", DateTime(timezone=True), nullable=False, comment="File modification time"),
            *sql_audit_columns(),
        )

    columns = asset_columns()
    _entry = Table(
        "ai_asset_entries",
        metadata,
        *columns,
        comment="Asset entries",
        **sql_table_options(),
    )
    sql_unique(_entry, "namespace_digest", "asset_key_digest")
    sql_query_index(_entry, "namespace_digest", "store_revision")
    sql_query_index(_entry, "updated_at")
    sql_query_index(_entry, "created_at")
    history_columns = asset_columns()
    _change = Table(
        "ai_asset_changes",
        metadata,
        *history_columns,
        comment="Asset change history",
        **sql_table_options(),
    )
    sql_unique(_change, "namespace_digest", "asset_key_digest", "entry_revision")
    sql_query_index(_change, "namespace_digest", "store_revision")
    sql_query_index(_change, "updated_at")
    sql_query_index(_change, "created_at")
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

        dialect_for_name(engine.dialect.name)
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
        self._namespace_digest = digest
        self._object_keys = AssetObjectKeyFactory(namespace)
        self._object_store = object_store if object_store is not None else SqlObjectStore(engine)
        from sqlalchemy import MetaData

        self._metadata = build_asset_sql_metadata(metadata=MetaData())
        if object_store is None:
            build_object_sql_metadata(metadata=self._metadata)
        self._context = create_sql_storage_context(engine)
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
        await self._context.initialize(metadata=self._metadata)
        from sqlalchemy import insert, select
        from sqlalchemy.exc import IntegrityError

        table = self._metadata.tables["ai_asset_heads"]
        try:
            async with self._begin() as connection:
                row = (await connection.execute(select(table.c.id).where(table.c.namespace_digest == self._namespace_digest))).first()
                if row is None:
                    await connection.execute(insert(table).values(namespace_digest=self._namespace_digest, store_revision=0))
        except IntegrityError:
            async with self._connect() as connection:
                row = await connection.scalar(select(table.c.id).where(table.c.namespace_digest == self._namespace_digest))
                if row is None:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
        await self._validate_persisted_objects()
        self._ready = True
        _logger.debug("SQL Asset backend initialized: namespace_digest=%s", self._namespace_digest)

    async def close(self) -> None:
        self._ready = False
        await self._context.close()

    async def head_revision(self) -> StorageRevision:
        await self._ensure_ready()
        from sqlalchemy import select
        table = self._metadata.tables["ai_asset_heads"]
        async with self._connect() as connection:
            value = await connection.scalar(select(table.c.store_revision).where(table.c.namespace_digest == self._namespace_digest))
        if value is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return StorageRevision(str(int(value)))

    async def load_metadata(self, after_revision: StorageRevision | None) -> MetadataLoad[AssetKey, AssetInfo]:
        await self._ensure_ready()
        from sqlalchemy import select
        revision_table = self._metadata.tables["ai_asset_heads"]
        entries_table = self._metadata.tables["ai_asset_entries"]
        async with self._connect() as connection:
            value = await connection.scalar(select(revision_table.c.store_revision).where(revision_table.c.namespace_digest == self._namespace_digest))
            if value is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            current = int(value)
            if after_revision is not None and int(after_revision) == current:
                return MetadataLoad(MetadataLoadMode.PATCH, StorageRevision(str(current)), ())
            rows = (await connection.execute(select(entries_table).where(entries_table.c.namespace_digest == self._namespace_digest))).mappings().all()
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

    async def put(self, key: AssetKey, value: bytes, *, expected_revision: StorageEntryRevision | None = None, metadata: Mapping[str, JsonValue] | None = None) -> StoragePutResult[AssetInfo]:
        return await self._mutate("put", key, value, expected_revision=expected_revision, metadata=metadata)

    async def delete(self, key: AssetKey, *, expected_revision: StorageEntryRevision | None = None, metadata: Mapping[str, JsonValue] | None = None) -> StorageDeleteResult[AssetKey]:
        return await self._mutate("delete", key, None, expected_revision=expected_revision, metadata=metadata)

    async def reset(self, key: AssetKey, *, expected_revision: StorageEntryRevision | None = None, metadata: Mapping[str, JsonValue] | None = None) -> StorageResetResult[AssetKey]:
        return await self._mutate("reset", key, None, expected_revision=expected_revision, metadata=metadata)

    async def apply_batch(self, changes: Sequence[StorageChange[AssetKey, bytes]], *, expected_revision: StorageRevision | None = None) -> StorageBatchResult[AssetInfo, AssetKey]:
        await self._ensure_ready()
        async with self._lock:
            return await self._apply_batch_direct(changes, expected_revision=expected_revision)

    async def list_versions(self, key: AssetKey) -> tuple[VersionSummary, ...]:
        await self._ensure_ready()
        from sqlalchemy import select
        table = self._metadata.tables["ai_asset_changes"]
        async with self._connect() as connection:
            rows = (await connection.execute(select(table).where(table.c.namespace_digest == self._namespace_digest, table.c.asset_key_digest == _asset_key_digest(key)).order_by(table.c.entry_revision))).mappings().all()
        summaries = []
        for row in rows:
            info = self._info_from_row(row)
            summaries.append(VersionSummary(info.revision, info.etag, info.size, info.modified_at, info.status, info.metadata))
        return tuple(summaries)

    async def get_at_revision(self, key: AssetKey, entry_revision: StorageEntryRevision) -> bytes | None:
        await self._ensure_ready()
        from sqlalchemy import select
        table = self._metadata.tables["ai_asset_changes"]
        async with self._connect() as connection:
            row = (await connection.execute(select(table).where(table.c.namespace_digest == self._namespace_digest, table.c.asset_key_digest == _asset_key_digest(key), table.c.entry_revision == entry_revision.value))).mappings().first()
        if row is None:
            return None
        info = self._info_from_row(row)
        if info.status is not StorageEntryStatus.NORMAL:
            return None
        return await read_object(self._object_store, self._object_key_from_row(row), expected_digest=info.etag, expected_size=info.size)

    async def get_at_version(self, key: AssetKey, version: int) -> bytes | None:
        return await self.get_at_revision(key, StorageEntryRevision(version))

    async def _mutate(self, operation: str, key: AssetKey, value: bytes | None, *, expected_revision: StorageEntryRevision | None, metadata: Mapping[str, JsonValue] | None) -> object:
        await self._ensure_ready()
        if operation == "put" and value is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        async with self._lock:
            from sqlalchemy import delete, insert, select, update

            revision_table = self._metadata.tables["ai_asset_heads"]
            entry_table = self._metadata.tables["ai_asset_entries"]
            change_table = self._metadata.tables["ai_asset_changes"]
            content = bytes(value or b"")
            object_key = self._object_keys.key(_etag(content)) if operation == "put" else None
            if operation == "put":
                await self._object_store.put(object_key, _one(content), expected_size=len(content), expected_digest=_etag(content))
            async with self._begin() as connection:
                revision_row = (await connection.execute(select(revision_table.c.store_revision).where(revision_table.c.namespace_digest == self._namespace_digest).with_for_update())).first()
                if revision_row is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                current_row = (await connection.execute(select(entry_table).where(entry_table.c.namespace_digest == self._namespace_digest, entry_table.c.asset_key_digest == _asset_key_digest(key)).with_for_update())).mappings().first()
                current = None if current_row is None else self._info_from_row(current_row)
                _check_entry_revision(current, expected_revision)
                if not _operation_mutates(operation, current, _etag(content)):
                    return _mutation_result(operation, key, current, StorageRevision(str(int(revision_row[0]))), changed=False)
                store_revision = StorageRevision(str(int(revision_row[0]) + 1))
                info = _next_info(
                    self._root,
                    key,
                    content,
                    current,
                    operation,
                    store_revision,
                    metadata,
                )
                row = _row_for_info(info, self._object_store.store_id, object_key)
                reservation = await connection.execute(
                    update(revision_table)
                    .where(
                        revision_table.c.namespace_digest == self._namespace_digest,
                        revision_table.c.store_revision == int(revision_row[0]),
                    )
                    .values(store_revision=int(store_revision.value), updated_at=datetime.now(timezone.utc))
                )
                if reservation.rowcount != 1:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                await connection.execute(insert(change_table).values(**row))
                await connection.execute(delete(entry_table).where(entry_table.c.namespace_digest == self._namespace_digest, entry_table.c.asset_key_digest == _asset_key_digest(key)))
                await connection.execute(insert(entry_table).values(**row))
            _logger.debug("SQL Asset mutation committed: namespace_digest=%s revision=%s", self._namespace_digest, store_revision)
            return _mutation_result(operation, key, info, store_revision, changed=True)

    async def _apply_batch_direct(self, changes: Sequence[StorageChange[AssetKey, bytes]], *, expected_revision: StorageRevision | None) -> StorageBatchResult[AssetInfo, AssetKey]:
        from sqlalchemy import delete, insert, select, update

        if len({change.key for change in changes}) != len(changes):
            raise AIError(ErrorCode.STORAGE_BATCH_DUPLICATE_KEY)
        revision_table = self._metadata.tables["ai_asset_heads"]
        entry_table = self._metadata.tables["ai_asset_entries"]
        change_table = self._metadata.tables["ai_asset_changes"]
        prepared: dict[AssetKey, bytes] = {}
        for change in changes:
            if change.operation.name == "PUT":
                prepared[change.key] = bytes(change.value or b"")
                digest = _etag(prepared[change.key])
                await self._object_store.put(self._object_keys.key(digest), _one(prepared[change.key]), expected_size=len(prepared[change.key]), expected_digest=digest)
        async with self._begin() as connection:
            revision_row = (await connection.execute(select(revision_table.c.store_revision).where(revision_table.c.namespace_digest == self._namespace_digest).with_for_update())).first()
            if revision_row is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            current_revision = StorageRevision(str(int(revision_row[0])))
            if expected_revision is not None and expected_revision != current_revision:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            rows = (await connection.execute(select(entry_table).where(entry_table.c.namespace_digest == self._namespace_digest, entry_table.c.asset_key_digest.in_([_asset_key_digest(change.key) for change in changes])))).mappings().all()
            current = {AssetKey(str(row["asset_kind"]), str(row["asset_id"])): self._info_from_row(row) for row in rows}
            for change in changes:
                _check_entry_revision(current.get(change.key), change.expected_revision)
            mutates = tuple(_operation_mutates(change.operation.name.lower(), current.get(change.key), _etag(prepared.get(change.key, b""))) for change in changes)
            if not any(mutates):
                return StorageBatchResult(current_revision, True, tuple(_mutation_result(change.operation.name.lower(), change.key, current.get(change.key), current_revision, changed=False) for change in changes))
            next_revision = StorageRevision(str(int(current_revision.value) + 1))
            reservation = await connection.execute(
                update(revision_table)
                .where(
                    revision_table.c.namespace_digest == self._namespace_digest,
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
                info = _next_info(
                    self._root,
                    change.key,
                    content,
                    previous,
                    operation,
                    next_revision,
                    change.metadata,
                )
                row = _row_for_info(info, self._object_store.store_id, None if operation != "put" else self._object_keys.key(info.etag))
                await connection.execute(insert(change_table).values(**row))
                await connection.execute(delete(entry_table).where(entry_table.c.namespace_digest == self._namespace_digest, entry_table.c.asset_key_digest == _asset_key_digest(change.key)))
                await connection.execute(insert(entry_table).values(**row))
                results.append(_mutation_result(operation, change.key, info, next_revision, changed=True))
        _logger.debug("SQL Asset batch committed: namespace_digest=%s revision=%s", self._namespace_digest, next_revision)
        return StorageBatchResult(next_revision, True, tuple(results))

    async def _validate_persisted_objects(self) -> None:
        from sqlalchemy import select

        entry_table = self._metadata.tables["ai_asset_entries"]
        change_table = self._metadata.tables["ai_asset_changes"]
        async with self._connect() as connection:
            rows = list(
                (
                    await connection.execute(
                        select(entry_table).where(entry_table.c.namespace_digest == self._namespace_digest)
                    )
                ).mappings().all()
            )
            rows.extend(
                (
                    await connection.execute(
                        select(change_table).where(change_table.c.namespace_digest == self._namespace_digest)
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
        table = self._metadata.tables["ai_asset_entries"]
        async with self._connect() as connection:
            return (await connection.execute(select(table).where(table.c.namespace_digest == self._namespace_digest, table.c.asset_key_digest == _asset_key_digest(key)))).mappings().first()

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
                normalize_storage_metadata(row["metadata"]),
            )
        except AIError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def _validate_object_columns(self, row: Mapping[str, object], status: StorageEntryStatus) -> None:
        fields = (row.get("content_store_id"), row.get("content_key"))
        if status is StorageEntryStatus.NORMAL:
            if any(value is None for value in fields):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if str(row["content_store_id"]) != self._object_store.store_id:
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            if str(row["content_key"]) != self._object_keys.key(str(row["etag"])):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        elif any(value is not None for value in fields):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _object_key_from_row(self, row: Mapping[str, object]) -> str:
        self._validate_object_columns(row, StorageEntryStatus(str(row["status"])))
        return str(row["content_key"])

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


def _next_info(
    root: AssetRoot,
    key: AssetKey,
    content: bytes,
    previous: AssetInfo | None,
    operation: str,
    store_revision: StorageRevision,
    metadata: Mapping[str, JsonValue] | None = None,
) -> AssetInfo:
    status = StorageEntryStatus.NORMAL if operation == "put" else StorageEntryStatus.DELETED if operation == "delete" else StorageEntryStatus.RESET
    value = content if status is StorageEntryStatus.NORMAL else b""
    entry_revision = 1 if previous is None else previous.revision.value + 1
    return AssetInfo(
        key,
        StorageEntryRevision(entry_revision),
        store_revision,
        _etag(value),
        len(value),
        status,
        root.root_id,
        root.digest,
        datetime.now(timezone.utc),
        normalize_storage_metadata(metadata),
    )


def _row_for_info(info: AssetInfo, store_id: str, object_key: str | None) -> dict[str, object]:
    normal = info.status is StorageEntryStatus.NORMAL
    return {
        "namespace_digest": info.root_digest,
        "asset_key_digest": _asset_key_digest(info.key),
        "asset_kind": info.key.kind,
        "asset_id": info.key.id,
        "entry_revision": info.revision.value,
        "store_revision": info.store_revision.value,
        "etag": info.etag,
        "size": info.size,
        "status": info.status.value,
        "content_store_id": store_id if normal else None,
        "content_key": object_key if normal else None,
        "metadata": dict(info.metadata),
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


def _asset_key_digest(key: AssetKey) -> str:
    return canonical_identity_digest("asset-key", {"kind": key.kind, "id": key.id})


def _etag(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


__all__ = ["SqlAssetBackend", "build_asset_sql_metadata"]
