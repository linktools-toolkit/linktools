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
    ObjectRef,
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
_SQL_BATCH_KEY_LIMIT = 256
_SQL_OPTIMISTIC_RETRY_LIMIT = 8


class _AssetRetry(Exception):
    pass

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
        from sqlalchemy import select

        table = self._metadata.tables["ai_asset_heads"]
        async with self._begin() as connection:
            await self._context.dialect.insert_ignore_conflict(
                connection,
                table=table,
                values={"namespace_digest": self._namespace_digest, "store_revision": 0},
                index_elements=("namespace_digest",),
            )
        async with self._connect() as connection:
            row = await connection.scalar(
                select(table.c.id).where(
                    table.c.namespace_digest == self._namespace_digest
                )
            )
        if row is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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
        from sqlalchemy import and_, select
        revision_table = self._metadata.tables["ai_asset_heads"]
        entries_table = self._metadata.tables["ai_asset_entries"]
        async with self._connect() as connection:
            entry_columns = [
                entries_table.c[column].label(f"entry_{column}")
                for column in entries_table.c.keys()
                if column != "id"
            ]
            condition = [
                entries_table.c.namespace_digest == self._namespace_digest,
            ]
            if after_revision is not None:
                condition.append(
                    revision_table.c.store_revision != int(after_revision.value)
                )
            statement = (
                select(
                    revision_table.c.store_revision.label("head_store_revision"),
                    entries_table.c.id.label("entry_id"),
                    *entry_columns,
                )
                .select_from(
                    revision_table.outerjoin(
                        entries_table,
                        and_(*condition),
                    )
                )
                .where(
                    revision_table.c.namespace_digest == self._namespace_digest
                )
            )
            rows = (await connection.execute(statement)).mappings().all()
            if not rows:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            current = int(rows[0]["head_store_revision"])
            if after_revision is not None and int(after_revision) == current:
                return MetadataLoad(MetadataLoadMode.PATCH, StorageRevision(str(current)), ())
            changes = []
            for row in rows:
                if row["entry_id"] is None:
                    continue
                entry = {
                    column: row[f"entry_{column}"]
                    for column in entries_table.c.keys()
                    if column != "id"
                }
                changes.append(
                    MetadataChange(
                        AssetKey(str(entry["asset_kind"]), str(entry["asset_id"])),
                        self._info_from_row(entry),
                    )
                )
        return MetadataLoad(
            MetadataLoadMode.REPLACE,
            StorageRevision(str(current)),
            tuple(changes),
        )

    async def get(self, key: AssetKey) -> bytes | None:
        return (await self.get_many((key,))).get(key)

    async def get_many(self, keys: Sequence[AssetKey]) -> dict[AssetKey, bytes]:
        await self._ensure_ready()
        unique = tuple(dict.fromkeys(keys))
        if not unique:
            return {}
        async with self._connect() as connection:
            rows = await self._current_rows_many(connection, unique)
        refs: dict[AssetKey, ObjectRef] = {}
        for key in unique:
            row = rows.get(key)
            if row is None:
                continue
            info = self._info_from_row(row)
            if info.status is StorageEntryStatus.NORMAL:
                refs[key] = ObjectRef(
                    self._object_store.store_id,
                    self._object_key_from_row(row),
                    info.etag,
                    info.size,
                )
        if isinstance(self._object_store, SqlObjectStore):
            values = await self._object_store.read_many(tuple(refs.values()))
        else:
            values = {
                key: await read_object(
                    self._object_store,
                    ref.key,
                    expected_digest=ref.digest,
                    expected_size=ref.size,
                )
                for key, ref in refs.items()
            }
        return {key: values[ref.key] for key, ref in refs.items()}

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
        if isinstance(self._object_store, SqlObjectStore):
            prepared = await self._prepare_batch_objects(changes)
            async with self._lock:
                return await self._apply_batch_direct(
                    changes,
                    expected_revision=expected_revision,
                    prepared=prepared,
                )
        async with self._lock:
            prepared = await self._prepare_batch_objects(changes)
            return await self._apply_batch_direct(
                changes,
                expected_revision=expected_revision,
                prepared=prepared,
            )

    async def list_versions(self, key: AssetKey) -> tuple[VersionSummary, ...]:
        await self._ensure_ready()
        from sqlalchemy import select
        table = self._metadata.tables["ai_asset_changes"]
        async with self._connect() as connection:
            rows = (await connection.execute(select(table).where(table.c.namespace_digest == self._namespace_digest, table.c.asset_key_digest == _asset_key_digest(key)).order_by(table.c.entry_revision))).mappings().all()
        summaries = []
        for row in rows:
            info = self._info_from_row(row)
            _validate_asset_row_identity(row, key)
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
        _validate_asset_row_identity(row, key)
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
        content = bytes(value or b"")
        object_key = self._object_keys.key(_etag(content)) if operation == "put" else None

        async def _put_object() -> None:
            if operation == "put":
                await self._object_store.put(
                    object_key,
                    _one(content),
                    expected_size=len(content),
                    expected_digest=_etag(content),
                )

        if isinstance(self._object_store, SqlObjectStore):
            await _put_object()
            async with self._lock:
                return await self._mutate_attempts(
                    operation,
                    key,
                    content,
                    object_key,
                    expected_revision,
                    metadata,
                )
        async with self._lock:
            await _put_object()
            return await self._mutate_attempts(
                operation,
                key,
                content,
                object_key,
                expected_revision,
                metadata,
            )

    async def _mutate_attempts(
        self,
        operation: str,
        key: AssetKey,
        content: bytes,
        object_key: str | None,
        expected_revision: StorageEntryRevision | None,
        metadata: Mapping[str, JsonValue] | None,
    ) -> object:
        from sqlalchemy import delete, insert, select, update

        revision_table = self._metadata.tables["ai_asset_heads"]
        entry_table = self._metadata.tables["ai_asset_entries"]
        change_table = self._metadata.tables["ai_asset_changes"]
        key_digest = _asset_key_digest(key)
        for attempt in range(_SQL_OPTIMISTIC_RETRY_LIMIT):
            try:
                no_op: tuple[AssetInfo | None, StorageRevision] | None = None
                async with self._begin() as connection:
                    revision_row = (
                        await connection.execute(
                            select(revision_table.c.store_revision).where(
                                revision_table.c.namespace_digest
                                == self._namespace_digest
                            )
                        )
                    ).first()
                    if revision_row is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    observed = int(revision_row[0])
                    current_rows = await self._current_rows_many(connection, (key,))
                    current_row = current_rows.get(key)
                    current = (
                        None
                        if current_row is None
                        else self._info_from_row(current_row)
                    )
                    _check_entry_revision(current, expected_revision)
                    if not _operation_mutates(
                        operation,
                        current,
                        _etag(content),
                    ):
                        no_op = (current, StorageRevision(str(observed)))
                    else:
                        store_revision = StorageRevision(str(observed + 1))
                        info = _next_info(
                            self._root,
                            key,
                            content,
                            current,
                            operation,
                            store_revision,
                            metadata,
                        )
                        row = _row_for_info(
                            info,
                            self._object_store.store_id,
                            object_key,
                        )
                        reservation = await connection.execute(
                            update(revision_table)
                            .where(
                                revision_table.c.namespace_digest
                                == self._namespace_digest,
                                revision_table.c.store_revision == observed,
                            )
                            .values(
                                store_revision=int(store_revision.value),
                                updated_at=datetime.now(timezone.utc),
                            )
                        )
                        if reservation.rowcount == 0:
                            raise _AssetRetry
                        if reservation.rowcount != 1:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        await connection.execute(insert(change_table).values(**row))
                        deleted = await connection.execute(
                            delete(entry_table).where(
                                entry_table.c.namespace_digest
                                == self._namespace_digest,
                                entry_table.c.asset_key_digest == key_digest,
                            )
                        )
                        if deleted.rowcount != (1 if current_row is not None else 0):
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        await connection.execute(insert(entry_table).values(**row))
                if no_op is not None:
                    current, store_revision = no_op
                    if await self._head_matches(store_revision):
                        return _mutation_result(
                            operation,
                            key,
                            current,
                            store_revision,
                            changed=False,
                        )
                    raise _AssetRetry
                _logger.debug(
                    "SQL Asset mutation committed: namespace_digest=%s revision=%s attempt=%s",
                    self._namespace_digest,
                    store_revision,
                    attempt + 1,
                )
                return _mutation_result(
                    operation,
                    key,
                    info,
                    store_revision,
                    changed=True,
                )
            except _AssetRetry:
                if attempt + 1 == _SQL_OPTIMISTIC_RETRY_LIMIT:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                await asyncio.sleep(0)
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def _head_matches(self, revision: StorageRevision) -> bool:
        from sqlalchemy import select

        table = self._metadata.tables["ai_asset_heads"]
        async with self._connect() as connection:
            value = await connection.scalar(
                select(table.c.store_revision).where(
                    table.c.namespace_digest == self._namespace_digest
                )
            )
        if value is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return int(value) == int(revision)

    async def _prepare_batch_objects(
        self,
        changes: Sequence[StorageChange[AssetKey, bytes]],
    ) -> dict[AssetKey, bytes]:
        prepared: dict[AssetKey, bytes] = {}
        for change in changes:
            if change.operation.name != "PUT":
                continue
            value = bytes(change.value or b"")
            prepared[change.key] = value
            digest = _etag(value)
            await self._object_store.put(
                self._object_keys.key(digest),
                _one(value),
                expected_size=len(value),
                expected_digest=digest,
            )
        return prepared

    async def _apply_batch_direct(
        self,
        changes: Sequence[StorageChange[AssetKey, bytes]],
        *,
        expected_revision: StorageRevision | None,
        prepared: Mapping[AssetKey, bytes],
    ) -> StorageBatchResult[AssetInfo, AssetKey]:
        from sqlalchemy import delete, insert, select, update

        if len({change.key for change in changes}) != len(changes):
            raise AIError(ErrorCode.STORAGE_BATCH_DUPLICATE_KEY)
        revision_table = self._metadata.tables["ai_asset_heads"]
        entry_table = self._metadata.tables["ai_asset_entries"]
        change_table = self._metadata.tables["ai_asset_changes"]
        for attempt in range(_SQL_OPTIMISTIC_RETRY_LIMIT):
            try:
                no_op: tuple[StorageRevision, tuple[object, ...]] | None = None
                async with self._begin() as connection:
                    revision_row = (
                        await connection.execute(
                            select(revision_table.c.store_revision).where(
                                revision_table.c.namespace_digest
                                == self._namespace_digest
                            )
                        )
                    ).first()
                    if revision_row is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    observed = int(revision_row[0])
                    current_revision = StorageRevision(str(observed))
                    if (
                        expected_revision is not None
                        and expected_revision != current_revision
                    ):
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                    current_rows = await self._current_rows_many(
                        connection,
                        tuple(change.key for change in changes),
                    )
                    current = {
                        key: self._info_from_row(row)
                        for key, row in current_rows.items()
                    }
                    for change in changes:
                        _check_entry_revision(
                            current.get(change.key),
                            change.expected_revision,
                        )
                    mutates = tuple(
                        _operation_mutates(
                            change.operation.name.lower(),
                            current.get(change.key),
                            _etag(prepared.get(change.key, b"")),
                        )
                        for change in changes
                    )
                    if not any(mutates):
                        no_op = (
                            current_revision,
                            tuple(
                                _mutation_result(
                                    change.operation.name.lower(),
                                    change.key,
                                    current.get(change.key),
                                    current_revision,
                                    changed=False,
                                )
                                for change in changes
                            ),
                        )
                    else:
                        next_revision = StorageRevision(str(observed + 1))
                        reservation = await connection.execute(
                            update(revision_table)
                            .where(
                                revision_table.c.namespace_digest
                                == self._namespace_digest,
                                revision_table.c.store_revision == observed,
                            )
                            .values(
                                store_revision=int(next_revision.value),
                                updated_at=datetime.now(timezone.utc),
                            )
                        )
                        if reservation.rowcount == 0:
                            raise _AssetRetry
                        if reservation.rowcount != 1:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        rows_to_insert: list[dict[str, object]] = []
                        results: list[object] = []
                        for change, mutates_change in zip(
                            changes,
                            mutates,
                            strict=True,
                        ):
                            previous = current.get(change.key)
                            operation = change.operation.name.lower()
                            if not mutates_change:
                                results.append(
                                    _mutation_result(
                                        operation,
                                        change.key,
                                        previous,
                                        next_revision,
                                        changed=False,
                                    )
                                )
                                continue
                            content = prepared.get(change.key, b"")
                            info = _next_info(
                                self._root,
                                change.key,
                                content,
                                previous,
                                operation,
                                next_revision,
                                change.metadata,
                            )
                            row = _row_for_info(
                                info,
                                self._object_store.store_id,
                                (
                                    None
                                    if operation != "put"
                                    else self._object_keys.key(info.etag)
                                ),
                            )
                            rows_to_insert.append(row)
                            results.append(
                                _mutation_result(
                                    operation,
                                    change.key,
                                    info,
                                    next_revision,
                                    changed=True,
                                )
                            )
                        await connection.execute(insert(change_table), rows_to_insert)
                        mutating_digests = [
                            _asset_key_digest(change.key)
                            for change, is_mutating in zip(
                                changes,
                                mutates,
                                strict=True,
                            )
                            if is_mutating
                        ]
                        deleted_count = 0
                        for batch in _batches(mutating_digests):
                            deleted = await connection.execute(
                                delete(entry_table).where(
                                    entry_table.c.namespace_digest
                                    == self._namespace_digest,
                                    entry_table.c.asset_key_digest.in_(batch),
                                )
                            )
                            deleted_count += deleted.rowcount
                        expected_deleted = sum(
                            1
                            for change, is_mutating in zip(
                                changes,
                                mutates,
                                strict=True,
                            )
                            if is_mutating and change.key in current
                        )
                        if deleted_count != expected_deleted:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        await connection.execute(insert(entry_table), rows_to_insert)
                if no_op is not None:
                    revision, results = no_op
                    if await self._head_matches(revision):
                        return StorageBatchResult(revision, True, results)
                    raise _AssetRetry
                _logger.debug(
                    "SQL Asset batch committed: namespace_digest=%s revision=%s attempt=%s",
                    self._namespace_digest,
                    next_revision,
                    attempt + 1,
                )
                return StorageBatchResult(next_revision, True, tuple(results))
            except _AssetRetry:
                if attempt + 1 == _SQL_OPTIMISTIC_RETRY_LIMIT:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                await asyncio.sleep(0)
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def _validate_persisted_objects(self) -> None:
        from sqlalchemy import select, union_all

        entry_table = self._metadata.tables["ai_asset_entries"]
        change_table = self._metadata.tables["ai_asset_changes"]
        columns = tuple(
            entry_table.c[column]
            for column in entry_table.c.keys()
            if column != "id"
        )
        statement = union_all(
            select(*columns).where(
                entry_table.c.namespace_digest == self._namespace_digest
            ),
            select(
                *tuple(
                    change_table.c[column]
                    for column in change_table.c.keys()
                    if column != "id"
                )
            ).where(
                change_table.c.namespace_digest == self._namespace_digest
            ),
        )
        async with self._connect() as connection:
            rows = (await connection.execute(statement)).mappings().all()
        references: dict[str, ObjectRef] = {}
        for row in rows:
            info = self._info_from_row(row)
            if str(row["asset_key_digest"]) != _asset_key_digest(info.key):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if info.status is not StorageEntryStatus.NORMAL:
                continue
            object_key = self._object_key_from_row(row)
            reference = ObjectRef(
                self._object_store.store_id,
                object_key,
                info.etag,
                info.size,
            )
            previous = references.get(object_key)
            if previous is not None and (
                previous.digest != reference.digest
                or previous.size != reference.size
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            references[object_key] = reference
        if isinstance(self._object_store, SqlObjectStore):
            stats = await self._object_store.stat_many(tuple(references))
        else:
            stats = {
                key: await self._object_store.stat(key)
                for key in references
            }
        for key, reference in references.items():
            stat = stats.get(key)
            if (
                stat is None
                or stat.digest != reference.digest
                or stat.size != reference.size
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _current_row(self, key: AssetKey) -> Mapping[str, object] | None:
        async with self._connect() as connection:
            rows = await self._current_rows_many(connection, (key,))
        return rows.get(key)

    async def _current_rows_many(
        self,
        connection: object,
        keys: Sequence[AssetKey],
    ) -> dict[AssetKey, Mapping[str, object]]:
        from sqlalchemy import select

        unique = tuple(dict.fromkeys(keys))
        if not unique:
            return {}
        digests = _asset_key_digests(unique)
        table = self._metadata.tables["ai_asset_entries"]
        result: dict[AssetKey, Mapping[str, object]] = {}
        seen_digests: set[str] = set()
        for batch in _batches(digests):
            rows = (
                await connection.execute(
                    select(table).where(
                        table.c.namespace_digest == self._namespace_digest,
                        table.c.asset_key_digest.in_(batch),
                    )
                )
            ).mappings().all()
            for row in rows:
                persisted_key = AssetKey(
                    str(row["asset_kind"]),
                    str(row["asset_id"]),
                )
                persisted_digest = str(row["asset_key_digest"])
                if _asset_key_digest(persisted_key) != persisted_digest:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                requested_key = next(
                    (
                        key
                        for key in unique
                        if _asset_key_digest(key) == persisted_digest
                    ),
                    None,
                )
                if requested_key is None or persisted_key != requested_key:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                if persisted_digest in seen_digests:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                seen_digests.add(persisted_digest)
                result[persisted_key] = row
        return result

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


def _asset_key_digests(keys: Sequence[AssetKey]) -> tuple[str, ...]:
    digests: dict[str, AssetKey] = {}
    for key in keys:
        digest = _asset_key_digest(key)
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


def _validate_asset_row_identity(
    row: Mapping[str, object],
    expected: AssetKey,
) -> None:
    actual = AssetKey(str(row["asset_kind"]), str(row["asset_id"]))
    if str(row["asset_key_digest"]) != _asset_key_digest(actual):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if actual != expected:
        raise AIError(ErrorCode.STORAGE_CONFLICT)


def _etag(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


__all__ = ["SqlAssetBackend", "build_asset_sql_metadata"]
