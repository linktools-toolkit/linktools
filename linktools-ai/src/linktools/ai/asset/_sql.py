#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL Asset backend with optimistic global revision CAS."""

import hashlib
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from linktools.core import environ

from ..core import JsonValue, validate_asset_namespace, validate_idempotency_key
from ..errors import AIError, ErrorCode
from ..storage import (
    MetadataChange,
    MetadataLoad,
    MetadataLoadMode,
    ObjectRef,
    ObjectStore,
    SqlObjectStore,
    StorageBatchResult,
    StorageChange,
    StorageDeleteResult,
    StorageEntryRevision,
    StorageEntryStatus,
    StorageOperation,
    StoragePutResult,
    StorageResetResult,
    StorageRevision,
    StoredPayload,
    VersionSummary,
    build_object_sql_metadata,
    create_sql_storage_context,
    dialect_for_name,
    sql_audit_columns,
    sql_audit_indexes,
    sql_id_column,
    sql_query_index,
    sql_table_options,
    sql_text_key,
    sql_unique,
)
from ._domain import AssetInfo, AssetKey, AssetRoot
from ._object import AssetObjectKeyFactory
from ._receipt import (
    decode_asset_batch_receipt,
    encode_asset_batch_receipt,
    validate_batch_receipt_identity,
)

if TYPE_CHECKING:
    from sqlalchemy import MetaData
    from sqlalchemy.ext.asyncio import AsyncEngine

_logger = environ.get_logger("ai.asset.sql")
_EMPTY_DIGEST = hashlib.sha256(b"").hexdigest()
_RETRY_LIMIT = 8
_ASSET_INFO_VERSION = 1
_ASSET_INFO_FIELDS = frozenset(
    {
        "kind", "id", "revision", "store_revision", "digest", "size", "status",
        "modified_at", "metadata", "content",
    }
)


def build_asset_sql_metadata(*, metadata: "MetaData | None" = None) -> "MetaData":
    from sqlalchemy import JSON, BigInteger, Column, LargeBinary, MetaData, Table
    from sqlalchemy.dialects.mysql import VARBINARY

    if metadata is None:
        metadata = MetaData()
    if "ai_asset_heads" in metadata.tables:
        return metadata
    heads = Table(
        "ai_asset_heads",
        metadata,
        sql_id_column(),
        Column(
            "namespace",
            sql_text_key(128),
            nullable=False,
            comment="Original AssetStore namespace.",
        ),
        Column(
            "store_revision", BigInteger, nullable=False, comment="Last committed namespace-wide AssetStore revision."
        ),
        *sql_audit_columns(),
        comment="Namespace-level optimistic revision head that serializes AssetStore mutations.",
        **sql_table_options(),
    )
    sql_unique(heads, "namespace")
    sql_audit_indexes(heads)
    entries = Table(
        "ai_asset_entries",
        metadata,
        sql_id_column(),
        Column(
            "namespace",
            sql_text_key(128),
            nullable=False,
            comment="Original AssetStore namespace.",
        ),
        Column("kind", sql_text_key(128), nullable=False, comment="Original Asset kind namespace."),
        Column(
            "asset_id", LargeBinary(512).with_variant(VARBINARY(512), "mysql"),
            nullable=False, comment="Exact UTF-8 bytes of the opaque Asset identifier.",
        ),
        Column("entry_revision", BigInteger, nullable=False, comment="Latest committed per-asset revision."),
        Column(
            "store_revision",
            BigInteger,
            nullable=False,
            comment="Namespace-wide revision that produced this current projection.",
        ),
        Column(
            "payload_json",
            JSON,
            nullable=False,
            comment=(
                "Versioned canonical current AssetInfo payload including status, metadata, "
                "content reference, and semantic timestamps."
            ),
        ),
        *sql_audit_columns(),
        comment="Current AssetStore projection for the latest committed revision of each asset key.",
        **sql_table_options(),
    )
    sql_unique(entries, "namespace", "kind", "asset_id")
    sql_query_index(entries, "namespace", "store_revision")
    sql_audit_indexes(entries)
    changes = Table(
        "ai_asset_changes",
        metadata,
        sql_id_column(),
        Column(
            "entry_id", BigInteger, nullable=False,
            comment="Surrogate identifier of the Asset entry owning this immutable revision.",
        ),
        Column(
            "entry_revision",
            BigInteger,
            nullable=False,
            comment="Immutable per-asset revision number represented by this history row.",
        ),
        Column(
            "namespace",
            sql_text_key(128),
            nullable=False,
            comment="Original AssetStore namespace.",
        ),
        Column(
            "store_revision",
            BigInteger,
            nullable=False,
            comment="Namespace-wide revision in which this asset change committed.",
        ),
        Column(
            "payload_json", JSON, nullable=False, comment="Versioned canonical immutable AssetInfo history payload."
        ),
        *sql_audit_columns(),
        comment="Immutable AssetStore history containing every committed per-asset revision.",
        **sql_table_options(),
    )
    sql_unique(changes, "entry_id", "entry_revision")
    sql_unique(changes, "namespace", "store_revision", "entry_id")
    sql_audit_indexes(changes)
    receipts = Table(
        "ai_asset_batch_receipts",
        metadata,
        sql_id_column(),
        Column(
            "namespace",
            sql_text_key(128),
            nullable=False,
            comment="Original AssetStore namespace.",
        ),
        Column(
            "idempotency_key",
            sql_text_key(256),
            nullable=False,
            comment="Original caller batch idempotency key.",
        ),
        Column(
            "payload_json",
            JSON,
            nullable=False,
            comment="Versioned committed Asset batch receipt without input file bytes.",
        ),
        *sql_audit_columns(),
        comment="Durable idempotency receipts for committed atomic AssetStore batches.",
        **sql_table_options(),
    )
    sql_unique(receipts, "namespace", "idempotency_key")
    sql_audit_indexes(receipts)
    return metadata


class SqlAssetBackend:
    def __init__(
        self,
        engine: "AsyncEngine",
        *,
        namespace: str,
        object_store: ObjectStore | None = None,
    ) -> None:
        validate_asset_namespace(namespace)
        dialect_for_name(engine.dialect.name)
        self._namespace = namespace
        self._root = AssetRoot("sql", namespace)
        self._context = create_sql_storage_context(engine)
        self._metadata = build_asset_sql_metadata()
        self._object_store = object_store or SqlObjectStore.from_context(self._context)
        if object_store is None:
            build_object_sql_metadata(metadata=self._metadata)
        self._ready = False

    @property
    def retains_versions(self) -> bool:
        return True

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
        await self._context.initialize()
        table = self._metadata.tables["ai_asset_heads"]

        async def initialize_head(session) -> None:
            await self._context.run_statement(self._context.dialect.insert_ignore_conflict(
                session,
                table=table,
                values={"namespace": self._namespace, "store_revision": 0},
                index_elements=("namespace",),
            ))

        await self._context.run_mutation(initialize_head)
        self._ready = True
        _logger.info("SQL Asset backend initialized")

    async def close(self) -> None:
        self._ready = False
        await self._context.close()

    async def head_revision(self) -> StorageRevision:
        row = await self._head()
        return StorageRevision(str(row))

    async def load_metadata(self, after_revision: StorageRevision | None) -> MetadataLoad[AssetKey, AssetInfo]:
        self._ensure_ready()
        after = None if after_revision is None else int(after_revision.value)
        heads = self._metadata.tables["ai_asset_heads"]
        entries = self._metadata.tables["ai_asset_entries"]
        session = self._context.sessions()
        try:
            from sqlalchemy import null, select

            # One statement binds the head and entries to the same database
            # snapshot. Mutually exclusive branches avoid scanning entries when
            # the head is unchanged, including on SQLite's nested-loop join.
            statement = (
                select(heads.c.store_revision, entries.c.id, entries.c.payload_json)
                .select_from(heads.outerjoin(entries, heads.c.namespace == entries.c.namespace))
                .where(heads.c.namespace == self._namespace)
            )
            if after is not None:
                statement = statement.where(heads.c.store_revision != after).union_all(
                    select(heads.c.store_revision, null(), null()).where(
                        heads.c.namespace == self._namespace,
                        heads.c.store_revision == after,
                    )
                )
            rows = (
                (
                    await self._context.run_statement(session.execute(statement))
                )
                .mappings()
                .all()
            )
        finally:
            await session.close()
        if not rows:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        current = int(rows[0]["store_revision"])
        if after == current:
            return MetadataLoad(MetadataLoadMode.PATCH, StorageRevision(str(current)), ())
        return MetadataLoad(
            MetadataLoadMode.REPLACE,
            StorageRevision(str(current)),
            tuple(
                MetadataChange(_key_from_data(row["payload_json"]), _info_from_data(row["payload_json"]))
                for row in rows if row["id"] is not None
            ),
        )

    async def get(self, key: AssetKey) -> bytes | None:
        info = await self.stat(key)
        if info is None or info.status is not StorageEntryStatus.NORMAL:
            return None
        return await _read_asset_object(self._object_store, self._namespace, key, info)

    async def get_many(self, keys: Sequence[AssetKey]) -> dict[AssetKey, bytes]:
        self._ensure_ready()
        current = await self._load_current(tuple(dict.fromkeys(keys)))
        result: dict[AssetKey, bytes] = {}
        for key in dict.fromkeys(keys):
            entry = current[key]
            info = None if entry is None else entry[1]
            if info is not None and info.status is StorageEntryStatus.NORMAL:
                result[key] = await _read_asset_object(
                    self._object_store,
                    self._namespace,
                    key,
                    info,
                )
        return result

    async def stat(self, key: AssetKey) -> AssetInfo | None:
        self._ensure_ready()
        entries = self._metadata.tables["ai_asset_entries"]
        session = self._context.sessions()
        try:
            from sqlalchemy import select

            row = (
                await self._context.run_statement(session.execute(
                    select(entries.c.payload_json).where(
                        entries.c.kind == key.kind,
                        entries.c.asset_id == key.id.encode("utf-8"),
                        entries.c.namespace == self._namespace,
                    )
                ))
            ).scalar_one_or_none()
        finally:
            await session.close()
        return None if row is None else _info_from_data(row)

    async def put(
        self,
        key: AssetKey,
        value: bytes,
        *,
        expected_revision: StorageEntryRevision | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> StoragePutResult[AssetInfo]:
        result = await self.apply_batch(
            (StorageChange(StorageOperation.PUT, key, value, expected_revision, metadata or {}),)
        )
        return result.results[0]

    async def delete(
        self,
        key: AssetKey,
        *,
        expected_revision: StorageEntryRevision | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> StorageDeleteResult[AssetKey]:
        result = await self.apply_batch(
            (StorageChange(StorageOperation.DELETE, key, None, expected_revision, metadata or {}),)
        )
        return result.results[0]

    async def reset(
        self,
        key: AssetKey,
        *,
        expected_revision: StorageEntryRevision | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> StorageResetResult[AssetKey]:
        result = await self.apply_batch(
            (StorageChange(StorageOperation.RESET, key, None, expected_revision, metadata or {}),)
        )
        return result.results[0]

    async def apply_batch(
        self,
        changes: Sequence[StorageChange[AssetKey, bytes]],
        *,
        expected_revision: StorageRevision | None = None,
        idempotency_key: str | None = None,
        request_digest: str | None = None,
    ) -> StorageBatchResult[AssetInfo, AssetKey]:
        self._ensure_ready()
        validate_batch_receipt_identity(idempotency_key, request_digest)
        if len({change.key for change in changes}) != len(changes):
            raise AIError(ErrorCode.STORAGE_BATCH_DUPLICATE_KEY)
        if idempotency_key is not None:
            previous = await self.batch_result(idempotency_key)
            if previous is not None:
                if previous.request_digest != request_digest:
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                return previous
        for _ in range(_RETRY_LIMIT):
            try:
                result = await self._apply_once(
                    changes,
                    expected_revision,
                    idempotency_key,
                    request_digest,
                )
            except AIError as error:
                if idempotency_key is None or error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                previous = await self.batch_result(idempotency_key)
                if previous is not None:
                    if previous.request_digest != request_digest:
                        raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT) from error
                    return previous
                continue
            if result is not None:
                return result
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def batch_result(
        self,
        idempotency_key: str,
    ) -> StorageBatchResult[AssetInfo, AssetKey] | None:
        self._ensure_ready()
        validate_idempotency_key(idempotency_key)
        table = self._metadata.tables["ai_asset_batch_receipts"]
        session = self._context.sessions()
        try:
            from sqlalchemy import select

            row = (
                await self._context.run_statement(session.execute(
                    select(table.c.payload_json).where(
                        table.c.namespace == self._namespace,
                        table.c.idempotency_key == idempotency_key,
                    )
                ))
            ).scalar_one_or_none()
        finally:
            await session.close()
        if row is None:
            return None
        return decode_asset_batch_receipt(
            row,
            idempotency_key=idempotency_key,
        )

    async def _apply_once(
        self,
        changes: Sequence[StorageChange[AssetKey, bytes]],
        expected_revision: StorageRevision | None,
        idempotency_key: str | None,
        request_digest: str | None,
    ) -> StorageBatchResult[AssetInfo, AssetKey] | None:
        try:
            return await self._apply_once_transaction(
                changes,
                expected_revision,
                idempotency_key,
                request_digest,
            )
        except AIError:
            raise
        except Exception as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error

    async def _apply_once_transaction(
        self,
        changes: Sequence[StorageChange[AssetKey, bytes]],
        expected_revision: StorageRevision | None,
        idempotency_key: str | None,
        request_digest: str | None,
    ) -> StorageBatchResult[AssetInfo, AssetKey] | None:
        current_revision = await self._head()
        if expected_revision is not None and int(expected_revision.value) != current_revision:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        loaded = await self._load_current(tuple(change.key for change in changes))
        current = {key: None if entry is None else entry[1] for key, entry in loaded.items()}
        for change in changes:
            previous = current[change.key]
            if change.expected_revision is not None and (
                previous is None or previous.revision != change.expected_revision
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        content_digests = {
            change.key: hashlib.sha256(bytes(change.value or b"")).hexdigest()
            for change in changes if change.operation is StorageOperation.PUT
        }
        mutations = tuple(
            change for change in changes
            if _mutates(change, current[change.key], content_digests.get(change.key, _EMPTY_DIGEST))
        )
        next_revision = current_revision + bool(mutations)
        if next_revision == current_revision and idempotency_key is None:
            return StorageBatchResult(
                StorageRevision(str(current_revision)),
                True,
                tuple(_unchanged(change, current[change.key], current_revision) for change in changes),
            )
        prepared: dict[AssetKey, AssetInfo] = {}
        if next_revision != current_revision:
            for change in mutations:
                info = _next_info(
                    change, current[change.key], next_revision,
                    content_digests.get(change.key, _EMPTY_DIGEST),
                )
                if change.operation is StorageOperation.PUT:
                    content = bytes(change.value or b"")
                    object_key = _asset_object_key(self._namespace, info.digest)
                    await _put_asset_object(self._object_store, object_key, content, info.digest)
                    info = replace(
                        info,
                        content=StoredPayload.object(
                            ObjectRef(
                                self._object_store.store_id,
                                object_key,
                                info.digest,
                                info.size,
                            )
                        ),
                    )
                prepared[change.key] = info
        entries = self._metadata.tables["ai_asset_entries"]
        history = self._metadata.tables["ai_asset_changes"]
        heads = self._metadata.tables["ai_asset_heads"]
        receipts = self._metadata.tables["ai_asset_batch_receipts"]

        async def execute(session) -> StorageBatchResult[AssetInfo, AssetKey] | None:
            from sqlalchemy import func, select, update

            if idempotency_key is not None:
                existing_payload = (
                    await self._context.run_statement(session.execute(
                        select(receipts.c.payload_json).where(
                            receipts.c.namespace == self._namespace,
                            receipts.c.idempotency_key == idempotency_key,
                        )
                    ))
                ).scalar_one_or_none()
                if existing_payload is not None:
                    existing = decode_asset_batch_receipt(
                        existing_payload,
                        idempotency_key=idempotency_key,
                    )
                    if existing.request_digest != request_digest:
                        raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                    return existing
            head_result = await self._context.run_statement(session.execute(
                update(heads)
                .where(
                    heads.c.namespace == self._namespace,
                    heads.c.store_revision == current_revision,
                )
                .values(
                    store_revision=next_revision,
                    updated_at=func.current_timestamp(),
                )
            ))
            if head_result.rowcount != 1:
                return None
            values: list[
                StoragePutResult[AssetInfo] | StorageDeleteResult[AssetKey] | StorageResetResult[AssetKey]
            ] = []
            for change in changes:
                info = prepared.get(change.key)
                if info is None:
                    values.append(_unchanged(change, current[change.key], next_revision))
                    continue
                data = _info_data(info)
                entry = loaded[change.key]
                entry_values = {
                    "entry_revision": info.revision.value,
                    "store_revision": next_revision,
                    "payload_json": data,
                }
                if entry is None:
                    inserted = await self._context.run_statement(session.execute(
                        entries.insert().values(
                            namespace=self._namespace,
                            kind=change.key.kind,
                            asset_id=change.key.id.encode("utf-8"),
                            **entry_values,
                        )
                    ))
                    entry_id = inserted.inserted_primary_key[0]
                else:
                    entry_id = entry[0]
                    await self._context.run_statement(session.execute(
                        entries.update().where(entries.c.id == entry_id).values(
                            **entry_values, updated_at=func.current_timestamp(),
                        )
                    ))
                await self._context.run_statement(session.execute(
                    history.insert().values(
                        entry_id=entry_id,
                        entry_revision=info.revision.value,
                        namespace=self._namespace,
                        store_revision=next_revision,
                        payload_json=data,
                    )
                ))
                values.append(_result(change, info, next_revision))
            result = StorageBatchResult(
                StorageRevision(str(next_revision)),
                True,
                tuple(values),
                request_digest,
                idempotency_key,
            )
            if idempotency_key is not None:
                await self._context.run_statement(session.execute(
                    receipts.insert().values(
                        namespace=self._namespace,
                        idempotency_key=idempotency_key,
                        payload_json=encode_asset_batch_receipt(result),
                    )
                ))
            return result

        return await self._context.run_mutation(execute)

    async def _load_current(self, keys: Sequence[AssetKey]) -> dict[AssetKey, tuple[int, AssetInfo] | None]:
        entries = self._metadata.tables["ai_asset_entries"]
        session = self._context.sessions()
        try:
            from sqlalchemy import select, tuple_

            rows = (
                await self._context.run_statement(session.execute(
                    select(entries.c.id, entries.c.kind, entries.c.asset_id, entries.c.payload_json).where(
                        entries.c.namespace == self._namespace,
                        tuple_(entries.c.kind, entries.c.asset_id).in_(
                            tuple((key.kind, key.id.encode("utf-8")) for key in keys)
                        ),
                    )
                ))
            ).all()
        finally:
            await session.close()
        by_key = {
            AssetKey(row.kind, bytes(row.asset_id).decode("utf-8")):
                (int(row.id), _info_from_data(row.payload_json))
            for row in rows
        }
        return {key: by_key.get(key) for key in keys}

    async def list_versions(self, key: AssetKey) -> tuple[VersionSummary, ...]:
        self._ensure_ready()
        history = self._metadata.tables["ai_asset_changes"]
        entries = self._metadata.tables["ai_asset_entries"]
        session = self._context.sessions()
        try:
            from sqlalchemy import select

            rows = (
                (
                    await self._context.run_statement(session.execute(
                        select(history.c.payload_json)
                        .join(entries, entries.c.id == history.c.entry_id)
                        .where(
                            entries.c.namespace == self._namespace,
                            entries.c.kind == key.kind,
                            entries.c.asset_id == key.id.encode("utf-8"),
                            history.c.namespace == self._namespace,
                        )
                        .order_by(history.c.entry_revision)
                    ))
                )
                .scalars()
                .all()
            )
        finally:
            await session.close()
        return tuple(_summary(_info_from_data(row)) for row in rows)

    async def get_at_revision(self, key: AssetKey, entry_revision: StorageEntryRevision) -> bytes | None:
        history = self._metadata.tables["ai_asset_changes"]
        entries = self._metadata.tables["ai_asset_entries"]
        session = self._context.sessions()
        try:
            from sqlalchemy import select

            data = await self._context.run_statement(session.scalar(
                select(history.c.payload_json).join(entries, entries.c.id == history.c.entry_id).where(
                    entries.c.namespace == self._namespace,
                    entries.c.kind == key.kind,
                    entries.c.asset_id == key.id.encode("utf-8"),
                    history.c.entry_revision == entry_revision.value,
                    history.c.namespace == self._namespace,
                )
            ))
        finally:
            await session.close()
        if data is None:
            return None
        info = _info_from_data(data)
        return (
            None
            if info.status is not StorageEntryStatus.NORMAL
            else await _read_asset_object(self._object_store, self._namespace, key, info)
        )

    async def _head(self) -> int:
        heads = self._metadata.tables["ai_asset_heads"]
        session = self._context.sessions()
        try:
            from sqlalchemy import select

            value = await self._context.run_statement(session.scalar(
                select(heads.c.store_revision).where(heads.c.namespace == self._namespace)
            ))
        finally:
            await session.close()
        if value is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return int(value)

    async def validate_integrity(self) -> None:
        self._ensure_ready()
        entries = self._metadata.tables["ai_asset_entries"]
        history = self._metadata.tables["ai_asset_changes"]
        heads = self._metadata.tables["ai_asset_heads"]
        session = self._context.sessions()
        try:
            from sqlalchemy import select

            namespace = self._namespace
            head = await self._context.run_statement(session.scalar(select(heads.c.store_revision).where(heads.c.namespace == namespace)))
            if head is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            head_revision = int(head)
            if head_revision < 0:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            entry_rows = (
                (
                    await self._context.run_statement(session.execute(
                        select(entries.c.kind, entries.c.asset_id, entries.c.entry_revision, entries.c.payload_json).where(
                            entries.c.namespace == namespace
                        )
                    ))
                )
                .mappings()
                .all()
            )
            history_rows = (
                (
                    await self._context.run_statement(session.execute(
                        select(entries.c.kind, entries.c.asset_id, history.c.entry_revision, history.c.payload_json)
                        .select_from(history.outerjoin(
                            entries,
                            (entries.c.id == history.c.entry_id) & (entries.c.namespace == history.c.namespace),
                        ))
                        .where(history.c.namespace == namespace)
                        .order_by(history.c.entry_id, history.c.entry_revision)
                    ))
                )
                .mappings()
                .all()
            )
        finally:
            await session.close()
        parsed_entries = [(row, self._audit_info(row, head_revision)) for row in entry_rows]
        parsed_history = [(row, self._audit_info(row, head_revision)) for row in history_rows]
        checked_objects: set[str] = set()
        for _, info in (*parsed_entries, *parsed_history):
            if info.status is not StorageEntryStatus.NORMAL:
                continue
            reference = _asset_object_ref(self._object_store, self._namespace, info)
            if reference.key in checked_objects:
                continue
            checked_objects.add(reference.key)
            stat = await self._object_store.stat(reference.key)
            if stat is None or stat.digest != info.digest or stat.size != info.size:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        grouped: dict[AssetKey, list[tuple[Mapping[str, object], AssetInfo]]] = {}
        for row, info in parsed_history:
            grouped.setdefault(info.key, []).append((row, info))
        current_by_key = {info.key: (row, info) for row, info in parsed_entries}
        for key, rows in grouped.items():
            revisions = [info.revision.value for _, info in rows]
            if revisions != list(range(1, len(revisions) + 1)):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            current = current_by_key.get(key)
            if current is None or current[1].revision.value != revisions[-1]:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current[1] != rows[-1][1]:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            store_revisions = [int(info.store_revision.value) for _, info in rows]
            if store_revisions != sorted(store_revisions) or len(set(store_revisions)) != len(store_revisions):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if set(current_by_key) != set(grouped):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        store_revisions = [int(info.store_revision.value) for _, info in (*parsed_entries, *parsed_history)]
        if head_revision == 0 and store_revisions:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if head_revision > 0 and (not store_revisions or max(store_revisions) != head_revision):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.info("SQL Asset integrity validated")

    def _audit_info(self, row: Mapping[str, object], head: int) -> AssetInfo:
        try:
            info = _info_from_data(row["payload_json"])
            if row["kind"] != info.key.kind or row["asset_id"] != info.key.id.encode("utf-8"):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if int(row["entry_revision"]) != info.revision.value:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            store_revision = int(info.store_revision.value)
            if store_revision < 1 or store_revision > head:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return info
        except AIError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def _ensure_ready(self) -> None:
        if not self._ready:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)


def _info_data(info: AssetInfo) -> dict[str, JsonValue]:
    return {
        "v": _ASSET_INFO_VERSION,
        "value": {
            "kind": info.key.kind,
            "id": info.key.id,
            "revision": info.revision.value,
            "store_revision": info.store_revision.value,
            "digest": info.digest,
            "size": info.size,
            "status": info.status.value,
            "modified_at": info.modified_at.isoformat(),
            "metadata": dict(info.metadata),
            "content": None if info.content is None else info.content.to_json(),
        },
    }


def _asset_info_fields(data: Mapping[str, object]) -> Mapping[str, object]:
    if set(data) != {"v", "value"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    version = data["v"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if version != _ASSET_INFO_VERSION:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    value = data["value"]
    if not isinstance(value, Mapping) or set(value) != _ASSET_INFO_FIELDS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _asset_text(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _asset_int(value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _asset_datetime(value: object) -> datetime:
    raw = _asset_text(value)
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if parsed.tzinfo is None or parsed.isoformat() != raw:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return parsed


def _info_from_data(data: Mapping[str, object]) -> AssetInfo:
    if not isinstance(data, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    value = _asset_info_fields(data)
    metadata = value["metadata"]
    if not isinstance(metadata, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    content = value["content"]
    try:
        return AssetInfo(
            AssetKey(_asset_text(value["kind"]), _asset_text(value["id"])),
            StorageEntryRevision(_asset_int(value["revision"], minimum=1)),
            StorageRevision(_asset_text(value["store_revision"])),
            _asset_text(value["digest"]),
            _asset_int(value["size"], minimum=0),
            StorageEntryStatus(_asset_text(value["status"])),
            _asset_datetime(value["modified_at"]),
            dict(metadata),
            None if content is None else StoredPayload.from_json(content),
        )
    except AIError:
        raise
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _key_from_data(data: Mapping[str, object]) -> AssetKey:
    value = _asset_info_fields(data)
    try:
        return AssetKey(_asset_text(value["kind"]), _asset_text(value["id"]))
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

def _next_info(
    change: StorageChange[AssetKey, bytes], previous: AssetInfo | None, store_revision: int, digest: str
) -> AssetInfo:
    revision = 1 if previous is None else previous.revision.value + 1
    value = bytes(change.value or b"") if change.operation is StorageOperation.PUT else b""
    status = {
        StorageOperation.PUT: StorageEntryStatus.NORMAL,
        StorageOperation.DELETE: StorageEntryStatus.DELETED,
        StorageOperation.RESET: StorageEntryStatus.RESET,
    }[change.operation]
    return AssetInfo(
        change.key,
        StorageEntryRevision(revision),
        StorageRevision(str(store_revision)),
        digest,
        len(value),
        status,
        datetime.now(timezone.utc),
        change.metadata,
    )


def _summary(info: AssetInfo) -> VersionSummary:
    return VersionSummary(info.revision, info.digest, info.size, info.modified_at, info.status, info.metadata)


def _result(change: StorageChange[AssetKey, bytes], info: AssetInfo, store_revision: int) -> object:
    if change.operation is StorageOperation.PUT:
        return StoragePutResult(info, info.revision, StorageRevision(str(store_revision)), True)
    if change.operation is StorageOperation.DELETE:
        return StorageDeleteResult(change.key, True, info.revision, StorageRevision(str(store_revision)))
    return StorageResetResult(change.key, True, StorageRevision(str(store_revision)))


def _unchanged(change: StorageChange[AssetKey, bytes], info: AssetInfo | None, store_revision: int) -> object:
    if change.operation is StorageOperation.PUT and info is not None:
        return StoragePutResult(info, info.revision, StorageRevision(str(store_revision)), False)
    if change.operation is StorageOperation.DELETE:
        return StorageDeleteResult(change.key, False, None, StorageRevision(str(store_revision)))
    return StorageResetResult(change.key, False, StorageRevision(str(store_revision)))


def _mutates(change: StorageChange[AssetKey, bytes], previous: AssetInfo | None, digest: str) -> bool:
    if previous is None:
        return change.operation is StorageOperation.PUT
    if change.operation is StorageOperation.PUT:
        return (
            previous.status is not StorageEntryStatus.NORMAL
            or previous.digest != digest
        )
    target = {
        StorageOperation.DELETE: StorageEntryStatus.DELETED,
        StorageOperation.RESET: StorageEntryStatus.RESET,
    }[change.operation]
    return previous.status is not target


async def _put_asset_object(store: ObjectStore, key: str, value: bytes, digest: str) -> None:

    async def chunks() -> AsyncIterator[bytes]:
        if value:
            yield value

    await store.put(key, chunks(), expected_size=len(value), expected_digest=digest)


async def _read_asset_object(store: ObjectStore, namespace: str, key: AssetKey, info: AssetInfo) -> bytes:
    if info.key != key:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    reference = _asset_object_ref(store, namespace, info)
    return await _read_object_bytes(store, reference.key, reference.digest, reference.size)


def _asset_object_ref(store: ObjectStore, namespace: str, info: AssetInfo) -> ObjectRef:
    content = info.content
    if (content is None or content.kind != "object" or content.ref is None
            or content.ref.store_id != store.store_id
            or content.ref.key != _asset_object_key(namespace, info.digest)):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return content.ref


async def _read_object_bytes(store: ObjectStore, key: str, digest: str, size: int) -> bytes:
    from ..storage import read_object

    return await read_object(store, key, expected_digest=digest, expected_size=size)


def _asset_object_key(namespace: str, digest: str) -> str:
    return AssetObjectKeyFactory(namespace).key(digest)


__all__ = ["SqlAssetBackend", "build_asset_sql_metadata"]
