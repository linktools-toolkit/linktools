#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalized SQL backend for versioned raw Asset files."""

import base64
import binascii
import hashlib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, TypeVar

from linktools.core import environ

from ..core import validate_asset_namespace
from ..errors import AIError, ErrorCode
from ..storage import (
    MetadataLoad,
    SqlErrorKind,
    SqlSchemaRegistry,
    StorageBatchResult,
    StorageChange,
    StorageDeleteResult,
    StorageEntryRevision,
    StorageEntryStatus,
    StoragePutResult,
    StorageResetResult,
    StorageRevision,
    VersionSummary,
    classify_sql_error,
    resolve_dialect,
    sql_blob,
    sql_digest,
    sql_index,
    sql_integer_id,
    sql_table_options,
    sql_text_key,
    storage_name,
    get_sql_storage_context,
    register_sql_schema_contributor,
    validate_schema,
)
from ._backend import InMemoryAssetBackend
from ._domain import AssetInfo, AssetKey, AssetRoot

if TYPE_CHECKING:
    from sqlalchemy import Table
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession


_logger = environ.get_logger("ai.asset.sql")
_ResultT = TypeVar("_ResultT")
_SQL_BATCH_ROWS = 64


@dataclass(frozen=True, slots=True)
class SqlAssetTables:
    entry: "Table"
    change: "Table"
    blob: "Table"
    revision: "Table"


class SqlAssetSchema:
    """Register the SQL tables owned by Asset storage."""

    @classmethod
    def register_schema(cls, registry: SqlSchemaRegistry) -> SqlAssetTables:
        from sqlalchemy import (
            BigInteger,
            Column,
            DateTime,
            Index,
            String,
            Table,
            UniqueConstraint,
        )
        from sqlalchemy.sql import func

        metadata = registry.metadata
        integer_id = sql_integer_id()
        key_hash = sql_digest()
        namespace = sql_text_key(128)
        asset_kind = sql_text_key(128)
        asset_id = sql_text_key(512)

        def timestamps() -> tuple[Column, Column]:
            return (
                Column(
                    "updated_at",
                    DateTime(timezone=True),
                    nullable=False,
                    server_default=func.current_timestamp(),
                    onupdate=func.current_timestamp(),
                    comment="Last update timestamp",
                ),
                Column(
                    "created_at",
                    DateTime(timezone=True),
                    nullable=False,
                    server_default=func.current_timestamp(),
                    comment="Creation timestamp",
                ),
            )

        revision = Table(
            storage_name("asset_revision"),
            metadata,
            Column("id", integer_id, primary_key=True, autoincrement=True, comment="Surrogate primary key"),
            Column("namespace", namespace, nullable=False, comment="Asset namespace"),
            Column("store_revision", BigInteger, nullable=False, comment="Asset store revision"),
            *timestamps(),
            UniqueConstraint("namespace", name="uk_namespace"),
            **sql_table_options(),
            comment="Asset store revision counters",
        )
        entry = Table(
            storage_name("asset_entries"),
            metadata,
            Column("id", integer_id, primary_key=True, autoincrement=True, comment="Surrogate primary key"),
            Column("namespace", namespace, nullable=False, comment="Asset namespace"),
            Column("asset_key_hash", key_hash, nullable=False, comment="Asset key digest"),
            Column("asset_kind", asset_kind, nullable=False, comment="Asset kind"),
            Column("asset_id", asset_id, nullable=False, comment="Asset file identifier"),
            Column("entry_revision", BigInteger, nullable=False, comment="File entry revision"),
            Column("store_revision", BigInteger, nullable=False, comment="Store revision at file update"),
            Column("etag", key_hash, nullable=False, comment="File content digest"),
            Column("size", BigInteger, nullable=False, comment="File content size"),
            Column("status", String(16), nullable=False, comment="Current file status"),
            Column("blob_digest", key_hash, nullable=True, comment="Content blob digest"),
            Column("modified_at", DateTime(timezone=True), nullable=False, comment="File modification timestamp"),
            *timestamps(),
            UniqueConstraint("namespace", "asset_key_hash", name="uk_namespace_asset_key_hash"),
            sql_index(Index("ix_namespace_store_revision", "namespace", "store_revision")),
            **sql_table_options(),
            comment="Current Asset file entries",
        )
        change = Table(
            storage_name("asset_changes"),
            metadata,
            Column("id", integer_id, primary_key=True, autoincrement=True, comment="Surrogate primary key"),
            Column("namespace", namespace, nullable=False, comment="Asset namespace"),
            Column("asset_key_hash", key_hash, nullable=False, comment="Asset key digest"),
            Column("asset_kind", asset_kind, nullable=False, comment="Asset kind"),
            Column("asset_id", asset_id, nullable=False, comment="Asset file identifier"),
            Column("entry_revision", BigInteger, nullable=False, comment="File entry revision"),
            Column("store_revision", BigInteger, nullable=False, comment="Store revision at file update"),
            Column("etag", key_hash, nullable=False, comment="File content digest"),
            Column("size", BigInteger, nullable=False, comment="File content size"),
            Column("status", String(16), nullable=False, comment="File status at this history revision"),
            Column("blob_digest", key_hash, nullable=True, comment="Content blob digest"),
            Column("modified_at", DateTime(timezone=True), nullable=False, comment="File modification timestamp"),
            *timestamps(),
            sql_index(Index("ix_namespace_asset_key_hash_entry_revision", "namespace", "asset_key_hash", "entry_revision")),
            sql_index(Index("ix_namespace_store_revision", "namespace", "store_revision")),
            sql_index(Index("ix_blob_digest", "blob_digest")),
            **sql_table_options(),
            comment="Asset file change history",
        )
        blob = Table(
            storage_name("asset_blobs"),
            metadata,
            Column("id", integer_id, primary_key=True, autoincrement=True, comment="Surrogate primary key"),
            Column("digest", key_hash, nullable=False, comment="Content digest"),
            Column("content", sql_blob(), nullable=False, comment="File content"),
            *timestamps(),
            UniqueConstraint("digest", name="uk_digest"),
            **sql_table_options(),
            comment="Content-addressed Asset file blobs",
        )
        tables = SqlAssetTables(entry, change, blob, revision)
        for table in (entry, change, blob, revision):
            sql_index(Index("ix_updated_at", table.c.updated_at))
            sql_index(Index("ix_created_at", table.c.created_at))
            registry.add_table(table, owner="asset.sql")
        return tables


register_sql_schema_contributor("asset.sql", SqlAssetSchema.register_schema)


class SqlAssetBackend(InMemoryAssetBackend):
    """Persist current files, history, content blobs, and revisions separately."""

    def __init__(
        self,
        engine: "AsyncEngine",
        *,
        namespace: str,
    ) -> None:
        from sqlalchemy.ext.asyncio import AsyncEngine

        if not isinstance(engine, AsyncEngine):
            raise ValueError("SqlAssetBackend requires an AsyncEngine")
        try:
            validate_asset_namespace(namespace)
        except AIError as error:
            raise ValueError("SQL asset namespace is invalid") from error
        digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
        super().__init__(AssetRoot(f"sql:{digest[:16]}", "sql", namespace, digest))
        registry = SqlSchemaRegistry()
        self._tables = SqlAssetSchema.register_schema(registry)
        registry.freeze()
        self._engine = engine
        self._context = get_sql_storage_context(engine, namespace)
        self._session_factory = self._context.sessions
        self._namespace = namespace
        self._state_loaded = False

    async def initialize(self) -> None:
        if self._context.schema_manifest_digest is None:
            await validate_schema(self._engine, self._tables.revision.metadata)
        await self._refresh_state()
        _logger.debug(
            "SQL asset backend initialized: namespace=%s revision=%s",
            self._namespace,
            self._revision,
        )

    async def head_revision(self) -> StorageRevision:
        session = self._session_factory()
        async with session:
            revision = await self._load_revision(session)
        return StorageRevision(str(0 if revision is None else revision))

    async def load_metadata(
        self,
        after_revision: "StorageRevision | None",
    ) -> "MetadataLoad[AssetKey, AssetInfo]":
        await self._refresh_state()
        return await super().load_metadata(after_revision)

    async def get(self, key: AssetKey) -> "bytes | None":
        await self._refresh_state()
        return await super().get(key)

    async def get_many(self, keys: "Sequence[AssetKey]") -> "dict[AssetKey, bytes]":
        await self._refresh_state()
        return await super().get_many(keys)

    async def stat(self, key: AssetKey) -> "AssetInfo | None":
        await self._refresh_state()
        return await super().stat(key)

    async def list_versions(self, key: AssetKey) -> "tuple[VersionSummary, ...]":
        await self._refresh_state()
        return await super().list_versions(key)

    async def get_at_revision(
        self,
        key: AssetKey,
        entry_revision: StorageEntryRevision,
    ) -> "bytes | None":
        await self._refresh_state()
        return await super().get_at_revision(key, entry_revision)

    async def get_at_version(self, key: AssetKey, version: int) -> "bytes | None":
        await self._refresh_state()
        return await super().get_at_version(key, version)

    async def put(
        self,
        key: AssetKey,
        value: bytes,
        *,
        expected_entry_revision: "StorageEntryRevision | None" = None,
    ) -> "StoragePutResult[AssetInfo]":
        return await self._mutate(
            lambda backend: backend.put(
                key,
                value,
                expected_entry_revision=expected_entry_revision,
            )
        )

    async def delete(
        self,
        key: AssetKey,
        *,
        expected_entry_revision: "StorageEntryRevision | None" = None,
    ) -> "StorageDeleteResult[AssetKey]":
        return await self._mutate(
            lambda backend: backend.delete(
                key,
                expected_entry_revision=expected_entry_revision,
            )
        )

    async def reset(
        self,
        key: AssetKey,
        *,
        expected_entry_revision: "StorageEntryRevision | None" = None,
    ) -> "StorageResetResult[AssetKey]":
        return await self._mutate(
            lambda backend: backend.reset(
                key,
                expected_entry_revision=expected_entry_revision,
            )
        )

    async def apply_batch(
        self,
        changes: "Sequence[StorageChange[AssetKey, bytes]]",
        *,
        expected_revision: "StorageRevision | None" = None,
    ) -> "StorageBatchResult[AssetInfo, AssetKey]":
        return await self._mutate(
            lambda backend: backend.apply_batch(
                changes,
                expected_revision=expected_revision,
            )
        )

    async def _refresh_state(self) -> None:
        session = self._session_factory()
        async with session:
            for _ in range(3):
                revision = await self._load_revision(session)
                resolved_revision = 0 if revision is None else revision
                if self._state_loaded and resolved_revision == self._revision:
                    return
                raw = await self._load_state_for_revision(session, revision)
                if await self._load_revision(session) == revision:
                    break
            else:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        self.import_state(raw)
        self._state_loaded = True

    async def _mutate(
        self,
        mutation: "Callable[[InMemoryAssetBackend], Awaitable[_ResultT]]",
    ) -> _ResultT:
        from sqlalchemy import select

        session = self._session_factory()
        result: _ResultT
        next_state: dict[str, object]
        try:
            async with session:
                async with session.begin():
                    state_row = (
                        await session.execute(
                            select(self._tables.revision.c.store_revision)
                            .where(self._tables.revision.c.namespace == self._namespace)
                            .with_for_update()
                        )
                    ).first()
                    expected_revision = 0 if state_row is None else int(state_row[0])
                    previous_state = await self._load_state_for_revision(
                        session,
                        None if state_row is None else int(state_row[0]),
                    )
                    previous_versions = previous_state["versions"]
                    if not isinstance(previous_versions, list):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    backend = InMemoryAssetBackend(self.root)
                    backend.import_state(previous_state)
                    result = await mutation(backend)
                    next_state = backend.export_state()
                    next_revision = int(next_state["store_revision"])
                    if next_revision != expected_revision:
                        next_versions = next_state["versions"]
                        next_entries = next_state["entries"]
                        if (
                            not isinstance(next_versions, list)
                            or len(next_versions) < len(previous_versions)
                            or not isinstance(next_entries, list)
                        ):
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        await self._persist_changes(
                            session,
                            next_versions[len(previous_versions) :],
                            next_entries,
                        )
                        stored_revision = await resolve_dialect(session).upsert_increment(
                            session,
                            table=self._tables.revision,
                            values={"namespace": self._namespace},
                            column="store_revision",
                            index_elements=("namespace",),
                        )
                        if stored_revision != next_revision:
                            raise AIError(ErrorCode.STORAGE_CONFLICT)
        except AIError:
            raise
        except Exception as error:
            kind = classify_sql_error(error)
            if kind in {SqlErrorKind.INTEGRITY, SqlErrorKind.RETRYABLE_TRANSACTION}:
                raise AIError(ErrorCode.STORAGE_CONFLICT) from error
            if kind is SqlErrorKind.DATABASE:
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
            raise
        self.import_state(next_state)
        self._state_loaded = True
        _logger.debug(
            "SQL asset mutation committed: namespace=%s revision=%s",
            self._namespace,
            next_state["store_revision"],
        )
        return result

    async def _load_revision(self, session: "AsyncSession") -> "int | None":
        from sqlalchemy import select

        revision = await session.scalar(
            select(self._tables.revision.c.store_revision).where(
                self._tables.revision.c.namespace == self._namespace
            )
        )
        return None if revision is None else int(revision)

    async def _load_state_for_revision(
        self,
        session: "AsyncSession",
        revision: "int | None",
    ) -> dict[str, object]:
        from sqlalchemy import literal, select, union_all

        def record_select(table: "Table", source: str) -> object:
            return select(
                literal(source).label("_source"),
                table.c.id.label("_record_id"),
                table.c.asset_kind,
                table.c.asset_id,
                table.c.entry_revision,
                table.c.store_revision,
                table.c.etag,
                table.c.size,
                table.c.status,
                table.c.modified_at,
                self._tables.blob.c.content.label("content"),
            ).select_from(
                table.outerjoin(
                    self._tables.blob,
                    table.c.blob_digest == self._tables.blob.c.digest,
                )
            ).where(table.c.namespace == self._namespace)

        rows = (
            await session.execute(
                union_all(
                    record_select(self._tables.change, "change"),
                    record_select(self._tables.entry, "entry"),
                )
            )
        ).mappings().all()
        change_rows = sorted(
            (row for row in rows if row["_source"] == "change"),
            key=lambda row: int(row["_record_id"]),
        )
        entry_rows = sorted(
            (row for row in rows if row["_source"] == "entry"),
            key=lambda row: int(row["_record_id"]),
        )
        if revision is None:
            if change_rows or entry_rows:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "asset revision counter is missing")
            return {"store_revision": 0, "entries": [], "versions": []}
        return {
            "store_revision": int(revision),
            "entries": [_version_state(row) for row in entry_rows],
            "versions": [_version_state(row) for row in change_rows],
        }

    async def _persist_changes(
        self,
        session: "AsyncSession",
        changes: "Sequence[object]",
        entries: "Sequence[object]",
    ) -> None:
        from sqlalchemy import insert, select

        change_values: list[dict[str, object]] = []
        entry_values: list[dict[str, object]] = []
        blob_contents: dict[str, bytes] = {}
        changed_hashes: set[str] = set()
        for raw in changes:
            values, content = _change_values(raw, self._namespace)
            change_values.append(values)
            changed_hashes.add(str(values["asset_key_hash"]))
            digest = values["blob_digest"]
            if digest is not None:
                blob_contents[str(digest)] = content
        for raw in entries:
            values, content = _change_values(raw, self._namespace)
            if str(values["asset_key_hash"]) not in changed_hashes:
                continue
            entry_values.append(values)
            digest = values["blob_digest"]
            if digest is not None:
                blob_contents[str(digest)] = content
        dialect = resolve_dialect(session)
        if blob_contents:
            blob_values = tuple(
                {"digest": digest, "content": content}
                for digest, content in blob_contents.items()
            )
            blob_rows = []
            for offset in range(0, len(blob_values), _SQL_BATCH_ROWS):
                batch = blob_values[offset:offset + _SQL_BATCH_ROWS]
                await dialect.insert_ignore_conflict_many(
                    session,
                    table=self._tables.blob,
                    rows=batch,
                    index_elements=("digest",),
                )
                digests = tuple(str(values["digest"]) for values in batch)
                blob_rows.extend(
                    (
                        await session.execute(
                            select(self._tables.blob.c.digest, self._tables.blob.c.content).where(
                                self._tables.blob.c.digest.in_(digests)
                            )
                        )
                    )
                    .all()
                )
            if len(blob_rows) != len(blob_contents):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for digest, content in blob_rows:
                if hashlib.sha256(bytes(content)).hexdigest() != str(digest):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for offset in range(0, len(change_values), _SQL_BATCH_ROWS):
            await session.execute(
                insert(self._tables.change).values(
                    change_values[offset:offset + _SQL_BATCH_ROWS]
                )
            )
        for offset in range(0, len(entry_values), _SQL_BATCH_ROWS):
            await dialect.upsert_many(
                session,
                table=self._tables.entry,
                rows=entry_values[offset:offset + _SQL_BATCH_ROWS],
                set_columns=(
                    "asset_kind",
                    "asset_id",
                    "entry_revision",
                    "store_revision",
                    "etag",
                    "size",
                    "status",
                    "blob_digest",
                    "modified_at",
                ),
                index_elements=("namespace", "asset_key_hash"),
            )


def _version_state(row: object) -> dict[str, object]:
    values = row
    content = values["content"]
    try:
        status = StorageEntryStatus(str(values["status"]))
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "asset status is invalid") from error
    if status is not StorageEntryStatus.NORMAL:
        content_bytes = b""
    elif content is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "asset content blob is missing")
    else:
        content_bytes = bytes(content)
    etag = str(values["etag"])
    if len(content_bytes) != int(values["size"]) or hashlib.sha256(content_bytes).hexdigest() != etag:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "asset content blob is invalid")
    modified_at = values["modified_at"]
    if not isinstance(modified_at, datetime):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "asset modification timestamp is invalid")
    if modified_at.tzinfo is None:
        modified_at = modified_at.replace(tzinfo=timezone.utc)
    return {
        "kind": str(values["asset_kind"]),
        "id": str(values["asset_id"]),
        "revision": int(values["entry_revision"]),
        "store_revision": str(values["store_revision"]),
        "etag": etag,
        "size": int(values["size"]),
        "status": status.value,
        "modified_at": modified_at.astimezone(timezone.utc).isoformat(),
        "content": base64.b64encode(content_bytes).decode("ascii"),
    }


def _change_values(raw: object, namespace: str) -> tuple[dict[str, object], bytes]:
    if not isinstance(raw, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        kind = str(raw["kind"])
        asset_id = str(raw["id"])
        content = base64.b64decode(str(raw["content"]), validate=True)
        etag = str(raw["etag"])
        size = int(raw["size"])
        status = StorageEntryStatus(str(raw["status"]))
        modified_at = datetime.fromisoformat(str(raw["modified_at"]))
        entry_revision = int(raw["revision"])
        store_revision = int(str(raw["store_revision"]))
    except (KeyError, TypeError, ValueError, binascii.Error) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if modified_at.tzinfo is None:
        modified_at = modified_at.replace(tzinfo=timezone.utc)
    if status is not StorageEntryStatus.NORMAL:
        content = b""
    if size != len(content) or hashlib.sha256(content).hexdigest() != etag:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if status is not StorageEntryStatus.NORMAL and (size != 0 or etag != hashlib.sha256(b"").hexdigest()):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    key_hash = _asset_key_hash(AssetKey(kind, asset_id))
    values = {
        "namespace": namespace,
        "asset_key_hash": key_hash,
        "asset_kind": kind,
        "asset_id": asset_id,
        "entry_revision": entry_revision,
        "store_revision": store_revision,
        "etag": etag,
        "size": size,
        "status": status.value,
        "blob_digest": None if status is not StorageEntryStatus.NORMAL else etag,
        "modified_at": modified_at.astimezone(timezone.utc),
    }
    return values, content


def _asset_key_hash(key: AssetKey) -> str:
    return hashlib.sha256(f"{key.kind}\0{key.id}".encode("utf-8")).hexdigest()


__all__ = [
    "SqlAssetBackend",
    "SqlAssetSchema",
    "SqlAssetTables",
]
