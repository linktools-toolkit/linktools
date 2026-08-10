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

from ..errors import AIError, ErrorCode
from ..storage import (
    MetadataLoad,
    SqlSchemaRegistry,
    StorageBatchResult,
    StorageChange,
    StorageDeleteResult,
    StorageEntryRevision,
    StoragePutResult,
    StorageResetResult,
    StorageRevision,
    VersionSummary,
    storage_name,
    validate_schema,
)
from ._backend import InMemoryAssetBackend
from ._domain import AssetInfo, AssetKey, AssetRoot

if TYPE_CHECKING:
    from sqlalchemy import Index, Table
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


_logger = environ.get_logger("ai.asset.sql")
_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True, slots=True)
class SqlAssetTables:
    entry: "Table"
    change: "Table"
    blob: "Table"
    revision: "Table"


def _mysql_index(index: "Index") -> "Index":
    index.info["ddl_dialect"] = "mysql"
    return index.ddl_if(dialect="mysql")


class SqlAssetBackend(InMemoryAssetBackend):
    """Persist current files, history, content blobs, and revisions separately."""

    _registered_tables: "SqlAssetTables | None" = None

    @classmethod
    def register_schema(cls, registry: SqlSchemaRegistry) -> SqlAssetTables:
        from sqlalchemy import (
            BigInteger,
            Boolean,
            CHAR,
            Column,
            DateTime,
            Index,
            Integer,
            LargeBinary,
            String,
            Table,
            UniqueConstraint,
        )
        from sqlalchemy.dialects import mysql
        from sqlalchemy.sql import func

        metadata = registry.metadata
        integer_id = BigInteger().with_variant(Integer, "sqlite")
        key_hash = CHAR(64).with_variant(mysql.CHAR(64, charset="utf8mb4", collation="utf8mb4_bin"), "mysql")
        namespace = String(128).with_variant(mysql.VARCHAR(128, charset="utf8mb4", collation="utf8mb4_bin"), "mysql")
        asset_kind = String(128).with_variant(mysql.VARCHAR(128, charset="utf8mb4", collation="utf8mb4_bin"), "mysql")
        asset_id = String(512).with_variant(mysql.VARCHAR(512, charset="utf8mb4", collation="utf8mb4_bin"), "mysql")

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
            mysql_engine="InnoDB",
            mysql_charset="utf8mb4",
            mysql_collate="utf8mb4_bin",
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
            Column("deleted", Boolean, nullable=False, comment="Whether the current file is deleted"),
            Column("blob_digest", key_hash, nullable=True, comment="Content blob digest"),
            Column("modified_at", DateTime(timezone=True), nullable=False, comment="File modification timestamp"),
            *timestamps(),
            UniqueConstraint("namespace", "asset_key_hash", name="uk_namespace_asset_key_hash"),
            _mysql_index(Index("ix_namespace_store_revision", "namespace", "store_revision")),
            mysql_engine="InnoDB",
            mysql_charset="utf8mb4",
            mysql_collate="utf8mb4_bin",
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
            Column("deleted", Boolean, nullable=False, comment="Whether this history row is a tombstone"),
            Column("blob_digest", key_hash, nullable=True, comment="Content blob digest"),
            Column("modified_at", DateTime(timezone=True), nullable=False, comment="File modification timestamp"),
            *timestamps(),
            _mysql_index(Index("ix_namespace_asset_key_hash_entry_revision", "namespace", "asset_key_hash", "entry_revision")),
            _mysql_index(Index("ix_namespace_store_revision", "namespace", "store_revision")),
            _mysql_index(Index("ix_blob_digest", "blob_digest")),
            mysql_engine="InnoDB",
            mysql_charset="utf8mb4",
            mysql_collate="utf8mb4_bin",
            comment="Asset file change history",
        )
        blob = Table(
            storage_name("asset_blobs"),
            metadata,
            Column("id", integer_id, primary_key=True, autoincrement=True, comment="Surrogate primary key"),
            Column("digest", key_hash, nullable=False, comment="Content digest"),
            Column("content", LargeBinary().with_variant(mysql.LONGBLOB(), "mysql"), nullable=False, comment="File content"),
            *timestamps(),
            UniqueConstraint("digest", name="uk_digest"),
            mysql_engine="InnoDB",
            mysql_charset="utf8mb4",
            mysql_collate="utf8mb4_bin",
            comment="Content-addressed Asset file blobs",
        )
        tables = SqlAssetTables(entry, change, blob, revision)
        for table in (entry, change, blob, revision):
            _mysql_index(Index("ix_updated_at", table.c.updated_at))
            _mysql_index(Index("ix_created_at", table.c.created_at))
            registry.add_table(table, owner="asset.sql")
        cls._registered_tables = tables
        return tables

    def __init__(
        self,
        session_factory: "async_sessionmaker[AsyncSession]",
        *,
        namespace: str,
    ) -> None:
        if (
            not isinstance(namespace, str)
            or not namespace.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in namespace)
        ):
            raise ValueError("SQL asset namespace is invalid")
        digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
        super().__init__(AssetRoot(f"sql:{digest[:16]}", "sql", namespace, digest))
        if self._registered_tables is None:
            raise ValueError("SqlAssetBackend schema is not registered")
        self._session_factory = session_factory
        self._tables = self._registered_tables
        self._namespace = namespace

    async def initialize(self) -> None:
        await self._validate_schema()
        await self._refresh_state()
        _logger.info(
            "SQL asset backend initialized: namespace=%s revision=%s",
            self._namespace,
            self._revision,
        )

    async def initialize_storage(self) -> None:
        await self.initialize()

    async def _validate_schema(self) -> None:
        await validate_schema(self._session_factory, self._tables.revision.metadata)

    async def head_revision(self) -> StorageRevision:
        await self._refresh_state()
        return await super().head_revision()

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

    async def reset(self) -> StorageResetResult:
        return await self._mutate(lambda backend: backend.reset())

    async def apply_batch(
        self,
        changes: "Sequence[StorageChange[AssetKey, bytes]]",
        *,
        expected_store_revision: "StorageRevision | None" = None,
    ) -> "StorageBatchResult[AssetInfo, AssetKey]":
        return await self._mutate(
            lambda backend: backend.apply_batch(
                changes,
                expected_store_revision=expected_store_revision,
            )
        )

    async def _refresh_state(self) -> None:
        session = self._session_factory()
        async with session:
            raw = await self._load_state(session)
        self.import_state(raw)

    async def _mutate(
        self,
        mutation: "Callable[[InMemoryAssetBackend], Awaitable[_ResultT]]",
    ) -> _ResultT:
        from sqlalchemy import insert, select, update
        from sqlalchemy.exc import IntegrityError, OperationalError

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
                    previous_state = await self._load_state(session)
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
                        if state_row is None:
                            await session.execute(
                                insert(self._tables.revision).values(
                                    namespace=self._namespace,
                                    store_revision=next_revision,
                                )
                            )
                        else:
                            updated = await session.execute(
                                update(self._tables.revision)
                                .where(
                                    self._tables.revision.c.namespace == self._namespace,
                                    self._tables.revision.c.store_revision == expected_revision,
                                )
                                .values(store_revision=next_revision)
                            )
                            if updated.rowcount != 1:
                                raise AIError(ErrorCode.STORAGE_CONFLICT)
        except AIError:
            raise
        except (IntegrityError, OperationalError) as error:
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error
        self.import_state(next_state)
        _logger.info(
            "SQL asset mutation committed: namespace=%s revision=%s",
            self._namespace,
            next_state["store_revision"],
        )
        return result

    async def _load_state(self, session: "AsyncSession") -> dict[str, object]:
        from sqlalchemy import select

        revision = await session.scalar(
            select(self._tables.revision.c.store_revision).where(
                self._tables.revision.c.namespace == self._namespace
            )
        )
        change_rows = await self._load_records(session, self._tables.change)
        entry_rows = await self._load_records(session, self._tables.entry)
        if revision is None:
            if change_rows or entry_rows:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "asset revision counter is missing")
            return {"store_revision": 0, "entries": [], "versions": []}
        return {
            "store_revision": int(revision),
            "entries": [_version_state(row) for row in entry_rows],
            "versions": [_version_state(row) for row in change_rows],
        }

    async def _load_records(self, session: "AsyncSession", table: "Table") -> list[object]:
        from sqlalchemy import select

        return list(
            (
                await session.execute(
                    select(
                        table.c.asset_kind,
                        table.c.asset_id,
                        table.c.entry_revision,
                        table.c.store_revision,
                        table.c.etag,
                        table.c.size,
                        table.c.deleted,
                        table.c.modified_at,
                        self._tables.blob.c.content.label("content"),
                    )
                    .select_from(
                        table.outerjoin(
                            self._tables.blob,
                            table.c.blob_digest == self._tables.blob.c.digest,
                        )
                    )
                    .where(table.c.namespace == self._namespace)
                    .order_by(table.c.id)
                )
            )
            .mappings()
            .all()
        )

    async def _persist_changes(
        self,
        session: "AsyncSession",
        changes: "Sequence[object]",
        entries: "Sequence[object]",
    ) -> None:
        from sqlalchemy import delete, insert, select, update

        for raw in changes:
            values, content = _change_values(raw, self._namespace)
            if values["blob_digest"] is not None:
                await self._ensure_blob(session, values["blob_digest"], content)
            await session.execute(insert(self._tables.change).values(**values))
        entry_values: list[dict[str, object]] = []
        for raw in entries:
            values, content = _change_values(raw, self._namespace)
            if values["blob_digest"] is not None:
                await self._ensure_blob(session, values["blob_digest"], content)
            entry_values.append(values)
        desired_hashes = {str(values["asset_key_hash"]) for values in entry_values}
        current_rows = (
            await session.execute(
                select(self._tables.entry.c.id, self._tables.entry.c.asset_key_hash).where(
                    self._tables.entry.c.namespace == self._namespace
                )
            )
        ).all()
        for current_id, key_hash in current_rows:
            if str(key_hash) not in desired_hashes:
                await session.execute(delete(self._tables.entry).where(self._tables.entry.c.id == current_id))
        for values in entry_values:
            current_id = await session.scalar(
                select(self._tables.entry.c.id).where(
                    self._tables.entry.c.namespace == self._namespace,
                    self._tables.entry.c.asset_key_hash == values["asset_key_hash"],
                )
            )
            if current_id is None:
                await session.execute(insert(self._tables.entry).values(**values))
            else:
                await session.execute(
                    update(self._tables.entry)
                    .where(self._tables.entry.c.id == current_id)
                    .values(**values)
                )

    async def _ensure_blob(
        self,
        session: "AsyncSession",
        digest: object,
        content: bytes,
    ) -> None:
        from sqlalchemy import insert, select
        from sqlalchemy.exc import IntegrityError

        existing = await session.scalar(
            select(self._tables.blob.c.content).where(self._tables.blob.c.digest == digest)
        )
        if existing is not None:
            if hashlib.sha256(bytes(existing)).hexdigest() != str(digest):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        try:
            async with session.begin_nested():
                await session.execute(
                    insert(self._tables.blob).values(digest=digest, content=content)
                )
        except IntegrityError:
            existing = await session.scalar(
                select(self._tables.blob.c.content).where(self._tables.blob.c.digest == digest)
            )
            if existing is None or hashlib.sha256(bytes(existing)).hexdigest() != str(digest):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _version_state(row: object) -> dict[str, object]:
    values = row
    content = values["content"]
    deleted = bool(values["deleted"])
    if deleted:
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
        "deleted": deleted,
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
        deleted = bool(raw["deleted"])
        modified_at = datetime.fromisoformat(str(raw["modified_at"]))
        entry_revision = int(raw["revision"])
        store_revision = int(str(raw["store_revision"]))
    except (KeyError, TypeError, ValueError, binascii.Error) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if modified_at.tzinfo is None:
        modified_at = modified_at.replace(tzinfo=timezone.utc)
    if deleted:
        content = b""
    if size != len(content) or hashlib.sha256(content).hexdigest() != etag:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if deleted and (size != 0 or etag != hashlib.sha256(b"").hexdigest()):
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
        "deleted": deleted,
        "blob_digest": None if deleted else etag,
        "modified_at": modified_at.astimezone(timezone.utc),
    }
    return values, content


def _asset_key_hash(key: AssetKey) -> str:
    return hashlib.sha256(f"{key.kind}\0{key.id}".encode("utf-8")).hexdigest()


__all__ = ["SqlAssetBackend", "SqlAssetTables"]
