#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL backend for the unified AssetStore tree ledger."""

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from linktools.core import environ

from ..storage import SqlSchemaRegistry, storage_name
from ._backend import InMemoryAssetBackend
from ._domain import (
    AssetDeleteResult,
    AssetEntryBatchResult,
    AssetEntryChange,
    AssetEntryDeleteResult,
    AssetEntryKey,
    AssetEntryRevision,
    AssetInfo,
    AssetKey,
    AssetRevision,
    AssetRoot,
    AssetStoreRevision,
)

if TYPE_CHECKING:
    from sqlalchemy import Table
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker


_logger = environ.get_logger("ai.asset.sql")
_STATE_DIGEST = hashlib.sha256(b"linktools.asset-tree.state").hexdigest()


@dataclass(frozen=True, slots=True)
class SqlAssetTables:
    root: "Table"
    entry: "Table"
    version: "Table"
    blob: "Table"


class SqlAssetBackend(InMemoryAssetBackend):
    """Persist the atomic tree ledger through the configured async SQL session."""

    _registered_tables: SqlAssetTables | None = None

    @classmethod
    def register_schema(cls, registry: SqlSchemaRegistry) -> SqlAssetTables:
        from sqlalchemy import (
            BigInteger,
            Boolean,
            Column,
            LargeBinary,
            MetaData,
            String,
            Table,
            UniqueConstraint,
        )

        metadata: MetaData = registry.metadata
        root = Table(
            storage_name("asset_revision"),
            metadata,
            Column("namespace", String(128), nullable=False),
            Column("asset_kind", String(128), nullable=False),
            Column("asset_id", String(512), nullable=False),
            Column("revision", BigInteger, nullable=False),
            Column("etag", String(64), nullable=False),
            Column("deleted", Boolean, nullable=False),
            UniqueConstraint("namespace", "asset_kind", "asset_id", "revision", name="uq_asset_revision"),
        )
        entry = Table(
            storage_name("asset_contents"),
            metadata,
            Column("namespace", String(128), nullable=False),
            Column("asset_kind", String(128), nullable=False),
            Column("asset_id", String(512), nullable=False),
            Column("rel_path", String(2048), nullable=False),
            Column("entry_revision", BigInteger, nullable=False),
            Column("etag", String(64), nullable=False),
            Column("deleted", Boolean, nullable=False),
            UniqueConstraint("namespace", "asset_kind", "asset_id", "rel_path", "entry_revision", name="uq_asset_entry_revision"),
        )
        version = Table(
            storage_name("asset_changes"),
            metadata,
            Column("namespace", String(128), nullable=False),
            Column("asset_kind", String(128), nullable=False),
            Column("asset_id", String(512), nullable=False),
            Column("rel_path", String(2048), nullable=False),
            Column("version", BigInteger, nullable=False),
            Column("etag", String(64), nullable=False),
            Column("deleted", Boolean, nullable=False),
            UniqueConstraint("namespace", "asset_kind", "asset_id", "rel_path", "version", name="uq_asset_change"),
        )
        blob = Table(
            storage_name("asset_blobs"),
            metadata,
            Column("digest", String(64), nullable=False),
            Column("content", LargeBinary, nullable=False),
            UniqueConstraint("digest", name="uq_asset_blob_digest"),
        )
        tables = SqlAssetTables(root, entry, version, blob)
        for table in (root, entry, version, blob):
            registry.add_table(table, owner="asset.sql")
        cls._registered_tables = tables
        return tables

    def __init__(self, session_factory: "async_sessionmaker[AsyncSession]", *, dialect: "object | None" = None) -> None:
        del dialect
        super().__init__(AssetRoot("sql:default", "sql", "sql", "sql"))
        self._session_factory = session_factory
        self._tables = self._registered_tables

    async def initialize_storage(self, engine: "AsyncEngine | None" = None) -> None:
        if engine is not None and self._tables is not None:
            metadata = self._tables.root.metadata
            async with engine.begin() as connection:
                await connection.run_sync(metadata.create_all)
        await self.initialize()
        await self._load_persisted_state()
        _logger.info("SQL asset tree initialized: revision=%s", self._store_revision)

    async def apply_file_batch(
        self,
        asset: AssetKey,
        changes: Sequence[AssetEntryChange],
        *,
        primary_path: str,
        expected_revision: "AssetRevision | None",
        expected_store_revision: "AssetStoreRevision | None",
    ) -> AssetEntryBatchResult:
        await self._load_persisted_state()
        result = await super().apply_file_batch(asset, changes, primary_path=primary_path, expected_revision=expected_revision, expected_store_revision=expected_store_revision)
        await self._persist_state()
        return result

    async def apply_asset_batch(
        self,
        changes: "Sequence[tuple[AssetKey, bytes | None, str, Literal['PUT', 'DELETE'], AssetRevision | None]]",
        *,
        expected_store_revision: "AssetStoreRevision | None",
    ) -> "tuple[AssetInfo | AssetDeleteResult, ...]":
        await self._load_persisted_state()
        result = await super().apply_asset_batch(changes, expected_store_revision=expected_store_revision)
        await self._persist_state()
        return result

    async def delete_file(
        self,
        key: AssetEntryKey,
        *,
        primary_path: str,
        expected_entry_revision: "AssetEntryRevision | None",
        expected_revision: "AssetRevision | None",
    ) -> AssetEntryDeleteResult:
        await self._load_persisted_state()
        result = await super().delete_file(key, primary_path=primary_path, expected_entry_revision=expected_entry_revision, expected_revision=expected_revision)
        await self._persist_state()
        return result

    async def replace_tree(
        self,
        asset: AssetKey,
        files: Mapping[str, bytes],
        *,
        deleted_rel_paths: Collection[str],
        primary_path: str,
        expected_revision: "AssetRevision | None",
    ) -> AssetInfo:
        await self._load_persisted_state()
        result = await super().replace_tree(asset, files, deleted_rel_paths=deleted_rel_paths, primary_path=primary_path, expected_revision=expected_revision)
        await self._persist_state()
        return result

    async def delete_asset(self, key: AssetKey, *, expected_revision: "AssetRevision | None") -> "tuple[bool, AssetInfo | None]":
        await self._load_persisted_state()
        result = await super().delete_asset(key, expected_revision=expected_revision)
        await self._persist_state()
        return result

    async def restore_asset(self, key: AssetKey, revision: AssetRevision, *, expected_revision: AssetRevision) -> AssetInfo:
        await self._load_persisted_state()
        result = await super().restore_asset(key, revision, expected_revision=expected_revision)
        await self._persist_state()
        return result

    async def rename_asset(self, source: AssetKey, target: AssetKey, *, expected_source_revision: AssetRevision) -> AssetInfo:
        await self._load_persisted_state()
        result = await super().rename_asset(source, target, expected_source_revision=expected_source_revision)
        await self._persist_state()
        return result

    async def sync_sources(self, source_files: "Mapping[AssetKey, Mapping[str, tuple[bytes, str]]]", primary_paths: "Mapping[str, str]") -> AssetStoreRevision:
        await self._load_persisted_state()
        result = await super().sync_sources(source_files, primary_paths)
        await self._persist_state()
        return result

    async def _load_persisted_state(self) -> None:
        if self._tables is None or self._session_factory is None:
            return
        session = self._session_factory()
        if session is None:
            return
        from sqlalchemy import select

        async with session:
            row = (await session.execute(select(self._tables.blob.c.content).where(self._tables.blob.c.digest == _STATE_DIGEST))).first()
        if row is None:
            return
        raw = json.loads(bytes(row[0]).decode("utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("SQL asset state is invalid")
        super()._load_state(raw)

    async def _persist_state(self) -> None:
        if self._tables is None or self._session_factory is None:
            return
        session = self._session_factory()
        if session is None:
            return
        from sqlalchemy import delete, insert

        payload = json.dumps(self._dump_state(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        async with session, session.begin():
                for table in (self._tables.root, self._tables.entry, self._tables.version):
                    await session.execute(delete(table))
                await session.execute(delete(self._tables.blob).where(self._tables.blob.c.digest == _STATE_DIGEST))
                root_rows = [
                    {
                        "namespace": self.root.root_id,
                        "asset_kind": manifest.info.key.kind,
                        "asset_id": manifest.info.key.id,
                        "revision": manifest.info.revision.value,
                        "etag": manifest.info.etag,
                        "deleted": manifest.info.deleted,
                    }
                    for histories in self._asset_history.values()
                    for manifest in histories.values()
                ]
                entry_rows = [
                    {
                        "namespace": self.root.root_id,
                        "asset_kind": info.key.asset.kind,
                        "asset_id": info.key.asset.id,
                        "rel_path": info.key.rel_path,
                        "entry_revision": info.entry_revision.value,
                        "etag": info.etag,
                        "deleted": info.deleted,
                    }
                    for info in self._entries.values()
                ]
                version_rows = [
                    {
                        "namespace": self.root.root_id,
                        "asset_kind": key.asset.kind,
                        "asset_id": key.asset.id,
                        "rel_path": key.rel_path,
                        "version": version.entry_revision.value,
                        "etag": version.etag,
                        "deleted": version.deleted,
                    }
                    for key, history in self._entry_history.items()
                    for version, _content in history.values()
                ]
                if root_rows:
                    await session.execute(insert(self._tables.root), root_rows)
                if entry_rows:
                    await session.execute(insert(self._tables.entry), entry_rows)
                if version_rows:
                    await session.execute(insert(self._tables.version), version_rows)
                await session.execute(insert(self._tables.blob).values(digest=_STATE_DIGEST, content=payload))


__all__ = ["SqlAssetBackend", "SqlAssetTables"]
