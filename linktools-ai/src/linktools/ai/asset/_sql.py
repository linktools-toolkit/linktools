#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Namespace-scoped SQL AssetStore backend with state-row CAS."""

import hashlib
import json
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeVar

from linktools.core import environ

from ..errors import AIError, ErrorCode
from ..storage import SqlSchemaRegistry, storage_name
from ._backend import InMemoryAssetBackend
from ._domain import (
    AssetDeleteResult,
    AssetEntryBatchResult,
    AssetEntryChange,
    AssetEntryDeleteResult,
    AssetEntryInfo,
    AssetEntryKey,
    AssetEntryRevision,
    AssetEntrySnapshot,
    AssetEntryVersion,
    AssetInfo,
    AssetKey,
    AssetRevision,
    AssetRoot,
    AssetStoreRevision,
    AssetVersion,
)

if TYPE_CHECKING:
    from sqlalchemy import Table
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker


_logger = environ.get_logger("ai.asset.sql")
_STATE_ROW = "asset_state"
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class SqlAssetTables:
    root: "Table"
    entry: "Table"
    version: "Table"
    blob: "Table"
    state: "Table"


class SqlAssetBackend(InMemoryAssetBackend):
    """Persist each namespace through one transactional CAS state row."""

    _registered_tables: SqlAssetTables | None = None

    @classmethod
    def register_schema(cls, registry: SqlSchemaRegistry) -> SqlAssetTables:
        from sqlalchemy import (
            BigInteger,
            Boolean,
            Column,
            LargeBinary,
            String,
            Table,
            UniqueConstraint,
        )

        metadata = registry.metadata
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
        state = Table(
            storage_name(_STATE_ROW),
            metadata,
            Column("namespace", String(128), primary_key=True),
            Column("store_revision", BigInteger, nullable=False),
            Column("payload", LargeBinary, nullable=False),
        )
        tables = SqlAssetTables(root, entry, version, blob, state)
        for table in (root, entry, version, blob, state):
            registry.add_table(table, owner="asset.sql")
        cls._registered_tables = tables
        return tables

    def __init__(self, session_factory: "async_sessionmaker[AsyncSession]", *, namespace: str, dialect: "object | None" = None) -> None:
        del dialect
        if not isinstance(namespace, str) or not namespace.strip() or any(ord(character) < 32 or ord(character) == 127 for character in namespace):
            raise ValueError("SQL asset namespace is invalid")
        digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
        super().__init__(AssetRoot(f"sql:{digest[:16]}", "sql", namespace, digest))
        self._session_factory = session_factory
        self._tables = self._registered_tables
        if self._tables is None:
            raise ValueError("SqlAssetBackend schema is not registered")
        self._namespace = namespace

    async def initialize(self) -> None:
        await self._refresh_state()
        _logger.info("SQL asset namespace initialized: namespace=%s revision=%s", self._namespace, self._store_revision)

    async def initialize_storage(self, engine: "AsyncEngine | None" = None) -> None:
        if engine is not None:
            metadata = self._tables.root.metadata
            async with engine.begin() as connection:
                await connection.run_sync(metadata.create_all)
        await self.initialize()

    async def current_revision(self) -> AssetStoreRevision:
        await self._refresh_state()
        return await InMemoryAssetBackend.current_revision(self)

    async def stat_asset(self, key: AssetKey) -> "AssetInfo | None":
        await self._refresh_state()
        return await InMemoryAssetBackend.stat_asset(self, key)

    async def list_assets(self) -> "tuple[AssetInfo, ...]":
        await self._refresh_state()
        return await InMemoryAssetBackend.list_assets(self)

    async def list_asset_versions(self, key: AssetKey) -> "tuple[AssetVersion, ...]":
        await self._refresh_state()
        return await InMemoryAssetBackend.list_asset_versions(self, key)

    async def asset_revision_files(self, key: AssetKey, revision: AssetRevision) -> "tuple[AssetEntrySnapshot, ...]":
        await self._refresh_state()
        return await InMemoryAssetBackend.asset_revision_files(self, key, revision)

    async def current_file(self, key: AssetEntryKey, *, include_deleted: bool = False) -> "tuple[AssetEntryInfo, bytes] | None":
        await self._refresh_state()
        return await InMemoryAssetBackend.current_file(self, key, include_deleted=include_deleted)

    async def list_current_files(self, asset: AssetKey, *, prefix: "str | None", include_deleted: bool) -> "tuple[tuple[AssetEntryInfo, bytes], ...]":
        await self._refresh_state()
        return await InMemoryAssetBackend.list_current_files(self, asset, prefix=prefix, include_deleted=include_deleted)

    async def list_file_versions(self, key: AssetEntryKey) -> "tuple[AssetEntryVersion, ...]":
        await self._refresh_state()
        return await InMemoryAssetBackend.list_file_versions(self, key)

    async def get_file_at_revision(self, key: AssetEntryKey, revision: AssetEntryRevision) -> "bytes | None":
        await self._refresh_state()
        return await InMemoryAssetBackend.get_file_at_revision(self, key, revision)

    async def snapshot_files(self, asset: AssetKey, revision: "AssetRevision | None", include_deleted: bool) -> "tuple[AssetEntrySnapshot, ...]":
        await self._refresh_state()
        return await InMemoryAssetBackend.snapshot_files(self, asset, revision, include_deleted)

    async def apply_file_batch(
        self,
        asset: AssetKey,
        changes: Sequence[AssetEntryChange],
        *,
        primary_path: str,
        expected_revision: "AssetRevision | None",
        expected_store_revision: "AssetStoreRevision | None",
    ) -> AssetEntryBatchResult:
        return await self._mutate_state(
            lambda state: state.apply_file_batch(asset, changes, primary_path=primary_path, expected_revision=expected_revision, expected_store_revision=expected_store_revision)
        )

    async def apply_asset_batch(
        self,
        changes: "Sequence[tuple[AssetKey, bytes | None, str, Literal['PUT', 'DELETE'], AssetRevision | None]]",
        *,
        expected_store_revision: "AssetStoreRevision | None",
    ) -> "tuple[AssetInfo | AssetDeleteResult, ...]":
        return await self._mutate_state(lambda state: state.apply_asset_batch(changes, expected_store_revision=expected_store_revision))

    async def delete_file(
        self,
        key: AssetEntryKey,
        *,
        primary_path: str,
        expected_entry_revision: "AssetEntryRevision | None",
        expected_revision: "AssetRevision | None",
    ) -> AssetEntryDeleteResult:
        return await self._mutate_state(
            lambda state: state.delete_file(key, primary_path=primary_path, expected_entry_revision=expected_entry_revision, expected_revision=expected_revision)
        )

    async def replace_tree(
        self,
        asset: AssetKey,
        files: Mapping[str, bytes],
        *,
        deleted_rel_paths: Collection[str],
        primary_path: str,
        expected_revision: "AssetRevision | None",
    ) -> AssetInfo:
        return await self._mutate_state(
            lambda state: state.replace_tree(asset, files, deleted_rel_paths=deleted_rel_paths, primary_path=primary_path, expected_revision=expected_revision)
        )

    async def delete_asset(self, key: AssetKey, *, expected_revision: "AssetRevision | None") -> "tuple[bool, AssetInfo | None]":
        return await self._mutate_state(lambda state: state.delete_asset(key, expected_revision=expected_revision))

    async def restore_asset(self, key: AssetKey, revision: AssetRevision, *, expected_revision: AssetRevision) -> AssetInfo:
        return await self._mutate_state(lambda state: state.restore_asset(key, revision, expected_revision=expected_revision))

    async def rename_asset(self, source: AssetKey, target: AssetKey, *, expected_source_revision: AssetRevision) -> AssetInfo:
        return await self._mutate_state(lambda state: state.rename_asset(source, target, expected_source_revision=expected_source_revision))

    async def sync_sources(self, source_files: "Mapping[AssetKey, Mapping[str, tuple[bytes, str]]]", primary_paths: "Mapping[str, str]") -> AssetStoreRevision:
        return await self._mutate_state(lambda state: state.sync_sources(source_files, primary_paths))

    async def _refresh_state(self) -> None:
        session = self._session_factory()
        async with session:
            from sqlalchemy import select

            row = (
                await session.execute(
                    select(self._tables.state.c.store_revision, self._tables.state.c.payload).where(self._tables.state.c.namespace == self._namespace)
                )
            ).first()
        if row is None:
            self.import_state({"store_revision": 0, "assets": []})
            return
        raw = json.loads(bytes(row[1]).decode("utf-8"))
        if not isinstance(raw, dict) or int(raw.get("store_revision", -1)) != int(row[0]):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "SQL asset state revision mismatch")
        self.import_state(raw)

    async def _mutate_state(self, mutation: "Callable[[InMemoryAssetBackend], Awaitable[_T]]") -> _T:
        session = self._session_factory()
        if session is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        from sqlalchemy import insert, select, update
        from sqlalchemy.exc import IntegrityError, OperationalError

        result: _T
        next_state: dict[str, object]
        try:
            async with session:
                async with session.begin():
                    row = (
                        await session.execute(
                            select(self._tables.state.c.store_revision, self._tables.state.c.payload)
                            .where(self._tables.state.c.namespace == self._namespace)
                            .with_for_update()
                        )
                    ).first()
                    expected_revision = 0 if row is None else int(row[0])
                    state = InMemoryAssetBackend(self.root)
                    if row is not None:
                        raw = json.loads(bytes(row[1]).decode("utf-8"))
                        if not isinstance(raw, dict) or int(raw.get("store_revision", -1)) != expected_revision:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "SQL asset state revision mismatch")
                        state.import_state(raw)
                    result = await mutation(state)
                    next_state = state.export_state()
                    next_revision = int(next_state["store_revision"])
                    if next_revision != expected_revision:
                        payload = json.dumps(next_state, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                        if row is None:
                            await session.execute(insert(self._tables.state).values(namespace=self._namespace, store_revision=next_revision, payload=payload))
                        else:
                            updated = await session.execute(
                                update(self._tables.state)
                                .where(self._tables.state.c.namespace == self._namespace, self._tables.state.c.store_revision == expected_revision)
                                .values(store_revision=next_revision, payload=payload)
                            )
                            if updated.rowcount != 1:
                                raise AIError(ErrorCode.STORAGE_CONFLICT)
        except AIError:
            raise
        except (IntegrityError, OperationalError) as error:
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error
        self.import_state(next_state)
        _logger.info("SQL asset mutation committed: namespace=%s revision=%s", self._namespace, next_state["store_revision"])
        return result


__all__ = ["SqlAssetBackend", "SqlAssetTables"]
