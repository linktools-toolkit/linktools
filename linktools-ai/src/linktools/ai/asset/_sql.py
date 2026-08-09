#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL backend for versioned Asset files."""

import hashlib
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
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
)
from ._backend import InMemoryAssetBackend
from ._domain import AssetInfo, AssetKey, AssetRoot

if TYPE_CHECKING:
    from sqlalchemy import Table
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

_logger = environ.get_logger("ai.asset.sql")
_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True, slots=True)
class SqlAssetTables:
    state: "Table"


class SqlAssetBackend(InMemoryAssetBackend):
    """Persist a namespace-scoped Asset file ledger in one transactional row."""

    _registered_tables: "SqlAssetTables | None" = None

    @classmethod
    def register_schema(cls, registry: SqlSchemaRegistry) -> SqlAssetTables:
        from sqlalchemy import BigInteger, Column, LargeBinary, String, Table

        state = Table(
            storage_name("asset_state"),
            registry.metadata,
            Column("namespace", String(128), primary_key=True),
            Column("store_revision", BigInteger, nullable=False),
            Column("payload", LargeBinary, nullable=False),
        )
        registry.add_table(state, owner="asset.sql")
        tables = SqlAssetTables(state)
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
        await self._refresh_state()
        _logger.info(
            "SQL asset backend initialized: namespace=%s revision=%s",
            self._namespace,
            self._revision,
        )

    async def initialize_storage(self, engine: "AsyncEngine | None" = None) -> None:
        if engine is not None:
            async with engine.begin() as connection:
                await connection.run_sync(self._tables.state.metadata.create_all)
        await self.initialize()

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
        from sqlalchemy import select

        session = self._session_factory()
        async with session:
            row = (
                await session.execute(
                    select(
                        self._tables.state.c.store_revision,
                        self._tables.state.c.payload,
                    ).where(self._tables.state.c.namespace == self._namespace)
                )
            ).first()
        if row is None:
            self.import_state({"store_revision": 0, "versions": []})
            return
        raw = json.loads(bytes(row[1]).decode("utf-8"))
        if not isinstance(raw, dict) or int(raw.get("store_revision", -1)) != int(row[0]):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "SQL asset state revision mismatch")
        self.import_state(raw)

    async def _mutate(
        self,
        mutation: "Callable[[InMemoryAssetBackend], Awaitable[_ResultT]]",
    ) -> _ResultT:
        from sqlalchemy import insert, select, update
        from sqlalchemy.exc import IntegrityError, OperationalError

        session = self._session_factory()
        try:
            async with session:
                async with session.begin():
                    row = (
                        await session.execute(
                            select(
                                self._tables.state.c.store_revision,
                                self._tables.state.c.payload,
                            )
                            .where(self._tables.state.c.namespace == self._namespace)
                            .with_for_update()
                        )
                    ).first()
                    expected_revision = 0 if row is None else int(row[0])
                    backend = InMemoryAssetBackend(self.root)
                    if row is not None:
                        raw = json.loads(bytes(row[1]).decode("utf-8"))
                        if not isinstance(raw, dict) or int(raw.get("store_revision", -1)) != expected_revision:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        backend.import_state(raw)
                    result = await mutation(backend)
                    next_state = backend.export_state()
                    next_revision = int(next_state["store_revision"])
                    if next_revision != expected_revision:
                        payload = json.dumps(
                            next_state,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        if row is None:
                            await session.execute(
                                insert(self._tables.state).values(
                                    namespace=self._namespace,
                                    store_revision=next_revision,
                                    payload=payload,
                                )
                            )
                        else:
                            updated = await session.execute(
                                update(self._tables.state)
                                .where(
                                    self._tables.state.c.namespace == self._namespace,
                                    self._tables.state.c.store_revision == expected_revision,
                                )
                                .values(store_revision=next_revision, payload=payload)
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
            next_revision,
        )
        return result


__all__ = ["SqlAssetBackend", "SqlAssetTables"]
