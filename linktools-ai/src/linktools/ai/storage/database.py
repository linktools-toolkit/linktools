#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lazy SQLAlchemy database construction and frozen schema registration."""

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit, unquote
from typing import TYPE_CHECKING, Protocol

from ..core.errors import ErrorCode, AIError
from ..core.json import canonical_json_bytes

if TYPE_CHECKING:
    from sqlalchemy import Constraint, Index, MetaData, Table
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

class CoordinationScope(StrEnum):
    PROCESS = "process"
    SHARED_DATABASE = "shared_database"


@dataclass(frozen=True, slots=True)
class SqlTableManifest:
    name: str
    owner: str
    columns: "tuple[str, ...]"
    constraints: "tuple[str, ...]"
    indexes: "tuple[str, ...]"


@dataclass(frozen=True, slots=True)
class SqlSchemaManifest:
    tables: "tuple[SqlTableManifest, ...]"
    digest: str


class SqlSchemaContributor(Protocol):
    @classmethod
    def register_schema(cls, registry: "SqlSchemaRegistry") -> "Table": ...


class SqlSchemaRegistry:
    def __init__(self, metadata: "MetaData | None" = None) -> None:
        self._metadata = _new_metadata() if metadata is None else metadata
        self._tables: dict[str, SqlTableManifest] = {}
        self._frozen = False
        self._manifest: SqlSchemaManifest | None = None

    @property
    def metadata(self) -> "MetaData":
        return self._metadata

    def add_table(self, table: "Table", *, owner: str) -> None:
        if self._frozen:
            raise ValueError("SQL schema registry is frozen")
        manifest = SqlTableManifest(
            table.name,
            owner,
            tuple(
                f"{column.name}:{column.type}:{int(column.nullable)}:{int(column.primary_key)}"
                for column in table.columns
            ),
            tuple(sorted(sql_constraint_signature(constraint) for constraint in table.constraints)),
            tuple(
                sorted(
                    _index_signature(index)
                    for index in table.indexes
                    if index.name is not None
                )
            ),
        )
        previous = self._tables.get(table.name)
        if previous is not None and previous != manifest:
            raise ValueError(f"conflicting SQL table registration: {table.name}")
        if previous is not None and previous.owner != owner:
            raise ValueError(f"duplicate SQL table owner: {table.name}")
        self._tables[table.name] = manifest

    def freeze(self) -> SqlSchemaManifest:
        if self._manifest is not None:
            return self._manifest
        self._frozen = True
        tables = tuple(self._tables[name] for name in sorted(self._tables))
        payload = {
            "tables": [
                {
                    "name": table.name,
                    "owner": table.owner,
                    "columns": list(table.columns),
                    "constraints": list(table.constraints),
                    "indexes": list(table.indexes),
                }
                for table in tables
            ]
        }
        digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        self._manifest = SqlSchemaManifest(tables, digest)
        return self._manifest

    @property
    def manifest(self) -> SqlSchemaManifest:
        if self._manifest is None:
            raise ValueError("SQL schema registry is not frozen")
        return self._manifest

    @property
    def frozen(self) -> bool:
        return self._frozen


@dataclass(frozen=True, slots=True)
class StorageDatabase:
    engine: "AsyncEngine"
    session_factory: "async_sessionmaker[AsyncSession]"
    coordination_scope: CoordinationScope
    metadata: "MetaData"
    target_identity: str
    schema_manifest_digest: str


def dialect_for_url(url: str) -> str:
    scheme = url.split(":", 1)[0]
    if scheme == "sqlite+aiosqlite":
        return "sqlite"
    if scheme == "mysql+asyncmy":
        return "mysql"
    if scheme == "postgresql+asyncpg":
        return "postgresql"
    raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "unsupported SQL storage dialect")


def scope_for_url(url: str) -> CoordinationScope:
    return CoordinationScope.PROCESS if url.startswith("sqlite") else CoordinationScope.SHARED_DATABASE


def sql_constraint_signature(constraint: "Constraint") -> str:
    name = constraint.name or ""
    columns = ",".join(column.name for column in constraint.columns)
    from sqlalchemy import CheckConstraint

    expression = str(constraint.sqltext) if isinstance(constraint, CheckConstraint) else ""
    return f"{type(constraint).__name__}:{name}:{columns}:{expression}"


def _index_signature(index: "Index") -> str:
    name = index.name or ""
    columns = ",".join(column.name for column in index.columns)
    return f"{name}:{columns}"


def build_storage(
    async_url: str,
    *,
    metadata: "MetaData",
    schema_manifest_digest: str,
) -> StorageDatabase:
    dialect = dialect_for_url(async_url)
    try:
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    except ModuleNotFoundError as error:
        if error.name == "sqlalchemy":
            raise AIError(
                ErrorCode.OPTIONAL_DEPENDENCY_MISSING,
                "SQLAlchemy is required for SQL storage",
            ) from error
        raise
    try:
        engine = create_async_engine(async_url)
    except ModuleNotFoundError as error:
        if error.name in {"aiosqlite", "asyncmy", "asyncpg"}:
            raise AIError(ErrorCode.OPTIONAL_DEPENDENCY_MISSING, f"SQL driver is required for {dialect} storage") from error
        raise
    if dialect == "sqlite":
        from sqlalchemy import event

        @event.listens_for(engine.sync_engine, "connect")
        def _configure_sqlite(connection: object, _: object) -> None:
            cursor = connection.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
                if str(cursor.fetchone()[0]).lower() != "wal":
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "SQLite WAL is unavailable")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=5000")
            finally:
                cursor.close()
    parsed = urlsplit(async_url)
    target = str(Path(unquote(parsed.path)).expanduser().resolve()) if dialect == "sqlite" else f"{dialect}:{parsed.hostname}:{parsed.port or 0}:{parsed.path.lstrip('/')}"
    return StorageDatabase(
        engine,
        async_sessionmaker(engine, expire_on_commit=False),
        scope_for_url(async_url),
        metadata,
        target,
        schema_manifest_digest,
    )


def build_sqlite_storage(
    path: Path,
    *,
    metadata: "MetaData",
    schema_manifest_digest: str,
) -> StorageDatabase:
    return build_storage(
        f"sqlite+aiosqlite:///{path}",
        metadata=metadata,
        schema_manifest_digest=schema_manifest_digest,
    )


async def close_storage(database: StorageDatabase) -> None:
    await database.engine.dispose()


def _new_metadata() -> "MetaData":
    try:
        from sqlalchemy import MetaData
    except ModuleNotFoundError as error:
        if error.name == "sqlalchemy":
            raise AIError(
                ErrorCode.OPTIONAL_DEPENDENCY_MISSING,
                "SQLAlchemy is required for SQL schema registration",
            ) from error
        raise
    return MetaData()


__all__ = [
    "CoordinationScope",
    "SqlSchemaContributor",
    "SqlSchemaManifest",
    "SqlSchemaRegistry",
    "SqlTableManifest",
    "StorageDatabase",
    "build_sqlite_storage",
    "build_storage",
    "close_storage",
    "dialect_for_url",
    "sql_constraint_signature",
    "scope_for_url",
]
