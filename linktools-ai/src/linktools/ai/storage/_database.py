#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Borrowed SQLAlchemy database metadata and frozen schema registration."""

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from ..errors import ErrorCode, AIError
from ..core import canonical_json_bytes

if TYPE_CHECKING:
    from sqlalchemy import Constraint, Index, MetaData, Table
    from sqlalchemy.ext.asyncio import AsyncEngine

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
    coordination_scope: CoordinationScope
    metadata: "MetaData"
    schema_manifest_digest: str


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
    *,
    engine: "AsyncEngine",
    metadata: "MetaData",
    schema_manifest_digest: str,
    coordination_scope: CoordinationScope = CoordinationScope.SHARED_DATABASE,
) -> StorageDatabase:
    return StorageDatabase(
        engine,
        coordination_scope,
        metadata,
        schema_manifest_digest,
    )


def build_sqlite_storage(
    *,
    engine: "AsyncEngine",
    metadata: "MetaData",
    schema_manifest_digest: str,
) -> StorageDatabase:
    _configure_sqlite_engine(engine)
    return build_storage(
        engine=engine,
        coordination_scope=CoordinationScope.PROCESS,
        metadata=metadata,
        schema_manifest_digest=schema_manifest_digest,
    )


async def close_storage(database: StorageDatabase) -> None:
    await database.engine.dispose()


def _new_metadata() -> "MetaData":
    from sqlalchemy import MetaData
    return MetaData()


def _configure_sqlite_engine(engine: "AsyncEngine") -> None:
    from sqlalchemy import event

    @event.listens_for(engine.sync_engine, "connect")
    def _configure_sqlite(connection: object, _: object) -> None:
        cursor: Any = connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            if str(cursor.fetchone()[0]).lower() != "wal":
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "SQLite WAL is unavailable")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
        finally:
            cursor.close()


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
    "sql_constraint_signature",
]
