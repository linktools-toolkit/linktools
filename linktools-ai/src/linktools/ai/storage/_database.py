#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Borrowed SQLAlchemy database metadata and frozen schema registration."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from ..core import canonical_json_bytes
from ..errors import AIError, ErrorCode

if TYPE_CHECKING:
    from sqlalchemy import BigInteger, CHAR, Constraint, Index, LargeBinary, MetaData, String, Table
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
    session_factory: "async_sessionmaker[AsyncSession]"
    coordination_scope: CoordinationScope
    metadata: "MetaData"
    schema_manifest_digest: str


def sql_constraint_signature(constraint: "Constraint") -> str:
    name = constraint.name or ""
    columns = ",".join(column.name for column in constraint.columns)
    from sqlalchemy import CheckConstraint

    expression = str(constraint.sqltext) if isinstance(constraint, CheckConstraint) else ""
    return f"{type(constraint).__name__}:{name}:{columns}:{expression}"


def sql_integer_id() -> "BigInteger":
    from sqlalchemy import BigInteger, Integer

    return BigInteger().with_variant(Integer, "sqlite")


def sql_text_key(length: int) -> "String":
    from sqlalchemy import String
    from sqlalchemy.dialects import mysql

    return String(length).with_variant(
        mysql.VARCHAR(length, charset="utf8mb4", collation="utf8mb4_bin"),
        "mysql",
    )


def sql_digest() -> "CHAR":
    from sqlalchemy import CHAR
    from sqlalchemy.dialects import mysql

    return CHAR(64).with_variant(
        mysql.CHAR(64, charset="utf8mb4", collation="utf8mb4_bin"),
        "mysql",
    )


def sql_blob() -> "LargeBinary":
    from sqlalchemy import LargeBinary
    from sqlalchemy.dialects import mysql

    return LargeBinary().with_variant(mysql.LONGBLOB(), "mysql")


def sql_table_options() -> "Mapping[str, str]":
    return {
        "mysql_engine": "InnoDB",
        "mysql_charset": "utf8mb4",
        "mysql_collate": "utf8mb4_bin",
    }


def sql_index(index: "Index") -> "Index":
    index.info["ddl_dialect"] = "mysql"
    return index.ddl_if(dialect="mysql")


def _index_signature(index: "Index") -> str:
    name = index.name or ""
    columns = ",".join(column.name for column in index.columns)
    return f"{name}:{columns}"


def build_storage(
    *,
    session_factory: "async_sessionmaker[AsyncSession]",
    metadata: "MetaData",
    schema_manifest_digest: str,
    coordination_scope: CoordinationScope = CoordinationScope.SHARED_DATABASE,
) -> StorageDatabase:
    return StorageDatabase(
        session_factory,
        coordination_scope,
        metadata,
        schema_manifest_digest,
    )


async def build_sqlite_storage(
    *,
    session_factory: "async_sessionmaker[AsyncSession]",
    metadata: "MetaData",
    schema_manifest_digest: str,
) -> StorageDatabase:
    await _configure_sqlite_engine(session_factory)
    return build_storage(
        session_factory=session_factory,
        coordination_scope=CoordinationScope.PROCESS,
        metadata=metadata,
        schema_manifest_digest=schema_manifest_digest,
    )


async def _resolve_engine(session_factory: "async_sessionmaker[AsyncSession]") -> "AsyncEngine":
    from sqlalchemy.ext.asyncio import AsyncEngine

    async with session_factory() as session:
        bound = session.bind
    if not isinstance(bound, AsyncEngine):
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    return bound


def _new_metadata() -> "MetaData":
    from sqlalchemy import MetaData
    return MetaData()


async def _configure_sqlite_engine(session_factory: "async_sessionmaker[AsyncSession]") -> None:
    from sqlalchemy import event

    bound = await _resolve_engine(session_factory)

    @event.listens_for(bound.sync_engine, "connect")
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
    "sql_blob",
    "sql_constraint_signature",
    "sql_digest",
    "sql_index",
    "sql_integer_id",
    "sql_table_options",
    "sql_text_key",
]
