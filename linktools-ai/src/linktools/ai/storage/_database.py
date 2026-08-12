#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Borrowed SQLAlchemy database metadata and frozen schema registration."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, Protocol

from linktools.core import environ

from ..core import canonical_json_bytes, validate_persistence_namespace
from ..errors import AIError, ErrorCode
from ._names import storage_name
from ._dialects import MySQLDialect, PostgreSQLDialect, SQLiteDialect, SqlAlchemyDialect

if TYPE_CHECKING:
    from sqlalchemy import (
        CHAR,
        BigInteger,
        Constraint,
        Index,
        LargeBinary,
        MetaData,
        String,
        Table,
    )
    from sqlalchemy.engine import Connection
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker


_logger = environ.get_logger("ai.storage.database")


class CoordinationScope(StrEnum):
    PROCESS = "process"
    SHARED_DATABASE = "shared_database"


@dataclass(slots=True)
class SqlStorageContext:
    """Runtime-wide SQL engine and session factory boundary."""

    engine: "AsyncEngine"
    sessions: "async_sessionmaker[AsyncSession]"
    dialect: SqlAlchemyDialect
    namespace: str
    owns_engine: bool = False
    schema_manifest_digest: "str | None" = None

    async def initialize(
        self,
        *,
        metadata: "MetaData | None" = None,
        schema_manifest_digest: "str | None" = None,
    ) -> None:
        if self.engine.dialect.name == "sqlite":
            await _configure_sqlite_engine(self.engine)
        if metadata is not None:
            if not schema_manifest_digest:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await validate_schema(self.engine, metadata)
            await _validate_schema_generation(self.engine, metadata, schema_manifest_digest, allow_create=self.owns_engine)
            self.schema_manifest_digest = schema_manifest_digest
        _logger.debug(
            "SQL storage context initialized: dialect=%s namespace=%s owns_engine=%s schema_manifest_digest=%s",
            self.dialect.name,
            self.namespace,
            self.owns_engine,
            schema_manifest_digest,
        )

    async def close(self) -> None:
        _SQL_CONTEXTS.pop((id(self.engine), self.namespace), None)
        if self.owns_engine:
            await self.engine.dispose()


_SQL_CONTEXTS: dict[tuple[int, str], SqlStorageContext] = {}
_SQL_SCHEMA_CONTRIBUTORS: dict[str, Callable[["SqlSchemaRegistry"], "SqlSchemaContribution"]] = {}
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


class _SqlTypeValue(Protocol):
    def __str__(self) -> str: ...


class SqlSchemaContribution(Protocol):
    """Value returned by a schema contributor."""


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
    sessions: "async_sessionmaker[AsyncSession]"
    engine: "AsyncEngine"
    coordination_scope: CoordinationScope
    metadata: "MetaData"
    schema_manifest_digest: str

    async def initialize(self) -> None:
        if not self.schema_manifest_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await validate_schema(self.engine, self.metadata)
        await _validate_schema_generation(self.engine, self.metadata, self.schema_manifest_digest, allow_create=False)


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


def _build_storage(
    *,
    sessions: "async_sessionmaker[AsyncSession]",
    engine: "AsyncEngine",
    metadata: "MetaData",
    schema_manifest_digest: str,
    coordination_scope: CoordinationScope = CoordinationScope.SHARED_DATABASE,
) -> StorageDatabase:
    return StorageDatabase(
        sessions,
        engine,
        coordination_scope,
        metadata,
        schema_manifest_digest,
    )


async def prepare_storage_database(
    engine: "AsyncEngine",
    *,
    metadata: "MetaData",
    schema_manifest_digest: str,
    context: "SqlStorageContext | None" = None,
) -> StorageDatabase:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    sessions = context.sessions if context is not None else async_sessionmaker(engine, expire_on_commit=False)
    coordination_scope = CoordinationScope.SHARED_DATABASE
    if engine.dialect.name == "sqlite":
        await _configure_sqlite_engine(engine)
        coordination_scope = CoordinationScope.PROCESS
    database = _build_storage(
        sessions=sessions,
        engine=engine,
        coordination_scope=coordination_scope,
        metadata=metadata,
        schema_manifest_digest=schema_manifest_digest,
    )
    _logger.debug(
        "SQL storage database prepared: dialect=%s coordination_scope=%s schema_manifest_digest=%s",
        engine.dialect.name,
        coordination_scope.value,
        schema_manifest_digest,
    )
    return database


async def validate_schema(
    engine: "AsyncEngine",
    metadata: "MetaData",
) -> None:
    """Validate owned tables without issuing schema-changing statements."""
    try:
        async with engine.begin() as connection:
            await connection.run_sync(_validate_schema, metadata)
    except AIError:
        _logger.exception("SQL schema validation failed: table_count=%s", len(metadata.tables))
        raise
    _logger.debug("SQL schema validated: table_count=%s", len(metadata.tables))


def _validate_schema(connection: "Connection", metadata: "MetaData") -> None:
    from sqlalchemy import inspect

    inspector = inspect(connection)
    actual_tables = set(inspector.get_table_names())
    expected_tables = {table.name for table in metadata.tables.values()}
    if not expected_tables.issubset(actual_tables):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    filter_names = tuple(sorted(expected_tables))
    columns = inspector.get_multi_columns(filter_names=filter_names)
    primary_keys = inspector.get_multi_pk_constraint(filter_names=filter_names)
    check_constraints = inspector.get_multi_check_constraints(filter_names=filter_names)
    unique_constraints = inspector.get_multi_unique_constraints(filter_names=filter_names)
    foreign_keys = inspector.get_multi_foreign_keys(filter_names=filter_names)
    indexes = inspector.get_multi_indexes(filter_names=filter_names)
    for table in metadata.tables.values():
        table_key = (table.schema, table.name)
        primary_key_record = primary_keys.get(table_key, {})
        primary_key = set(primary_key_record.get("constrained_columns", ()))
        actual_columns = {
            f"{column['name']}:{_type_name(column['type'], column['name'])}:{int(bool(column['nullable']))}:{int(column['name'] in primary_key)}"
            for column in columns.get(table_key, ())
        }
        expected_columns = {
            f"{column.name}:{_type_name(column.type.dialect_impl(connection.dialect), column.name)}:{int(bool(column.nullable))}:{int(column.primary_key)}"
            for column in table.columns
        }
        if actual_columns != expected_columns:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        expected_constraints = {
            sql_constraint_signature(constraint)
            for constraint in table.constraints
        }
        actual_constraints = {
            f"PrimaryKeyConstraint::{','.join(primary_key_record.get('constrained_columns', ())) }:"
        }
        actual_constraints.update(
            f"CheckConstraint:{item.get('name') or ''}::{item.get('sqltext') or ''}"
            for item in check_constraints.get(table_key, ())
        )
        actual_constraints.update(
            f"UniqueConstraint:{item.get('name') or ''}:{','.join(item.get('column_names', ())) }:"
            for item in unique_constraints.get(table_key, ())
        )
        actual_constraints.update(
            f"ForeignKeyConstraint:{item.get('name') or ''}:{','.join(item.get('constrained_columns', ())) }:"
            for item in foreign_keys.get(table_key, ())
        )
        if actual_constraints != expected_constraints:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        actual_indexes = {
            f"{item.get('name') or ''}:{','.join(item.get('column_names', ())) }"
            for item in indexes.get(table_key, ())
            if not item.get("unique")
        }
        expected_indexes = {
            index.name + ":" + ",".join(column.name for column in index.columns)
            for index in table.indexes
            if index.name is not None and index.info.get("ddl_dialect", connection.dialect.name) == connection.dialect.name
        }
        if actual_indexes != expected_indexes:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _type_name(value: _SqlTypeValue, column_name: "str | None" = None) -> str:
    name = str(value)
    normalized = name.upper()
    if "JSON" in normalized or (column_name == "payload" and normalized in {"LONGTEXT", "TEXT"}):
        return "JSON"
    if normalized in {"DATETIME", "TIMESTAMP"}:
        return "TIMESTAMP"
    return name


def _new_metadata() -> "MetaData":
    from sqlalchemy import MetaData
    return MetaData()


async def _configure_sqlite_engine(bound: "AsyncEngine") -> None:
    from sqlalchemy import event

    if not event.contains(bound.sync_engine, "checkout", _configure_sqlite_connection):
        event.listen(bound.sync_engine, "checkout", _configure_sqlite_connection)
    async with bound.connect() as connection:
        result = await connection.exec_driver_sql("PRAGMA journal_mode=WAL")
        if str(result.scalar_one()).lower() != "wal":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "SQLite WAL is unavailable")


async def _validate_schema_generation(
    engine: "AsyncEngine",
    metadata: "MetaData",
    schema_manifest_digest: str,
    *,
    allow_create: bool,
) -> None:
    from sqlalchemy import insert, select

    table = metadata.tables.get(storage_name("storage_schema_manifest"))
    if table is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    async with engine.begin() as connection:
        row = (await connection.execute(select(table.c.generation, table.c.manifest_digest).where(table.c.id == 1))).first()
        if row is None:
            if not allow_create:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await connection.execute(
                insert(table).values(
                    id=1,
                    generation=1,
                    manifest_digest=schema_manifest_digest,
                )
            )
            return
        if int(row.generation) != 1 or str(row.manifest_digest) != schema_manifest_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


async def provision_schema_generation(
    engine: "AsyncEngine",
    metadata: "MetaData",
    schema_manifest_digest: str,
) -> None:
    """Write the generation marker during an explicit owned-schema provision."""
    await _validate_schema_generation(engine, metadata, schema_manifest_digest, allow_create=True)


def register_storage_schema(registry: SqlSchemaRegistry) -> "Table":
    from sqlalchemy import Column, DateTime, Index, Integer, Table, UniqueConstraint
    from sqlalchemy.sql import func

    table = Table(
        storage_name("storage_schema_manifest"),
        registry.metadata,
        Column("id", sql_integer_id(), primary_key=True, autoincrement=True),
        Column("generation", Integer, nullable=False),
        Column("manifest_digest", sql_digest(), nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.current_timestamp(), onupdate=func.current_timestamp()),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()),
        UniqueConstraint("generation", name="uk_storage_generation"),
        **sql_table_options(),
    )
    sql_index(Index("ix_updated_at", table.c.updated_at))
    sql_index(Index("ix_created_at", table.c.created_at))
    registry.add_table(table, owner="storage.database")
    return table


def build_sql_schema_metadata() -> "tuple[MetaData, str]":
    if set(_SQL_SCHEMA_CONTRIBUTORS) != {"adapter.sql", "adapter._step", "asset.sql"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "application SQL schema contributors are incomplete")
    registry = SqlSchemaRegistry()
    register_storage_schema(registry)
    for contributor in _SQL_SCHEMA_CONTRIBUTORS.values():
        contributor(registry)
    manifest = registry.freeze()
    return registry.metadata, manifest.digest


def register_sql_schema_contributor(owner: str, contributor: Callable[[SqlSchemaRegistry], SqlSchemaContribution]) -> None:
    if not owner or owner in _SQL_SCHEMA_CONTRIBUTORS:
        raise ValueError("SQL schema contributor is already registered")
    _SQL_SCHEMA_CONTRIBUTORS[owner] = contributor


def create_sql_storage_context(
    engine: "AsyncEngine",
    namespace: str,
    *,
    owns_engine: bool = False,
) -> SqlStorageContext:
    from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

    if not isinstance(engine, AsyncEngine):
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    validate_persistence_namespace(namespace)
    dialect = _dialect_for_name(engine.dialect.name)
    context = SqlStorageContext(
        engine,
        async_sessionmaker(engine, expire_on_commit=False),
        dialect,
        namespace,
        owns_engine,
    )
    _SQL_CONTEXTS[(id(engine), namespace)] = context
    return context


def get_sql_storage_context(engine: "AsyncEngine", namespace: str) -> SqlStorageContext:
    validate_persistence_namespace(namespace)
    context = _SQL_CONTEXTS.get((id(engine), namespace))
    if context is not None and context.engine is engine:
        return context
    return create_sql_storage_context(engine, namespace)


def _dialect_for_name(name: str) -> SqlAlchemyDialect:
    if name == "sqlite":
        return SQLiteDialect()
    if name in {"postgresql", "postgres"}:
        return PostgreSQLDialect()
    if name in {"mysql", "mariadb"}:
        return MySQLDialect()
    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, f"unsupported SQL dialect: {name}")


def _configure_sqlite_connection(connection: Any, connection_record: Any, _: Any) -> None:
    if connection_record.info.get("linktools_ai_sqlite_configured"):
        return
    cursor = connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
    finally:
        cursor.close()
    connection_record.info["linktools_ai_sqlite_configured"] = True


__all__ = [
    "CoordinationScope",
    "SqlSchemaManifest",
    "SqlSchemaRegistry",
    "SqlTableManifest",
    "StorageDatabase",
    "SqlStorageContext",
    "create_sql_storage_context",
    "build_sql_schema_metadata",
    "get_sql_storage_context",
    "register_storage_schema",
    "register_sql_schema_contributor",
    "prepare_storage_database",
    "provision_schema_generation",
    "sql_blob",
    "sql_constraint_signature",
    "sql_digest",
    "sql_index",
    "sql_integer_id",
    "sql_table_options",
    "sql_text_key",
    "validate_schema",
]
