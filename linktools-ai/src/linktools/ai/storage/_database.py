#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Domain-independent SQL context, metadata primitives, and validation."""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from linktools.core import environ

from ..errors import AIError, ErrorCode
from ._dialects import (
    MySQLDialect,
    PostgreSQLDialect,
    SqlAlchemyDialect,
    SQLiteDialect,
)

if TYPE_CHECKING:
    from sqlalchemy import CHAR, BigInteger, Index, LargeBinary, MetaData, String
    from sqlalchemy.engine import Connection
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker


_logger = environ.get_logger("ai.storage.database")
_PROVISION_LOCK = asyncio.Lock()


@dataclass(slots=True)
class SqlContext:
    """The borrowed-engine boundary used by SQL adapters."""

    engine: "AsyncEngine"
    sessions: "async_sessionmaker[AsyncSession]"
    dialect: SqlAlchemyDialect
    owns_engine: bool = False
    _closed: bool = field(default=False, init=False, repr=False)

    async def initialize(self, *, metadata: "MetaData | None" = None) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if self.engine.dialect.name == "sqlite":
            await _configure_sqlite_engine(self.engine)
        if metadata is not None:
            await validate_sql(self.engine, metadata)
        _logger.debug(
            "SQL context initialized: dialect=%s owns_engine=%s metadata=%s",
            self.dialect.name,
            self.owns_engine,
            0 if metadata is None else len(metadata.tables),
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.owns_engine:
            await self.engine.dispose()


def create_sql_context(engine: "AsyncEngine", *, owns_engine: bool = False) -> SqlContext:
    from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

    if not isinstance(engine, AsyncEngine):
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    return SqlContext(
        engine=engine,
        sessions=async_sessionmaker(engine, expire_on_commit=False),
        dialect=_dialect_for_name(engine.dialect.name),
        owns_engine=owns_engine,
    )


async def provision_sql(engine: "AsyncEngine", metadata: "MetaData") -> None:
    """Create the explicitly requested metadata without a global schema."""

    if not metadata.tables:
        return
    if engine.dialect.name == "sqlite":
        await _configure_sqlite_engine(engine)
    async with _PROVISION_LOCK:
        async with engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
    await validate_sql(engine, metadata)
    _logger.info("SQL metadata provisioned: dialect=%s tables=%s", engine.dialect.name, len(metadata.tables))


async def validate_sql(engine: "AsyncEngine", metadata: "MetaData") -> None:
    """Validate that required metadata is a compatible subset of the database."""

    if engine.dialect.name == "sqlite":
        await _configure_sqlite_engine(engine)
    async with engine.connect() as connection:
        await connection.run_sync(_validate_connection_schema, metadata)
    _logger.debug("SQL metadata validated: dialect=%s tables=%s", engine.dialect.name, len(metadata.tables))


def sql_integer_id() -> "BigInteger":
    from sqlalchemy import BigInteger, Integer

    return BigInteger().with_variant(Integer, "sqlite")


def sql_text_key(length: int = 256) -> "String":
    from sqlalchemy import String
    from sqlalchemy.dialects import mysql

    return String(length).with_variant(
        mysql.VARCHAR(length, charset="utf8mb4", collation="utf8mb4_bin"),
        "mysql",
    )


def sql_digest() -> "CHAR":
    from sqlalchemy import CHAR
    from sqlalchemy.dialects import mysql

    return CHAR(64).with_variant(mysql.CHAR(64, charset="utf8mb4", collation="utf8mb4_bin"), "mysql")


def sql_blob() -> "LargeBinary":
    from sqlalchemy import LargeBinary
    from sqlalchemy.dialects import mysql

    return LargeBinary().with_variant(mysql.LONGBLOB(), "mysql")


def sql_table_options() -> "Mapping[str, str]":
    return {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_bin"}


def sql_index(index: "Index") -> "Index":
    index.info["ddl_dialect"] = "mysql"
    return index.ddl_if(dialect="mysql")


def _dialect_for_name(name: str) -> SqlAlchemyDialect:
    if name == "sqlite":
        return SQLiteDialect()
    if name in {"postgresql", "postgres"}:
        return PostgreSQLDialect()
    if name in {"mysql", "mariadb"}:
        return MySQLDialect()
    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, f"unsupported SQL dialect: {name}")


def _validate_connection_schema(connection: "Connection", metadata: "MetaData") -> None:
    from sqlalchemy import (
        UniqueConstraint,
        inspect,
    )

    inspector = inspect(connection)
    for table in metadata.tables.values():
        if table.name not in inspector.get_table_names(schema=table.schema):
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING, f"missing SQL table: {table.name}")
        actual = {column["name"]: column for column in inspector.get_columns(table.name, schema=table.schema)}
        for column in table.columns:
            value = actual.get(column.name)
            if value is None:
                raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING, f"missing SQL column: {table.name}.{column.name}")
            if bool(value.get("nullable", True)) != bool(column.nullable):
                raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING, f"nullable SQL column: {table.name}.{column.name}")
            if _type_family(column.type) != _type_family(value.get("type")):
                raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING, f"type mismatch: {table.name}.{column.name}")
        required_names = {column.name for column in table.columns}
        for column in actual.values():
            if column["name"] in required_names:
                continue
            if not bool(column.get("nullable", True)) and column.get("default") is None and not column.get("autoincrement") and column.get("computed") is None:
                raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING, f"required extra SQL column: {table.name}.{column['name']}")
        actual_pk = inspector.get_pk_constraint(table.name, schema=table.schema).get("constrained_columns", [])
        required_pk = [column.name for column in table.primary_key.columns]
        if required_pk and list(actual_pk) != required_pk:
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING, f"primary key mismatch: {table.name}")
        actual_unique = {
            tuple(item.get("column_names", ()))
            for item in inspector.get_unique_constraints(table.name, schema=table.schema)
        }
        actual_unique.update(
            tuple(item.get("column_names", ()))
            for item in inspector.get_indexes(table.name, schema=table.schema)
            if item.get("unique")
        )
        for constraint in table.constraints:
            if not isinstance(constraint, UniqueConstraint):
                continue
            columns = tuple(column.name for column in constraint.columns)
            if columns and columns not in actual_unique:
                raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING, f"unique constraint mismatch: {table.name}")
        actual_indexes = {
            tuple(item.get("column_names", ()))
            for item in inspector.get_indexes(table.name, schema=table.schema)
        }
        for index in table.indexes:
            columns = tuple(column.name for column in index.columns)
            if columns and columns not in actual_indexes and columns not in actual_unique:
                raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING, f"index mismatch: {table.name}")


def _type_family(value: object) -> str:
    from sqlalchemy import JSON, Boolean, DateTime, LargeBinary, Text

    name = type(value).__name__.lower() if value is not None else ""
    rendered = str(value).lower()
    if isinstance(value, (DateTime,)) or "datetime" in name or "timestamp" in rendered:
        return "datetime"
    if isinstance(value, (JSON,)) or name == "json" or "json" in rendered:
        return "json"
    if isinstance(value, (Boolean,)) or name == "boolean" or "bool" in rendered:
        return "boolean"
    if isinstance(value, (LargeBinary,)) or "binary" in name or "blob" in rendered:
        return "binary"
    if isinstance(value, Text) or "char" in rendered or "text" in rendered or "clob" in rendered:
        return "text"
    if "int" in name or "int" in rendered or "numeric" in rendered or "decimal" in rendered:
        return "integer"
    if "float" in name or "real" in rendered:
        return "float"
    return name or rendered


async def _configure_sqlite_engine(engine: "AsyncEngine") -> None:
    def configure(connection: Any) -> None:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        connection.exec_driver_sql("PRAGMA busy_timeout=5000")

    async with engine.connect() as connection:
        await connection.run_sync(configure)


__all__ = [
    "SqlContext",
    "create_sql_context",
    "provision_sql",
    "validate_sql",
    "sql_blob",
    "sql_digest",
    "sql_index",
    "sql_integer_id",
    "sql_table_options",
    "sql_text_key",
]
