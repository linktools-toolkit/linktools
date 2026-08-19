#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Domain-independent SQL context, metadata primitives, and validation."""

import asyncio
import sys
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from time import monotonic
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar

from linktools.core import environ

from ..errors import AIError, ErrorCode
from ._dialects import (
    SqlAlchemyDialect,
    SqlTransactionDisposition,
    SqlTransactionPhase,
    dialect_for_name,
)

if TYPE_CHECKING:
    from sqlalchemy import (
        CHAR,
        BigInteger,
        Column,
        Index,
        LargeBinary,
        MetaData,
        String,
        Table,
    )
    from sqlalchemy.engine import Connection
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker


_logger = environ.get_logger("ai.storage.database")
ValueT = TypeVar("ValueT")


@dataclass(slots=True)
class SqlStorageContext:
    """The borrowed-engine boundary used by SQL adapters."""

    engine: "AsyncEngine"
    sessions: "async_sessionmaker[AsyncSession]"
    dialect: SqlAlchemyDialect
    owns_engine: bool = False
    _closed: bool = field(default=False, init=False, repr=False)
    _initialize_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _sqlite_configured: bool = field(default=False, init=False, repr=False)
    _validated_metadata: "MetaData | None" = field(default=None, init=False, repr=False)

    async def initialize(self, *, metadata: "MetaData | None" = None) -> None:
        async with self._initialize_lock:
            if self._closed:
                raise AIError(ErrorCode.STORAGE_CLOSED)
            if self.engine.dialect.name == "sqlite" and not self._sqlite_configured:
                await _configure_sqlite_engine(self.engine)
                self._sqlite_configured = True
            if metadata is not None and metadata is not self._validated_metadata:
                await _validate_sql_schema(self.engine, metadata)
                self._validated_metadata = metadata
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

    async def run_mutation(
        self,
        callback: Callable[["AsyncSession"], Awaitable[ValueT]],
        *,
        retry_limit: int = 8,
        domain: str = "sql",
    ) -> ValueT:
        if retry_limit < 1:
            raise ValueError("retry_limit must be positive")
        started = monotonic()
        for attempt in range(retry_limit):
            session = self.sessions()
            transaction = session.begin()
            entered = False
            try:
                await transaction.__aenter__()
                entered = True
                try:
                    result = await callback(session)
                except BaseException as error:
                    disposition = self.dialect.classify_transaction_error(
                        error,
                        phase=SqlTransactionPhase.BODY,
                        connection_invalidated=_connection_invalidated(error),
                    )
                    await transaction.__aexit__(*sys.exc_info())
                    if disposition is SqlTransactionDisposition.RETRYABLE_ABORTED and attempt + 1 < retry_limit:
                        _logger.warning(
                            "retrying SQL mutation after aborted body: domain=%s dialect=%s attempt=%s",
                            domain,
                            self.dialect.name,
                            attempt + 1,
                        )
                        continue
                    raise
                try:
                    await transaction.__aexit__(None, None, None)
                except BaseException as error:
                    disposition = self.dialect.classify_transaction_error(
                        error,
                        phase=SqlTransactionPhase.COMMIT,
                        connection_invalidated=_connection_invalidated(error),
                    )
                    if disposition is SqlTransactionDisposition.RETRYABLE_ABORTED and attempt + 1 < retry_limit:
                        _logger.warning(
                            "retrying SQL mutation after aborted commit: domain=%s dialect=%s attempt=%s",
                            domain,
                            self.dialect.name,
                            attempt + 1,
                        )
                        continue
                    if disposition is SqlTransactionDisposition.COMMIT_UNKNOWN:
                        _logger.error(
                            "SQL mutation commit outcome unknown: domain=%s dialect=%s "
                            "attempt=%s duration_ms=%.3f outcome=unknown",
                            domain,
                            self.dialect.name,
                            attempt + 1,
                            (monotonic() - started) * 1000,
                        )
                        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from error
                    raise
                _logger.debug(
                    "SQL mutation committed: domain=%s dialect=%s attempt=%s duration_ms=%.3f outcome=committed",
                    domain,
                    self.dialect.name,
                    attempt + 1,
                    (monotonic() - started) * 1000,
                )
                return result
            finally:
                if not entered:
                    await transaction.__aexit__(*sys.exc_info())
                await session.close()
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

    @property
    def closed(self) -> bool:
        return self._closed


def create_sql_storage_context(engine: "AsyncEngine", *, owns_engine: bool = False) -> SqlStorageContext:
    dialect = dialect_for_name(engine.dialect.name)
    from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

    if not isinstance(engine, AsyncEngine):
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    return SqlStorageContext(
        engine=engine,
        sessions=async_sessionmaker(engine, expire_on_commit=False),
        dialect=dialect,
        owns_engine=owns_engine,
    )


async def provision_sql(engine: "AsyncEngine", metadata: "MetaData") -> None:
    """Create the explicitly requested metadata without a global schema."""

    dialect_for_name(engine.dialect.name)
    if engine.dialect.name == "sqlite":
        await _configure_sqlite_engine(engine)
    if not metadata.tables:
        return
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    await validate_sql(engine, metadata)
    _logger.info("SQL metadata provisioned: dialect=%s tables=%s", engine.dialect.name, len(metadata.tables))


async def validate_sql(engine: "AsyncEngine", metadata: "MetaData") -> None:
    """Validate that required metadata is a compatible subset of the database."""

    dialect_for_name(engine.dialect.name)
    if engine.dialect.name == "sqlite":
        await _configure_sqlite_engine(engine)
    await _validate_sql_schema(engine, metadata)
    _logger.debug("SQL metadata validated: dialect=%s tables=%s", engine.dialect.name, len(metadata.tables))


async def _validate_sql_schema(engine: "AsyncEngine", metadata: "MetaData") -> None:
    async with engine.connect() as connection:
        await connection.run_sync(_validate_connection_schema, metadata)


def sql_integer_id() -> "BigInteger":
    from sqlalchemy import BigInteger, Integer

    return BigInteger().with_variant(Integer, "sqlite")


def sql_id_column(
    comment: str = "Surrogate row identifier used only by the SQL backend.",
) -> "Column":
    from sqlalchemy import Column

    return Column(
        "id",
        sql_integer_id(),
        primary_key=True,
        autoincrement=True,
        comment=comment,
    )


def sql_text_key(length: int = 256) -> "String":
    from sqlalchemy import String
    from sqlalchemy.dialects import mysql

    return String(length).with_variant(
        mysql.VARCHAR(length, charset="utf8mb4", collation="utf8mb4_bin"),
        "mysql",
    )


def sql_sha256() -> "CHAR":
    from sqlalchemy import CHAR
    from sqlalchemy.dialects import mysql

    return CHAR(64).with_variant(mysql.CHAR(64, charset="utf8mb4", collation="utf8mb4_bin"), "mysql")


def sql_digest() -> "CHAR":
    """Return the canonical SQL SHA-256 type."""
    return sql_sha256()


def sql_state() -> "String":
    from sqlalchemy import String
    from sqlalchemy.dialects import mysql

    return String(64).with_variant(
        mysql.VARCHAR(64, charset="utf8mb4", collation="utf8mb4_bin"),
        "mysql",
    )


def sql_sort_key() -> "String":
    from sqlalchemy import String
    from sqlalchemy.dialects import mysql

    return String(128).with_variant(
        mysql.VARCHAR(128, charset="utf8mb4", collation="utf8mb4_bin"),
        "mysql",
    )


def sql_blob() -> "LargeBinary":
    from sqlalchemy import LargeBinary
    from sqlalchemy.dialects import mysql

    return LargeBinary().with_variant(mysql.LONGBLOB(), "mysql")


def sql_table_options() -> "Mapping[str, str]":
    return {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_bin"}


def sql_audit_columns() -> "tuple[Column, Column]":
    from sqlalchemy import Column, DateTime, DefaultClause
    from sqlalchemy.dialects import mysql
    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy.sql.elements import ClauseElement

    class AuditCurrentTimestamp(ClauseElement):
        inherit_cache = True

        def __str__(self) -> str:
            return "CURRENT_TIMESTAMP"

    class AuditCreatedTimestamp(ClauseElement):
        inherit_cache = True

    @compiles(AuditCurrentTimestamp)
    def compile_audit_timestamp(element: object, compiler: object, **kwargs: object) -> str:
        return "CURRENT_TIMESTAMP"

    @compiles(AuditCurrentTimestamp, "mysql")
    def compile_mysql_audit_timestamp(element: object, compiler: object, **kwargs: object) -> str:
        return "CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"

    @compiles(AuditCreatedTimestamp)
    def compile_created_timestamp(element: object, compiler: object, **kwargs: object) -> str:
        return "CURRENT_TIMESTAMP"

    @compiles(AuditCreatedTimestamp, "mysql")
    def compile_mysql_created_timestamp(element: object, compiler: object, **kwargs: object) -> str:
        return "CURRENT_TIMESTAMP"

    timestamp_type = DateTime(timezone=True).with_variant(mysql.DATETIME(), "mysql")
    return (
        Column(
            "updated_at",
            timestamp_type,
            nullable=False,
            server_default=DefaultClause(AuditCurrentTimestamp()),
            comment="Update timestamp",
        ),
        Column(
            "created_at",
            timestamp_type,
            nullable=False,
            server_default=DefaultClause(AuditCreatedTimestamp()),
            comment="Creation timestamp",
        ),
    )


def sql_audit_indexes(table: "Table") -> "tuple[Index, Index]":
    """Return the required timestamp query indexes for a physical table."""
    return (
        sql_query_index(table, "updated_at"),
        sql_query_index(table, "created_at"),
    )


def sql_unique(table: "Table", *columns: str) -> None:
    from sqlalchemy import Index, UniqueConstraint

    _validate_index_columns(columns)
    constraint = UniqueConstraint(*(table.c[column] for column in columns))
    constraint.info["ddl_dialect"] = "portable"
    constraint.ddl_if(dialect=("sqlite", "postgresql"))
    table.append_constraint(constraint)
    index = Index(
        f"uk_{'_'.join(columns)}",
        *(table.c[column] for column in columns),
        unique=True,
    )
    index.info["ddl_dialect"] = "mysql"
    index.ddl_if(dialect="mysql")


def sql_query_index(table: "Table", *columns: str, mysql_length: int | None = None) -> "Index":
    from sqlalchemy import Index

    _validate_index_columns(columns)
    column_objects = tuple(table.c[column] for column in columns)
    portable = Index(
        f"ix_{table.name}_{'_'.join(columns)}",
        *column_objects,
    )
    portable.info["ddl_dialect"] = "portable"
    portable.ddl_if(dialect=("sqlite", "postgresql"))
    mysql_options: dict[str, object] = {}
    if mysql_length is not None:
        mysql_options["mysql_length"] = {columns[-1]: mysql_length}
    mysql = Index(
        f"ix_{'_'.join(columns)}",
        *column_objects,
        **mysql_options,
    )
    mysql.info["ddl_dialect"] = "mysql"
    mysql.ddl_if(dialect="mysql")
    return portable


def _validate_index_columns(columns: tuple[str, ...]) -> None:
    if not columns or len(columns) > 3:
        raise ValueError("SQL composite indexes must contain one to three columns")


def _validate_connection_schema(connection: "Connection", metadata: "MetaData") -> None:
    from sqlalchemy import inspect

    inspector = inspect(connection)
    actual_tables = set(inspector.get_table_names())
    for expected_table in metadata.tables.values():
        if expected_table.name not in actual_tables:
            _schema_mismatch(table=expected_table.name, category="table")

        actual_columns = {
            str(column["name"]): column
            for column in inspector.get_columns(expected_table.name, schema=expected_table.schema)
        }
        for expected_column in expected_table.columns:
            actual_column = actual_columns.get(expected_column.name)
            if actual_column is None:
                _schema_mismatch(table=expected_table.name, category="column", column=expected_column.name)
            _validate_column_compatibility(
                connection,
                table_name=expected_table.name,
                expected=expected_column,
                actual=actual_column,
            )


def _schema_mismatch(*, table: str, category: str, column: str | None = None) -> NoReturn:
    details: dict[str, object] = {"table": table, "category": category}
    if column is not None:
        details["column"] = column
    raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING, safe_details=details)


def _validate_column_compatibility(
    connection: "Connection",
    *,
    table_name: str,
    expected: Any,
    actual: Mapping[str, Any],
) -> None:
    from sqlalchemy import JSON, LargeBinary

    expected_type = expected.type.dialect_impl(connection.dialect)
    actual_type = actual.get("type")
    dialect_name = connection.dialect.name
    if _type_family(expected_type) != _type_family(actual_type) and not _boolean_compatible(
        dialect_name, expected_type, actual_type
    ):
        _schema_mismatch(table=table_name, category="type", column=expected.name)
    if isinstance(expected_type, LargeBinary) and not _binary_compatible(dialect_name, expected_type, actual_type):
        _schema_mismatch(table=table_name, category="type", column=expected.name)
    if isinstance(expected_type, JSON) and not _json_compatible(expected_type, actual_type):
        _schema_mismatch(table=table_name, category="type", column=expected.name)
    if _type_family(expected_type) == "integer" and not _integer_compatible(dialect_name, expected_type, actual_type):
        _schema_mismatch(table=table_name, category="type", column=expected.name)


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


def _type_name(value: object) -> str:
    return str(value).replace(" ", "").lower()


def _boolean_compatible(dialect_name: str, expected: object, actual: object) -> bool:
    from sqlalchemy import Boolean

    if not isinstance(expected, Boolean):
        return False
    rendered = _type_name(actual)
    if dialect_name == "mysql":
        if rendered in {"boolean", "bool"}:
            return True
        if rendered.startswith("tinyint"):
            from sqlalchemy.dialects.mysql import TINYINT

            width = actual.display_width if isinstance(actual, TINYINT) else None
            return width == 1 or rendered == "tinyint(1)"
    return type(actual).__name__.lower() in {"boolean", "bool"} or rendered in {"boolean", "bool"}


def _binary_compatible(dialect_name: str, expected: object, actual: object) -> bool:
    rendered = _type_name(actual)
    if dialect_name == "mysql":
        return rendered == "longblob" or type(actual).__name__.lower() == "longblob"
    if dialect_name == "postgresql":
        return rendered == "bytea" or type(actual).__name__.lower() == "bytea"
    return rendered in {"blob", "largebinary"} or type(actual).__name__.lower() in {"blob", "largebinary"}


def _json_compatible(expected: object, actual: object) -> bool:
    expected_name = type(expected).__name__.lower()
    actual_name = type(actual).__name__.lower()
    rendered = _type_name(actual)
    if expected_name == "jsonb":
        return actual_name == "jsonb" or rendered == "jsonb"
    return (actual_name == "json" or rendered == "json") and "jsonb" not in rendered


def _integer_compatible(dialect_name: str, expected: object, actual: object) -> bool:
    if _type_family(actual) != "integer":
        return False
    if dialect_name == "sqlite":
        return True
    expected_name = _type_name(expected)
    actual_name = _type_name(actual)
    if "bigint" in expected_name:
        return "bigint" in actual_name
    if "smallint" in expected_name:
        return any(name in actual_name for name in ("smallint", "integer", "int", "bigint"))
    return any(name in actual_name for name in ("integer", "int", "bigint"))


async def _configure_sqlite_engine(engine: "AsyncEngine") -> None:
    if engine.dialect.name != "sqlite":
        raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
    from sqlalchemy import event

    sync_engine = engine.sync_engine
    if not event.contains(sync_engine, "checkout", _configure_sqlite_connection):
        event.listen(sync_engine, "checkout", _configure_sqlite_connection)
        _logger.debug("SQLite checkout PRAGMA listener registered")


def _configure_sqlite_connection(
    dbapi_connection: Any,
    _connection_record: Any,
    _connection_proxy: Any,
) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
    finally:
        cursor.close()


def _connection_invalidated(error: BaseException) -> bool:
    from sqlalchemy.exc import DBAPIError

    return isinstance(error, DBAPIError) and error.connection_invalidated


__all__ = [
    "SqlStorageContext",
    "create_sql_storage_context",
    "provision_sql",
    "validate_sql",
    "sql_blob",
    "sql_digest",
    "sql_sha256",
    "sql_sort_key",
    "sql_state",
    "sql_audit_columns",
    "sql_audit_indexes",
    "sql_integer_id",
    "sql_id_column",
    "sql_query_index",
    "sql_table_options",
    "sql_text_key",
    "sql_unique",
]
