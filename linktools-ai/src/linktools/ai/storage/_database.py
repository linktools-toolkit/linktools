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


def sql_audit_columns(
    updated_comment: str = "Update timestamp",
    created_comment: str = "Creation timestamp",
) -> "tuple[Column, Column]":
    from sqlalchemy import TIMESTAMP, Column, DefaultClause
    from sqlalchemy.dialects import mysql
    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy.sql.elements import ClauseElement

    class AuditCurrentTimestamp(ClauseElement):
        inherit_cache = True

        def __str__(self) -> str:
            return "CURRENT_TIMESTAMP(6)"

    class AuditCreatedTimestamp(ClauseElement):
        inherit_cache = True

    @compiles(AuditCurrentTimestamp)
    def compile_audit_timestamp(element: object, compiler: object, **kwargs: object) -> str:
        return "CURRENT_TIMESTAMP"

    @compiles(AuditCurrentTimestamp, "mysql")
    def compile_mysql_audit_timestamp(element: object, compiler: object, **kwargs: object) -> str:
        return "CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)"

    @compiles(AuditCreatedTimestamp)
    def compile_created_timestamp(element: object, compiler: object, **kwargs: object) -> str:
        return "CURRENT_TIMESTAMP"

    @compiles(AuditCreatedTimestamp, "mysql")
    def compile_mysql_created_timestamp(element: object, compiler: object, **kwargs: object) -> str:
        return "CURRENT_TIMESTAMP(6)"

    timestamp_type = TIMESTAMP(timezone=True).with_variant(mysql.TIMESTAMP(fsp=6), "mysql")
    return (
        Column(
            "updated_at",
            timestamp_type,
            nullable=False,
            server_default=DefaultClause(AuditCurrentTimestamp()),
            comment=updated_comment,
        ),
        Column(
            "created_at",
            timestamp_type,
            nullable=False,
            server_default=DefaultClause(AuditCreatedTimestamp()),
            comment=created_comment,
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
    from sqlalchemy import (
        CheckConstraint,
        ForeignKeyConstraint,
        UniqueConstraint,
        inspect,
    )

    inspector = inspect(connection)
    dialect_name = connection.dialect.name
    actual_tables = set(inspector.get_table_names())
    for expected_table in metadata.tables.values():
        if expected_table.name not in actual_tables:
            _schema_mismatch(table=expected_table.name, category="table")
        if dialect_name == "mysql":
            _validate_mysql_options(inspector, expected_table.name, expected_table.schema)
        _validate_comments(inspector, dialect_name, expected_table)

        actual_columns = {
            str(column["name"]): column
            for column in inspector.get_columns(expected_table.name, schema=expected_table.schema)
        }
        for expected_column in expected_table.columns:
            actual_column = actual_columns.get(expected_column.name)
            if actual_column is None:
                _schema_mismatch(table=expected_table.name, category="column", column=expected_column.name)
            if bool(actual_column.get("nullable", True)) != bool(expected_column.nullable):
                _schema_mismatch(table=expected_table.name, category="nullable", column=expected_column.name)
            _validate_column_compatibility(
                connection,
                table_name=expected_table.name,
                expected=expected_column,
                actual=actual_column,
            )
            _validate_server_default(
                dialect_name,
                table_name=expected_table.name,
                expected=expected_column,
                actual=actual_column,
            )

        expected_names = {column.name for column in expected_table.columns}
        for actual_column in actual_columns.values():
            if actual_column["name"] in expected_names:
                continue
            if (
                not bool(actual_column.get("nullable", True))
                and actual_column.get("default") is None
                and not actual_column.get("autoincrement")
                and actual_column.get("computed") is None
            ):
                _schema_mismatch(
                    table=expected_table.name, category="unsafe_extra_column", column=str(actual_column["name"])
                )

        expected_pk = tuple(column.name for column in expected_table.primary_key.columns)
        actual_pk = tuple(
            inspector.get_pk_constraint(expected_table.name, schema=expected_table.schema).get("constrained_columns")
            or ()
        )
        if actual_pk != expected_pk:
            _schema_mismatch(table=expected_table.name, category="primary_key")
        _validate_autoincrement(dialect_name, expected_table, actual_columns, actual_pk)

        expected_unique = {
            tuple(column.name for column in constraint.columns)
            for constraint in expected_table.constraints
            if isinstance(constraint, UniqueConstraint) and _ddl_matches(constraint, dialect_name)
        }
        expected_unique.update(
            tuple(column.name for column in index.columns)
            for index in expected_table.indexes
            if index.unique and _ddl_matches(index, dialect_name)
        )
        actual_unique: set[tuple[str, ...]] = set()
        for item in inspector.get_unique_constraints(expected_table.name, schema=expected_table.schema):
            actual_unique.add(_reflected_columns(expected_table.name, item, "unique"))
        actual_indexes: set[tuple[str, ...]] = set()
        actual_non_unique: set[tuple[str, ...]] = set()
        for item in inspector.get_indexes(expected_table.name, schema=expected_table.schema):
            signature = _reflected_columns(expected_table.name, item, "index")
            actual_indexes.add(signature)
            if item.get("unique"):
                actual_unique.add(signature)
            else:
                actual_non_unique.add(signature)
        if actual_unique != expected_unique:
            _schema_mismatch(table=expected_table.name, category="unique")

        expected_indexes = {
            tuple(column.name for column in index.columns)
            for index in expected_table.indexes
            if not index.unique and _ddl_matches(index, dialect_name)
        }
        if not expected_indexes.issubset(actual_non_unique):
            _schema_mismatch(table=expected_table.name, category="index")

        expected_fks = {
            _foreign_key_signature(
                constrained_columns=tuple(column.name for column in constraint.columns),
                referred_schema=next(iter(constraint.elements)).target_fullname.split(".")[0]
                if next(iter(constraint.elements)).target_fullname.count(".") == 2
                else None,
                referred_table=next(iter(constraint.elements)).column.table.name,
                referred_columns=tuple(element.column.name for element in constraint.elements),
                ondelete=next(iter(constraint.elements)).ondelete,
            )
            for constraint in expected_table.constraints
            if isinstance(constraint, ForeignKeyConstraint)
        }
        actual_fks = {
            _foreign_key_signature(
                constrained_columns=tuple(item.get("constrained_columns") or ()),
                referred_schema=item.get("referred_schema"),
                referred_table=str(item.get("referred_table") or ""),
                referred_columns=tuple(item.get("referred_columns") or ()),
                ondelete=item.get("options", {}).get("ondelete"),
            )
            for item in inspector.get_foreign_keys(expected_table.name, schema=expected_table.schema)
        }
        if actual_fks != expected_fks:
            _schema_mismatch(table=expected_table.name, category="foreign_key")

        expected_checks = {
            str(constraint.sqltext).strip()
            for constraint in expected_table.constraints
            if isinstance(constraint, CheckConstraint)
        }
        actual_checks = {
            str(item.get("sqltext", "")).strip()
            for item in inspector.get_check_constraints(expected_table.name, schema=expected_table.schema)
        }
        if actual_checks != expected_checks:
            _schema_mismatch(table=expected_table.name, category="check")


def _schema_mismatch(*, table: str, category: str, column: str | None = None) -> NoReturn:
    details: dict[str, object] = {"table": table, "category": category}
    if column is not None:
        details["column"] = column
    raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING, safe_details=details)


def _ddl_matches(item: Any, dialect_name: str) -> bool:
    dialect = item.info.get("ddl_dialect")
    return dialect is None or dialect == dialect_name or dialect == "portable" and dialect_name != "mysql"


def _validate_comments(inspector: Any, dialect_name: str, table: Any) -> None:
    if dialect_name not in {"mysql", "postgresql"}:
        return
    table_comment = inspector.get_table_comment(table.name, schema=table.schema).get("text")
    if table_comment != table.comment:
        _schema_mismatch(table=table.name, category="table_comment")
    actual_columns = {str(column["name"]): column for column in inspector.get_columns(table.name, schema=table.schema)}
    for column in table.columns:
        if actual_columns[column.name].get("comment") != column.comment:
            _schema_mismatch(table=table.name, category="column_comment", column=column.name)


def _reflected_columns(table_name: str, item: Mapping[str, Any], category: str) -> tuple[str, ...]:
    columns = item.get("column_names")
    if not isinstance(columns, (list, tuple)) or not columns or any(not isinstance(column, str) for column in columns):
        _schema_mismatch(table=table_name, category=f"{category}_reflection")
    return tuple(columns)


def _validate_column_compatibility(
    connection: "Connection",
    *,
    table_name: str,
    expected: Any,
    actual: Mapping[str, Any],
) -> None:
    from sqlalchemy import CHAR, JSON, LargeBinary, Numeric, String, Text

    expected_type = expected.type.dialect_impl(connection.dialect)
    actual_type = actual.get("type")
    dialect_name = connection.dialect.name
    if _type_family(expected_type) != _type_family(actual_type) and not _boolean_compatible(
        dialect_name, expected_type, actual_type
    ):
        _schema_mismatch(table=table_name, category="type", column=expected.name)
    if isinstance(expected_type, CHAR):
        if not _is_fixed_char(actual_type) or _type_length(actual_type) != expected_type.length:
            _schema_mismatch(table=table_name, category="capacity", column=expected.name)
    elif isinstance(expected_type, Text):
        if not _is_unbounded_text(actual_type):
            _schema_mismatch(table=table_name, category="capacity", column=expected.name)
    elif isinstance(expected_type, String):
        expected_length = expected_type.length
        actual_length = _type_length(actual_type)
        if expected_length is not None and actual_length is not None and actual_length < expected_length:
            _schema_mismatch(table=table_name, category="capacity", column=expected.name)
        if expected_length is not None and actual_length is None and not _is_unbounded_text(actual_type):
            _schema_mismatch(table=table_name, category="capacity", column=expected.name)
    elif isinstance(expected_type, LargeBinary):
        if not _binary_compatible(dialect_name, expected_type, actual_type):
            _schema_mismatch(table=table_name, category="type", column=expected.name)
    elif isinstance(expected_type, JSON):
        if not _json_compatible(expected_type, actual_type):
            _schema_mismatch(table=table_name, category="type", column=expected.name)
    elif _type_family(expected_type) == "integer" and not _integer_compatible(dialect_name, expected_type, actual_type):
        _schema_mismatch(table=table_name, category="type", column=expected.name)
    expected_precision = expected_type.precision if isinstance(expected_type, Numeric) else None
    actual_precision = actual_type.precision if isinstance(actual_type, Numeric) else None
    expected_scale = expected_type.scale if isinstance(expected_type, Numeric) else None
    actual_scale = actual_type.scale if isinstance(actual_type, Numeric) else None
    if expected_precision is not None and (actual_precision != expected_precision or actual_scale != expected_scale):
        _schema_mismatch(table=table_name, category="capacity", column=expected.name)
    if dialect_name == "mysql" and isinstance(expected_type, String):
        expected_charset = _type_charset(expected_type)
        actual_charset = _type_charset(actual_type)
        expected_collation = _type_collation(expected_type)
        actual_collation = _type_collation(actual_type)
        if expected_charset is not None and actual_charset != expected_charset:
            _schema_mismatch(table=table_name, category="charset", column=expected.name)
        if expected_collation is not None and actual_collation != expected_collation:
            _schema_mismatch(table=table_name, category="collation", column=expected.name)


def _validate_server_default(
    dialect_name: str,
    *,
    table_name: str,
    expected: Any,
    actual: Mapping[str, Any],
) -> None:
    if expected.server_default is None:
        return
    expected_value = _normalize_current_timestamp_default(dialect_name, expected.server_default.arg)
    actual_value = _normalize_current_timestamp_default(dialect_name, actual.get("default"))
    if expected_value in {"current_timestamp", "current_timestamp(6)"}:
        allowed = {"current_timestamp", "current_timestamp()", "current_timestamp(6)"}
        if dialect_name == "postgresql":
            allowed.add("now()")
        if dialect_name == "mysql" and expected.name == "updated_at":
            allowed.add("current_timestamp(6)onupdatecurrent_timestamp(6)")
        if actual_value not in allowed:
            _schema_mismatch(table=table_name, category="server_default", column=expected.name)
        return
    if actual_value != expected_value:
        _schema_mismatch(table=table_name, category="server_default", column=expected.name)


def _foreign_key_signature(
    *,
    constrained_columns: tuple[str, ...],
    referred_schema: str | None,
    referred_table: str,
    referred_columns: tuple[str, ...],
    ondelete: str | None,
) -> tuple[object, ...]:
    return (
        constrained_columns,
        referred_schema,
        referred_table,
        referred_columns,
        ondelete.upper() if ondelete else None,
    )


def _normalize_current_timestamp_default(dialect_name: str, value: object) -> str | None:
    if value is None:
        return None
    if type(value).__name__ in {"AuditCurrentTimestamp", "AuditCreatedTimestamp"}:
        return "current_timestamp"
    text = "".join(str(value).split()).lower()
    while text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    return text


def _validate_mysql_options(inspector: Any, table_name: str, schema: str | None) -> None:
    options = inspector.get_table_options(table_name, schema=schema)
    normalized = {
        "mysql_engine": options.get("mysql_engine", options.get("engine")),
        "mysql_charset": options.get("mysql_charset", options.get("charset")),
        "mysql_collate": options.get("mysql_collate", options.get("collation")),
    }
    if {key: str(value).lower() if value is not None else None for key, value in normalized.items()} != {
        "mysql_engine": "innodb",
        "mysql_charset": "utf8mb4",
        "mysql_collate": "utf8mb4_bin",
    }:
        _schema_mismatch(table=table_name, category="mysql_options")


def _validate_autoincrement(
    dialect_name: str, table: Any, actual_columns: Mapping[str, Any], actual_pk: tuple[str, ...]
) -> None:
    if len(actual_pk) != 1:
        return
    expected = table.c[actual_pk[0]]
    if (
        not expected.primary_key
        or expected.autoincrement not in (True, "auto")
        or _type_family(expected.type) != "integer"
    ):
        return
    actual = actual_columns[actual_pk[0]]
    if dialect_name == "sqlite":
        if _type_family(actual.get("type")) != "integer":
            _schema_mismatch(table=table.name, category="autoincrement", column=actual_pk[0])
        return
    value = actual.get("autoincrement")
    if dialect_name == "postgresql":
        default = str(actual.get("default") or "").lower()
        identity = actual.get("identity")
        if not value and not identity and not default.startswith("nextval("):
            _schema_mismatch(table=table.name, category="autoincrement", column=actual_pk[0])
    elif not value and str(value).lower() not in {"auto", "auto_increment", "true"}:
        _schema_mismatch(table=table.name, category="autoincrement", column=actual_pk[0])


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


def _type_length(value: object) -> int | None:
    from sqlalchemy import CHAR, String

    if not isinstance(value, (CHAR, String)):
        return None
    return None if value.length is None else int(value.length)


def _type_charset(value: object) -> str | None:
    from sqlalchemy.dialects.mysql import (
        CHAR,
        LONGTEXT,
        MEDIUMTEXT,
        TEXT,
        TINYTEXT,
        VARCHAR,
    )

    if not isinstance(value, (CHAR, LONGTEXT, MEDIUMTEXT, TEXT, TINYTEXT, VARCHAR)):
        return None
    return value.charset


def _type_collation(value: object) -> str | None:
    from sqlalchemy import String

    return value.collation if isinstance(value, String) else None


def _type_name(value: object) -> str:
    return str(value).replace(" ", "").lower()


def _is_fixed_char(value: object) -> bool:
    return _type_name(value).startswith("char(") or type(value).__name__.lower() == "char"


def _is_unbounded_text(value: object) -> bool:
    name = type(value).__name__.lower()
    rendered = _type_name(value)
    return name in {"text", "longtext", "mediumtext", "tinytext", "clob"} or rendered in {
        "text",
        "longtext",
        "mediumtext",
        "tinytext",
        "clob",
    }


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
