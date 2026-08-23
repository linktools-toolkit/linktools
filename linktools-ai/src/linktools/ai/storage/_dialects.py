"""Vendor-specific SQLAlchemy column types, statements, and engine configuration."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

from linktools.core import environ

from ..errors import AIError, ErrorCode

SqlValue: TypeAlias = str | int | bool | bytes | datetime | None
_logger = environ.get_logger("ai.storage.dialects")
_SQL_DELETE_RETURNING_BATCH_LIMIT = 64


class _SqliteCursor(Protocol):
    def execute(self, statement: str) -> object: ...

    def close(self) -> None: ...


class _SqliteConnection(Protocol):
    def cursor(self) -> _SqliteCursor: ...


class _SqliteEventValue(Protocol):
    pass

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sqlalchemy import (
        CHAR,
        BigInteger,
        Column,
        Index,
        LargeBinary,
        String,
        Table,
    )
    from sqlalchemy.engine import Connection, RowMapping
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
    from sqlalchemy.sql.elements import ColumnElement


@dataclass(frozen=True, slots=True)
class InsertResult:
    inserted: bool
    row_id: "int | None" = None


class IntegrityViolationKind(StrEnum):
    UNIQUE_CONFLICT = "unique_conflict"
    FOREIGN_KEY = "foreign_key"
    CHECK = "check"
    UNKNOWN = "unknown"


class SqlErrorKind(StrEnum):
    INTEGRITY = "integrity"
    RETRYABLE_TRANSACTION = "retryable_transaction"
    DATABASE = "database"
    UNKNOWN = "unknown"


class SqlTransactionPhase(StrEnum):
    BODY = "body"
    COMMIT = "commit"


class SqlTransactionDisposition(StrEnum):
    RETRYABLE_ABORTED = "retryable_aborted"
    NONRETRYABLE_ABORTED = "nonretryable_aborted"
    COMMIT_UNKNOWN = "commit_unknown"


@runtime_checkable
class SqlAlchemyDialect(Protocol):
    @property
    def name(self) -> str: ...

    async def database_now(self, session: "AsyncSession") -> datetime: ...

    async def insert_ignore_conflict(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        values: "Mapping[str, SqlValue]",
        index_elements: "Sequence[str]",
    ) -> InsertResult: ...

    async def insert_ignore_conflict_many(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        rows: "Sequence[Mapping[str, SqlValue]]",
        index_elements: "Sequence[str]",
    ) -> None: ...

    async def upsert(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        values: "Mapping[str, SqlValue]",
        set_values: "Mapping[str, SqlValue]",
        index_elements: "Sequence[str]",
    ) -> None: ...

    async def upsert_many(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        rows: "Sequence[Mapping[str, SqlValue]]",
        set_columns: "Sequence[str]",
        index_elements: "Sequence[str]",
    ) -> None: ...

    async def upsert_increment(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        values: "Mapping[str, SqlValue]",
        column: str,
        index_elements: "Sequence[str]",
        step: int = 1,
    ) -> int: ...

    async def upsert_increment_many(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        rows: "Sequence[Mapping[str, SqlValue]]",
        column: str,
        index_elements: "Sequence[str]",
    ) -> "Mapping[str, int]": ...

    async def delete_returning(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        where: "ColumnElement[bool]",
        returning: "Sequence[str]",
    ) -> "tuple[RowMapping, ...]": ...

    def classify_integrity_error(self, error: BaseException) -> IntegrityViolationKind: ...

    def classify_transaction_error(
        self,
        error: BaseException,
        *,
        phase: SqlTransactionPhase,
        connection_invalidated: bool,
    ) -> SqlTransactionDisposition: ...


class SQLiteDialect:
    @property
    def name(self) -> str:
        return "sqlite"

    async def database_now(self, session: "AsyncSession") -> datetime:
        from sqlalchemy import select

        _logger.debug("SQL authoritative time queried: dialect=%s", self.name)
        value = await session.scalar(select(self._database_now_expression()))
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
            except ValueError as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if not isinstance(value, datetime):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)

    def _database_now_expression(self) -> "ColumnElement[str]":
        from sqlalchemy import func

        return func.strftime("%Y-%m-%dT%H:%M:%f", "now")

    async def insert_ignore_conflict(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        values: "Mapping[str, SqlValue]",
        index_elements: "Sequence[str]",
    ) -> InsertResult:
        from sqlalchemy.dialects.sqlite import insert

        statement = (
            insert(table)
            .values(dict(values))
            .on_conflict_do_nothing(index_elements=list(index_elements))
            .returning(table.c.id)
        )
        row = (await session.execute(statement)).first()
        return InsertResult(row is not None, None if row is None else int(row[0]))

    async def insert_ignore_conflict_many(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        rows: "Sequence[Mapping[str, SqlValue]]",
        index_elements: "Sequence[str]",
    ) -> None:
        if not rows:
            return
        from sqlalchemy.dialects.sqlite import insert

        statement = (
            insert(table)
            .values([dict(row) for row in rows])
            .on_conflict_do_nothing(index_elements=list(index_elements))
        )
        await session.execute(statement)

    async def upsert(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        values: "Mapping[str, SqlValue]",
        set_values: "Mapping[str, SqlValue]",
        index_elements: "Sequence[str]",
    ) -> None:
        from sqlalchemy.dialects.sqlite import insert

        statement = (
            insert(table)
            .values(dict(values))
            .on_conflict_do_update(
                index_elements=list(index_elements),
                set_=dict(set_values),
            )
        )
        await session.execute(statement)

    async def upsert_increment(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        values: "Mapping[str, SqlValue]",
        column: str,
        index_elements: "Sequence[str]",
        step: int = 1,
    ) -> int:
        from sqlalchemy import func
        from sqlalchemy.dialects.sqlite import insert

        value_column = table.c[column]
        insert_values = dict(values)
        insert_values[column] = step
        set_values = {column: value_column + step}
        if "updated_at" in table.c:
            set_values["updated_at"] = func.current_timestamp()
        statement = (
            insert(table)
            .values(insert_values)
            .on_conflict_do_update(
                index_elements=list(index_elements),
                set_=set_values,
            )
            .returning(value_column)
        )
        return int((await session.execute(statement)).scalar_one())

    async def upsert_many(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        rows: "Sequence[Mapping[str, SqlValue]]",
        set_columns: "Sequence[str]",
        index_elements: "Sequence[str]",
    ) -> None:
        if not rows:
            return
        from sqlalchemy.dialects.sqlite import insert

        insert_statement = insert(table).values([dict(row) for row in rows])
        statement = insert_statement.on_conflict_do_update(
            index_elements=list(index_elements),
            set_={column: insert_statement.excluded[column] for column in set_columns},
        )
        await session.execute(statement)

    async def upsert_increment_many(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        rows: "Sequence[Mapping[str, SqlValue]]",
        column: str,
        index_elements: "Sequence[str]",
    ) -> "Mapping[str, int]":
        if not rows:
            return {}
        from sqlalchemy.dialects.sqlite import insert

        value_column = table.c[column]
        insert_statement = insert(table).values([dict(row) for row in rows])
        set_values = {column: value_column + insert_statement.excluded[column]}
        if "updated_at" in table.c:
            from sqlalchemy import func

            set_values["updated_at"] = func.current_timestamp()
        statement = (
            insert_statement
            .on_conflict_do_update(
                index_elements=list(index_elements),
                set_=set_values,
            )
            .returning(table.c[index_elements[0]], value_column)
        )
        result = (await session.execute(statement)).all()
        _logger.debug(
            "SQL batch executed: backend=%s operation=reserve_sequences "
            "batch_size=%s statement_count=1",
            self.name,
            len(rows),
        )
        return {str(row[0]): int(row[1]) for row in result}

    async def delete_returning(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        where: "ColumnElement[bool]",
        returning: "Sequence[str]",
    ) -> "tuple[RowMapping, ...]":
        from sqlalchemy import delete

        columns = [table.c[column] for column in returning]
        statement = delete(table).where(where).returning(*columns)
        return tuple((await session.execute(statement)).mappings().all())

    def classify_integrity_error(self, error: BaseException) -> IntegrityViolationKind:
        return _classify_error(error, ("unique constraint", "unique failed"))

    def classify_transaction_error(
        self,
        error: BaseException,
        *,
        phase: SqlTransactionPhase,
        connection_invalidated: bool,
    ) -> SqlTransactionDisposition:
        return _classify_transaction_error(error, phase, connection_invalidated)


class PostgreSQLDialect(SQLiteDialect):
    @property
    def name(self) -> str:
        return "postgresql"

    def _database_now_expression(self) -> "ColumnElement[datetime]":
        from sqlalchemy import func

        return func.now()

    async def insert_ignore_conflict(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        values: "Mapping[str, SqlValue]",
        index_elements: "Sequence[str]",
    ) -> InsertResult:
        from sqlalchemy.dialects.postgresql import insert

        statement = (
            insert(table)
            .values(dict(values))
            .on_conflict_do_nothing(index_elements=list(index_elements))
            .returning(table.c.id)
        )
        row = (await session.execute(statement)).first()
        return InsertResult(row is not None, None if row is None else int(row[0]))

    async def insert_ignore_conflict_many(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        rows: "Sequence[Mapping[str, SqlValue]]",
        index_elements: "Sequence[str]",
    ) -> None:
        if not rows:
            return
        from sqlalchemy.dialects.postgresql import insert

        statement = (
            insert(table)
            .values([dict(row) for row in rows])
            .on_conflict_do_nothing(index_elements=list(index_elements))
        )
        await session.execute(statement)

    async def upsert(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        values: "Mapping[str, SqlValue]",
        set_values: "Mapping[str, SqlValue]",
        index_elements: "Sequence[str]",
    ) -> None:
        from sqlalchemy.dialects.postgresql import insert

        statement = (
            insert(table)
            .values(dict(values))
            .on_conflict_do_update(
                index_elements=list(index_elements),
                set_=dict(set_values),
            )
        )
        await session.execute(statement)

    async def upsert_increment(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        values: "Mapping[str, SqlValue]",
        column: str,
        index_elements: "Sequence[str]",
        step: int = 1,
    ) -> int:
        from sqlalchemy import func
        from sqlalchemy.dialects.postgresql import insert

        value_column = table.c[column]
        insert_values = dict(values)
        insert_values[column] = step
        set_values = {column: value_column + step}
        if "updated_at" in table.c:
            set_values["updated_at"] = func.current_timestamp()
        statement = (
            insert(table)
            .values(insert_values)
            .on_conflict_do_update(
                index_elements=list(index_elements),
                set_=set_values,
            )
            .returning(value_column)
        )
        return int((await session.execute(statement)).scalar_one())

    async def upsert_many(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        rows: "Sequence[Mapping[str, SqlValue]]",
        set_columns: "Sequence[str]",
        index_elements: "Sequence[str]",
    ) -> None:
        if not rows:
            return
        from sqlalchemy.dialects.postgresql import insert

        insert_statement = insert(table).values([dict(row) for row in rows])
        statement = insert_statement.on_conflict_do_update(
            index_elements=list(index_elements),
            set_={column: insert_statement.excluded[column] for column in set_columns},
        )
        await session.execute(statement)

    async def upsert_increment_many(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        rows: "Sequence[Mapping[str, SqlValue]]",
        column: str,
        index_elements: "Sequence[str]",
    ) -> "Mapping[str, int]":
        if not rows:
            return {}
        from sqlalchemy.dialects.postgresql import insert

        value_column = table.c[column]
        insert_statement = insert(table).values([dict(row) for row in rows])
        set_values = {column: value_column + insert_statement.excluded[column]}
        if "updated_at" in table.c:
            from sqlalchemy import func

            set_values["updated_at"] = func.current_timestamp()
        statement = (
            insert_statement
            .on_conflict_do_update(
                index_elements=list(index_elements),
                set_=set_values,
            )
            .returning(table.c[index_elements[0]], value_column)
        )
        result = (await session.execute(statement)).all()
        _logger.debug(
            "SQL batch executed: backend=%s operation=reserve_sequences "
            "batch_size=%s statement_count=1",
            self.name,
            len(rows),
        )
        return {str(row[0]): int(row[1]) for row in result}


class MySQLDialect(SQLiteDialect):
    @property
    def name(self) -> str:
        return "mysql"

    def _database_now_expression(self) -> "ColumnElement[datetime]":
        from sqlalchemy import func

        return func.utc_timestamp(6)

    async def insert_ignore_conflict(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        values: "Mapping[str, SqlValue]",
        index_elements: "Sequence[str]",
    ) -> InsertResult:
        from sqlalchemy.dialects.mysql import insert

        statement = insert(table).values(dict(values))
        column = index_elements[0]
        statement = statement.on_duplicate_key_update(**{column: statement.inserted[column]})
        result = await session.execute(statement)
        return InsertResult(result.rowcount == 1, None)

    async def insert_ignore_conflict_many(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        rows: "Sequence[Mapping[str, SqlValue]]",
        index_elements: "Sequence[str]",
    ) -> None:
        if not rows:
            return
        from sqlalchemy.dialects.mysql import insert

        statement = insert(table).values([dict(row) for row in rows])
        column = index_elements[0]
        await session.execute(statement.on_duplicate_key_update(**{column: statement.inserted[column]}))

    async def upsert(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        values: "Mapping[str, SqlValue]",
        set_values: "Mapping[str, SqlValue]",
        index_elements: "Sequence[str]",
    ) -> None:
        from sqlalchemy.dialects.mysql import insert

        statement = insert(table).values(dict(values)).on_duplicate_key_update(**dict(set_values))
        await session.execute(statement)

    async def upsert_increment(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        values: "Mapping[str, SqlValue]",
        column: str,
        index_elements: "Sequence[str]",
        step: int = 1,
    ) -> int:
        from sqlalchemy import func, select
        from sqlalchemy.dialects.mysql import insert

        value_column = table.c[column]
        insert_values = dict(values)
        insert_values[column] = step
        set_values = {column: func.last_insert_id(value_column + step)}
        if "updated_at" in table.c:
            set_values["updated_at"] = func.current_timestamp()
        statement = insert(table).values(insert_values).on_duplicate_key_update(**set_values)
        result = await session.execute(statement)
        if result.rowcount == 1:
            return step
        value = result.lastrowid
        if value:
            return int(value)
        predicates = [table.c[index] == values[index] for index in index_elements]
        return int(await session.scalar(select(value_column).where(*predicates)))

    async def upsert_many(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        rows: "Sequence[Mapping[str, SqlValue]]",
        set_columns: "Sequence[str]",
        index_elements: "Sequence[str]",
    ) -> None:
        if not rows:
            return
        from sqlalchemy.dialects.mysql import insert

        statement = insert(table).values([dict(row) for row in rows])
        await session.execute(
            statement.on_duplicate_key_update(**{column: statement.inserted[column] for column in set_columns})
        )

    async def upsert_increment_many(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        rows: "Sequence[Mapping[str, SqlValue]]",
        column: str,
        index_elements: "Sequence[str]",
    ) -> "Mapping[str, int]":
        if not rows:
            return {}
        from sqlalchemy import select
        from sqlalchemy.dialects.mysql import insert

        insert_statement = insert(table).values([dict(row) for row in rows])
        value_column = table.c[column]
        set_values = {column: value_column + insert_statement.inserted[column]}
        if "updated_at" in table.c:
            from sqlalchemy import func

            set_values["updated_at"] = func.current_timestamp()
        await session.execute(insert_statement.on_duplicate_key_update(**set_values))
        key_column = index_elements[0]
        result = await session.execute(
            select(table.c[key_column], value_column).where(
                table.c[key_column].in_([row[key_column] for row in rows])
            )
        )
        _logger.debug(
            "SQL batch executed: backend=%s operation=reserve_sequences "
            "batch_size=%s statement_count=2",
            self.name,
            len(rows),
        )
        return {str(row[0]): int(row[1]) for row in result.all()}

    async def delete_returning(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        where: "ColumnElement[bool]",
        returning: "Sequence[str]",
    ) -> "tuple[RowMapping, ...]":
        from sqlalchemy import and_, delete, or_, select

        primary_columns = tuple(table.primary_key.columns)
        if not primary_columns:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return_columns = tuple(table.c[column] for column in returning)
        selected = tuple(
            (await session.execute(select(*primary_columns, *return_columns).where(where))).mappings().all()
        )
        if not selected:
            return ()

        deleted_count = 0
        for offset in range(0, len(selected), _SQL_DELETE_RETURNING_BATCH_LIMIT):
            batch = selected[offset : offset + _SQL_DELETE_RETURNING_BATCH_LIMIT]
            predicates = []
            for row in batch:
                values = []
                for column in (*primary_columns, *return_columns):
                    value = row[column.key]
                    values.append(column.is_(None) if value is None else column == value)
                predicates.append(and_(*values))
            result = await session.execute(delete(table).where(and_(where, or_(*predicates))))
            deleted_count += result.rowcount
        if deleted_count != len(selected):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return tuple({column: row[column] for column in returning} for row in selected)

    def classify_integrity_error(self, error: BaseException) -> IntegrityViolationKind:
        return _classify_error(error, ("duplicate entry", "1062"))

    def classify_transaction_error(
        self,
        error: BaseException,
        *,
        phase: SqlTransactionPhase,
        connection_invalidated: bool,
    ) -> SqlTransactionDisposition:
        return _classify_transaction_error(error, phase, connection_invalidated)


def dialect_for_name(name: str) -> SqlAlchemyDialect:
    if name == "sqlite":
        return SQLiteDialect()
    if name == "mysql":
        return MySQLDialect()
    if name == "postgresql":
        return PostgreSQLDialect()
    raise ValueError(f"unsupported SQLAlchemy dialect: {name}")


def resolve_dialect(session: "AsyncSession") -> SqlAlchemyDialect:
    bind = session.get_bind()
    return dialect_for_name(bind.dialect.name)


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


async def configure_sqlite_engine(engine: "AsyncEngine") -> None:
    if engine.dialect.name != "sqlite":
        raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
    from sqlalchemy import event

    sync_engine = engine.sync_engine
    if not event.contains(sync_engine, "checkout", configure_sqlite_connection):
        event.listen(sync_engine, "checkout", configure_sqlite_connection)
        _logger.debug("SQLite checkout PRAGMA listener registered")


def configure_sqlite_connection(
    dbapi_connection: _SqliteConnection,
    _connection_record: _SqliteEventValue,
    _connection_proxy: _SqliteEventValue,
) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
    finally:
        cursor.close()


def column_type_matches(
    connection: "Connection",
    *,
    expected: "Column",
    actual: "Mapping[str, object]",
) -> bool:
    """Return whether an expected column type is compatible with the reflected one."""
    from sqlalchemy import JSON, LargeBinary

    expected_type = expected.type.dialect_impl(connection.dialect)
    actual_type = actual.get("type")
    dialect_name = connection.dialect.name
    if _type_family(expected_type) != _type_family(actual_type) and not _boolean_compatible(
        dialect_name, expected_type, actual_type
    ):
        return False
    if isinstance(expected_type, LargeBinary) and not _binary_compatible(dialect_name, expected_type, actual_type):
        return False
    if isinstance(expected_type, JSON) and not _json_compatible(expected_type, actual_type):
        return False
    return not (_type_family(expected_type) == "integer" and not _integer_compatible(dialect_name, expected_type, actual_type))


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


def classify_integrity_error_by_message(
    error: BaseException,
    *,
    unique_markers: "Sequence[str]",
    foreign_key_markers: "Sequence[str]" = ("foreign key",),
    check_markers: "Sequence[str]" = ("check constraint",),
) -> IntegrityViolationKind:
    message = str(error).lower()
    if any(marker.lower() in message for marker in unique_markers):
        return IntegrityViolationKind.UNIQUE_CONFLICT
    if any(marker.lower() in message for marker in foreign_key_markers):
        return IntegrityViolationKind.FOREIGN_KEY
    if any(marker.lower() in message for marker in check_markers):
        return IntegrityViolationKind.CHECK
    return IntegrityViolationKind.UNKNOWN


def classify_sql_error(error: BaseException) -> SqlErrorKind:
    from sqlalchemy.exc import DBAPIError, IntegrityError

    if isinstance(error, IntegrityError):
        return SqlErrorKind.INTEGRITY
    if not isinstance(error, DBAPIError):
        return SqlErrorKind.UNKNOWN
    original = error.orig
    if _is_sqlite_busy(original):
        return SqlErrorKind.RETRYABLE_TRANSACTION
    if _read_sqlstate(original) in {"40001", "40P01"}:
        return SqlErrorKind.RETRYABLE_TRANSACTION
    if _read_mysql_errno(original) in {1205, 1213}:
        return SqlErrorKind.RETRYABLE_TRANSACTION
    if _retryable_message_fallback(original):
        return SqlErrorKind.RETRYABLE_TRANSACTION
    return SqlErrorKind.DATABASE


def is_retryable_sql_transaction(error: BaseException) -> bool:
    return classify_sql_error(error) is SqlErrorKind.RETRYABLE_TRANSACTION


def _classify_transaction_error(
    error: BaseException,
    phase: SqlTransactionPhase,
    connection_invalidated: bool,
) -> SqlTransactionDisposition:
    from sqlalchemy.exc import IntegrityError

    if isinstance(error, AIError):
        return SqlTransactionDisposition.NONRETRYABLE_ABORTED
    if isinstance(error, IntegrityError):
        return SqlTransactionDisposition.NONRETRYABLE_ABORTED
    if classify_sql_error(error) is SqlErrorKind.RETRYABLE_TRANSACTION:
        if phase is SqlTransactionPhase.BODY:
            return SqlTransactionDisposition.RETRYABLE_ABORTED
        if not connection_invalidated:
            return SqlTransactionDisposition.RETRYABLE_ABORTED
        return SqlTransactionDisposition.COMMIT_UNKNOWN
    if phase is SqlTransactionPhase.COMMIT:
        return SqlTransactionDisposition.COMMIT_UNKNOWN
    return SqlTransactionDisposition.NONRETRYABLE_ABORTED


def _is_sqlite_busy(error: BaseException) -> bool:
    try:
        code = error.sqlite_errorcode
    except AttributeError:
        code = None
    if isinstance(code, int) and (code & 0xFF) == 5:
        return True
    try:
        name = error.sqlite_errorname
    except AttributeError:
        return False
    return isinstance(name, str) and name.upper().startswith("SQLITE_BUSY")


def _read_sqlstate(error: BaseException) -> "str | None":
    try:
        value = error.sqlstate
    except AttributeError:
        pass
    else:
        if value is not None:
            return str(value).upper()
    try:
        value = error.pgcode
    except AttributeError:
        pass
    else:
        if value is not None:
            return str(value).upper()
    return None


def _read_mysql_errno(error: BaseException) -> "int | None":
    try:
        value = error.errno
    except AttributeError:
        value = None
    if value is None:
        value = error.args[0] if error.args else None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _retryable_message_fallback(error: BaseException) -> bool:
    values = [str(error).lower()]
    values.extend(str(value).lower() for value in error.args)
    message = " ".join(values)
    return any(
        marker in message
        for marker in (
            "database is locked",
            "database is busy",
            "could not serialize access",
            "serialization failure",
            "deadlock detected",
            "deadlock found",
            "lock wait timeout exceeded",
        )
    )


def _classify_error(error: BaseException, unique_markers: "tuple[str, ...]") -> IntegrityViolationKind:
    message = str(error).lower()
    if any(marker in message for marker in unique_markers):
        return IntegrityViolationKind.UNIQUE_CONFLICT
    if "foreign key" in message or "1451" in message or "1452" in message or "23503" in message:
        return IntegrityViolationKind.FOREIGN_KEY
    if "check constraint" in message or "3819" in message or "23514" in message:
        return IntegrityViolationKind.CHECK
    return IntegrityViolationKind.UNKNOWN


__all__ = [
    "InsertResult",
    "IntegrityViolationKind",
    "MySQLDialect",
    "PostgreSQLDialect",
    "SQLiteDialect",
    "SqlAlchemyDialect",
    "SqlErrorKind",
    "SqlTransactionDisposition",
    "SqlTransactionPhase",
    "SqlValue",
    "classify_integrity_error_by_message",
    "classify_sql_error",
    "column_type_matches",
    "configure_sqlite_engine",
    "dialect_for_name",
    "is_retryable_sql_transaction",
    "resolve_dialect",
    "sql_audit_columns",
    "sql_audit_indexes",
    "sql_blob",
    "sql_digest",
    "sql_id_column",
    "sql_integer_id",
    "sql_query_index",
    "sql_sha256",
    "sql_sort_key",
    "sql_state",
    "sql_table_options",
    "sql_text_key",
    "sql_unique",
]
