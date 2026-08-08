#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vendor-specific SQLAlchemy statements used by storage backends."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

SqlValue: TypeAlias = str | int | bool | bytes | datetime | None

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sqlalchemy import Column, Table
    from sqlalchemy.engine import RowMapping
    from sqlalchemy.ext.asyncio import AsyncSession
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


@runtime_checkable
class SqlAlchemyDialect(Protocol):
    @property
    def name(self) -> str: ...

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
        primary_key: SqlValue,
        column: str,
        step: int = 1,
    ) -> int: ...

    async def delete_returning(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        where: "ColumnElement[bool]",
        returning: "Sequence[str]",
    ) -> "tuple[RowMapping, ...]": ...

    async def update_returning(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        where: "ColumnElement[bool]",
        values: "Mapping[str, SqlValue]",
        returning: "Sequence[str]",
    ) -> "tuple[RowMapping, ...]": ...

    def classify_integrity_error(self, error: BaseException) -> IntegrityViolationKind: ...


class SQLiteDialect:
    @property
    def name(self) -> str:
        return "sqlite"

    async def insert_ignore_conflict(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        values: "Mapping[str, SqlValue]",
        index_elements: "Sequence[str]",
    ) -> InsertResult:
        from sqlalchemy.dialects.sqlite import insert

        statement = insert(table).values(dict(values)).on_conflict_do_nothing(
            index_elements=list(index_elements)
        ).returning(table.c.id)
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

        statement = insert(table).values([dict(row) for row in rows]).on_conflict_do_nothing(
            index_elements=list(index_elements)
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

        statement = insert(table).values(dict(values)).on_conflict_do_update(
            index_elements=list(index_elements),
            set_=dict(set_values),
        )
        await session.execute(statement)

    async def upsert_increment(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        primary_key: SqlValue,
        column: str,
        step: int = 1,
    ) -> int:
        from sqlalchemy.dialects.sqlite import insert

        primary_key_column = _primary_key_column(table)
        value_column = table.c[column]
        statement = insert(table).values(
            {primary_key_column.name: primary_key, column: step}
        ).on_conflict_do_update(
            index_elements=[primary_key_column.name],
            set_={column: value_column + step},
        ).returning(value_column)
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

    async def update_returning(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        where: "ColumnElement[bool]",
        values: "Mapping[str, SqlValue]",
        returning: "Sequence[str]",
    ) -> "tuple[RowMapping, ...]":
        from sqlalchemy import update

        columns = [table.c[column] for column in returning]
        statement = update(table).where(where).values(dict(values)).returning(*columns)
        return tuple((await session.execute(statement)).mappings().all())

    def classify_integrity_error(self, error: BaseException) -> IntegrityViolationKind:
        return _classify_error(error, ("unique constraint", "unique failed"))


class PostgreSQLDialect(SQLiteDialect):
    @property
    def name(self) -> str:
        return "postgresql"

    async def insert_ignore_conflict(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        values: "Mapping[str, SqlValue]",
        index_elements: "Sequence[str]",
    ) -> InsertResult:
        from sqlalchemy.dialects.postgresql import insert

        statement = insert(table).values(dict(values)).on_conflict_do_nothing(
            index_elements=list(index_elements)
        ).returning(table.c.id)
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

        statement = insert(table).values([dict(row) for row in rows]).on_conflict_do_nothing(
            index_elements=list(index_elements)
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

        statement = insert(table).values(dict(values)).on_conflict_do_update(
            index_elements=list(index_elements),
            set_=dict(set_values),
        )
        await session.execute(statement)

    async def upsert_increment(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        primary_key: SqlValue,
        column: str,
        step: int = 1,
    ) -> int:
        from sqlalchemy.dialects.postgresql import insert

        primary_key_column = _primary_key_column(table)
        value_column = table.c[column]
        statement = insert(table).values(
            {primary_key_column.name: primary_key, column: step}
        ).on_conflict_do_update(
            index_elements=[primary_key_column.name],
            set_={column: value_column + step},
        ).returning(value_column)
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


class MySQLDialect(SQLiteDialect):
    @property
    def name(self) -> str:
        return "mysql"

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
        primary_key: SqlValue,
        column: str,
        step: int = 1,
    ) -> int:
        from sqlalchemy import func, select
        from sqlalchemy.dialects.mysql import insert

        primary_key_column = _primary_key_column(table)
        value_column = table.c[column]
        statement = insert(table).values(
            {primary_key_column.name: primary_key, column: func.last_insert_id(step)}
        ).on_duplicate_key_update(
            **{column: func.last_insert_id(value_column + step)}
        )
        result = await session.execute(statement)
        value = result.lastrowid
        if value:
            return int(value)
        return int(await session.scalar(select(value_column).where(primary_key_column == primary_key)))

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
            statement.on_duplicate_key_update(
                **{column: statement.inserted[column] for column in set_columns}
            )
        )

    async def delete_returning(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        where: "ColumnElement[bool]",
        returning: "Sequence[str]",
    ) -> "tuple[RowMapping, ...]":
        from sqlalchemy import delete, select

        columns = [table.c[column] for column in returning]
        rows = tuple((await session.execute(select(*columns).where(where))).mappings().all())
        if rows:
            await session.execute(delete(table).where(where))
        return rows

    async def update_returning(
        self,
        session: "AsyncSession",
        *,
        table: "Table",
        where: "ColumnElement[bool]",
        values: "Mapping[str, SqlValue]",
        returning: "Sequence[str]",
    ) -> "tuple[RowMapping, ...]":
        from sqlalchemy import select, update

        await session.execute(update(table).where(where).values(dict(values)))
        columns = [table.c[column] for column in returning]
        return tuple((await session.execute(select(*columns).where(where))).mappings().all())

    def classify_integrity_error(self, error: BaseException) -> IntegrityViolationKind:
        return _classify_error(error, ("duplicate entry", "1062"))


SqliteDialect = SQLiteDialect


def resolve_dialect(session: "AsyncSession") -> SqlAlchemyDialect:
    name = session.bind.dialect.name
    if name == "sqlite":
        return SQLiteDialect()
    if name in {"postgresql", "postgres"}:
        return PostgreSQLDialect()
    if name in {"mysql", "mariadb"}:
        return MySQLDialect()
    raise ValueError(f"unsupported SQLAlchemy dialect: {name}")


def _primary_key_column(table: "Table") -> "Column":
    return primary_key_column(table)


def primary_key_column(table: "Table") -> "Column":
    columns = tuple(table.primary_key.columns)
    if len(columns) != 1:
        raise ValueError("dialect counter tables require one primary-key column")
    return columns[0]


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
    "SqliteDialect",
    "SqlAlchemyDialect",
    "SqlValue",
    "classify_integrity_error_by_message",
    "primary_key_column",
    "resolve_dialect",
]
