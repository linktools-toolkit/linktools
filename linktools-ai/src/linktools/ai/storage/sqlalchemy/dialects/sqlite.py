#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite reference dialect, shared by every SQL store/backend.

The reference dialect core ships (alongside :class:`MySQLDialect` and
:class:`PostgreSQLDialect`). Uses SQLite's ``INSERT ... ON CONFLICT (...)
DO NOTHING`` so a conflicting insert returns ``inserted=False`` with NO
exception -- safe under a surrounding UoW transaction (aiosqlite commits a
SAVEPOINT immediately, so a savepoint-based recovery would poison UoW
rollback, which is why the ON CONFLICT path is mandatory under SQLite).

Downstreams with a different engine (their own vendor) implement
:class:`SqlAlchemyDialect` themselves and run the kernel conformance suite in
their own CI."""


from typing import Any, Mapping, Sequence
from .base import InsertResult, classify_integrity_error_by_message, primary_key_column

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import IntegrityViolationKind

class SqliteDialect:
    """The SQLite reference dialect. ``insert_ignore_conflict`` issues an
    ``ON CONFLICT (...) DO NOTHING`` insert into the given model's table; the
    resulting ``rowcount`` signals whether the row landed (1) or was skipped
    because a row with the same unique-column values already existed (0)."""

    @property
    def name(self) -> str:
        return "sqlite"

    async def insert_ignore_conflict(
        self,
        session: Any,
        *,
        model: type,
        values: "Mapping[str, Any]",
        index_elements: "Sequence[str]",
    ) -> InsertResult:
        from sqlalchemy.dialects.sqlite import insert

        stmt = (
            insert(model)
            .values(**values)
            .on_conflict_do_nothing(index_elements=list(index_elements))
            .returning(model.id)
        )
        result = await session.execute(stmt)
        row = result.first()
        return InsertResult(
            inserted=row is not None,
            row_id=row[0] if row is not None else None,
        )

    async def insert_ignore_conflict_many(
        self,
        session: Any,
        *,
        model: type,
        rows: "Sequence[Mapping[str, Any]]",
        index_elements: "Sequence[str]",
    ) -> None:
        from sqlalchemy.dialects.sqlite import insert

        stmt = (
            insert(model)
            .values(list(rows))
            .on_conflict_do_nothing(index_elements=list(index_elements))
        )
        await session.execute(stmt)

    async def upsert_increment(
        self,
        session: Any,
        *,
        model: type,
        pk: Any,
        column: str,
        step: int = 1,
    ) -> int:
        from sqlalchemy.dialects.sqlite import insert

        pk_column = primary_key_column(model)
        col_attr = getattr(model, column)
        # INSERT ... ON CONFLICT (pk) DO UPDATE: the row is seeded with
        # column = step on first insert; on conflict column = column + step.
        stmt = (
            insert(model)
            .values(**{pk_column.name: pk, column: step})
            .on_conflict_do_update(
                index_elements=[pk_column.name],
                set_={column: col_attr + step},
            )
            .returning(col_attr)
        )
        result = await session.execute(stmt)
        return result.scalar()

    async def upsert(
        self,
        session: Any,
        *,
        model: type,
        values: "Mapping[str, Any]",
        set_values: "Mapping[str, Any]",
        index_elements: "Sequence[str]",
    ) -> None:
        from sqlalchemy.dialects.sqlite import insert

        stmt = (
            insert(model)
            .values(**values)
            .on_conflict_do_update(
                index_elements=list(index_elements),
                set_=dict(set_values),
            )
        )
        await session.execute(stmt)

    async def upsert_many(
        self,
        session: Any,
        *,
        model: type,
        rows: "Sequence[Mapping[str, Any]]",
        set_columns: "Sequence[str]",
        index_elements: "Sequence[str]",
    ) -> None:
        from sqlalchemy.dialects.sqlite import insert

        stmt = insert(model).values(list(rows))
        stmt = stmt.on_conflict_do_update(
            index_elements=list(index_elements),
            set_={col: stmt.excluded[col] for col in set_columns},
        )
        await session.execute(stmt)

    def classify_integrity_error(
        self, error: BaseException
    ) -> "IntegrityViolationKind":
        return classify_integrity_error_by_message(
            error,
            unique_markers=("unique constraint",),
            foreign_key_markers=("foreign key constraint", "foreignkey"),
        )


__all__: "list[str]" = ["SqliteDialect"]
