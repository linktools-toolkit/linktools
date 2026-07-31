#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MySQL dialect, shared by every SQL store/backend.

MySQL has no ``INSERT ... ON CONFLICT DO NOTHING``; the equivalent here is
``INSERT ... ON DUPLICATE KEY UPDATE col = col`` -- a no-op update (mirrors
the SQLite/PostgreSQL dialects' ``on_conflict_do_nothing(index_elements=
[...])``). MySQL's ``ON DUPLICATE KEY UPDATE`` fires on a conflict against
ANY unique key on the table, not just the named columns, but every model
this dialect targets has exactly one non-surrogate unique constraint, so
naming its first column as the no-op update target is unambiguous. MySQL
reports 0 affected rows when the no-op update runs against an existing row
and 1 when a fresh row is inserted, so the same ``rowcount == 1`` check the
other dialects use holds here too."""


from typing import Any, Mapping, Sequence

from .base import IntegrityViolationKind, InsertResult, classify_integrity_error_by_message, primary_key_column

# MySQL error codes (mysqlclient/PyMySQL/aiomysql/asyncmy all surface these as
# error.orig.args[0]).
_ER_DUP_ENTRY = 1062
_ER_NO_REFERENCED_ROW = 1451
_ER_ROW_IS_REFERENCED = 1452
_ER_CHECK_CONSTRAINT_VIOLATED = 3819


class MySQLDialect:
    """MySQL dialect. ``insert_ignore_conflict`` issues an ``ON DUPLICATE KEY
    UPDATE`` insert into the given model's table; the resulting ``rowcount``
    signals whether the row landed (1) or was left unchanged because a
    conflicting row already existed (0). ``upsert_increment`` issues a
    self-seeding ``INSERT ... ON DUPLICATE KEY UPDATE column =
    LAST_INSERT_ID(column + step)`` and reads the incremented value back from
    the driver's ``lastrowid`` in one statement (MySQL has no portable
    RETURNING)."""

    @property
    def name(self) -> str:
        return "mysql"

    async def insert_ignore_conflict(
        self,
        session: Any,
        *,
        model: type,
        values: "Mapping[str, Any]",
        index_elements: "Sequence[str]",
    ) -> InsertResult:
        from sqlalchemy.dialects.mysql import insert

        stmt = insert(model).values(**values)
        no_op_column = index_elements[0]
        stmt = stmt.on_duplicate_key_update(
            **{no_op_column: getattr(stmt.inserted, no_op_column)}
        )
        result = await session.execute(stmt)
        if result.rowcount == 1:
            # rowcount==1 means a fresh INSERT landed. The MySQL driver sets
            # cursor.lastrowid to the new auto-increment PK — this is
            # connection-local state set by the INSERT itself, zero extra
            # queries. (rowcount==2 would mean an existing row was updated;
            # rowcount==0 is impossible with ON DUPLICATE KEY UPDATE.)
            lastrowid = getattr(result, "lastrowid", None)
            return InsertResult(inserted=True, row_id=lastrowid or None)
        return InsertResult(inserted=False)

    async def insert_ignore_conflict_many(
        self,
        session: Any,
        *,
        model: type,
        rows: "Sequence[Mapping[str, Any]]",
        index_elements: "Sequence[str]",
    ) -> None:
        from sqlalchemy.dialects.mysql import insert

        # ON DUPLICATE KEY UPDATE with a no-op (col = col) mirrors the
        # single-row insert_ignore_conflict; fires on ANY unique key, same
        # single-unique-key assumption. rowcount is not inspected (batch caller
        # does not need per-row insert flags).
        no_op_column = index_elements[0]
        stmt = insert(model).values(list(rows))
        stmt = stmt.on_duplicate_key_update(
            **{no_op_column: getattr(stmt.inserted, no_op_column)}
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
        # MySQL has no portable RETURNING (only >= 8.0.21, and the project
        # keeps no minimum-version promise). The LAST_INSERT_ID(expr) idiom
        # makes the incremented value observable in one statement on BOTH
        # paths: wrap step in LAST_INSERT_ID on the INSERT values so a fresh
        # row's connection id == step, and LAST_INSERT_ID(col + step) on the
        # UPDATE branch so a conflict's id == existing + step. Either way the
        # server exposes the value via the C-API mysql_insert_id() and every
        # compliant driver forwards it as cursor.lastrowid. (Wrapping step on
        # the INSERT side is what makes the INSERT path correct -- without it
        # lastrowid would be the surrogate PK, not step.) Row-level locking
        # makes the increment itself race-free.
        from sqlalchemy import func, select
        from sqlalchemy.dialects.mysql import insert

        pk_column = primary_key_column(model)
        col_attr = getattr(model, column)
        result = await session.execute(
            insert(model)
            .values(
                **{
                    pk_column.name: pk,
                    column: func.last_insert_id(step),
                }
            )
            .on_duplicate_key_update(
                **{
                    column: func.last_insert_id(col_attr + step),
                }
            )
        )
        value = getattr(result, "lastrowid", None)
        if value:
            return int(value)
        # Defensive: a driver that does not forward LAST_INSERT_ID(expr) on the
        # UPDATE path reports lastrowid==0 -- read it back in the same tx so
        # the value is never wrong. This branch is unreachable under any
        # spec-compliant MySQL driver.
        return await session.scalar(select(col_attr).where(pk_column == pk))

    async def upsert(
        self,
        session: Any,
        *,
        model: type,
        values: "Mapping[str, Any]",
        set_values: "Mapping[str, Any]",
        index_elements: "Sequence[str]",
    ) -> None:
        from sqlalchemy.dialects.mysql import insert

        # MySQL's ON DUPLICATE KEY UPDATE fires on a conflict against ANY unique
        # key on the table, not just the named columns -- but every model this
        # dialect targets has exactly one non-surrogate unique constraint
        # (same assumption insert_ignore_conflict relies on), so the update
        # target is unambiguous.
        stmt = insert(model).values(**values).on_duplicate_key_update(**set_values)
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
        from sqlalchemy.dialects.mysql import insert

        # ON DUPLICATE KEY UPDATE with inserted.col references: each row's
        # conflict-branch update uses that row's own proposed value (VALUES(col)
        # idiom under MySQL). Same single-unique-key assumption as upsert.
        stmt = insert(model).values(list(rows))
        stmt = stmt.on_duplicate_key_update(
            **{col: stmt.inserted[col] for col in set_columns}
        )
        await session.execute(stmt)

    def classify_integrity_error(
        self, error: BaseException
    ) -> IntegrityViolationKind:
        orig = getattr(error, "orig", None)
        args = getattr(orig, "args", None)
        code = args[0] if args and isinstance(args[0], int) else None
        if code == _ER_DUP_ENTRY:
            return IntegrityViolationKind.UNIQUE_CONFLICT
        if code in (_ER_NO_REFERENCED_ROW, _ER_ROW_IS_REFERENCED):
            return IntegrityViolationKind.FOREIGN_KEY
        if code == _ER_CHECK_CONSTRAINT_VIOLATED:
            return IntegrityViolationKind.CHECK
        # Fall back to message sniffing for drivers that don't expose a
        # numeric error code the same way (or wrap it differently).
        return classify_integrity_error_by_message(
            error, unique_markers=("duplicate entry",)
        )


__all__: "list[str]" = ["MySQLDialect"]
