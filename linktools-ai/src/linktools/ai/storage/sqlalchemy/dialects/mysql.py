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

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import IntegrityViolationKind, InsertResult

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
    conflicting row already existed (0)."""

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
        return InsertResult(inserted=result.rowcount == 1)

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
        message = str(orig or error).lower()
        if "duplicate entry" in message:
            return IntegrityViolationKind.UNIQUE_CONFLICT
        if "foreign key constraint" in message:
            return IntegrityViolationKind.FOREIGN_KEY
        if "check constraint" in message:
            return IntegrityViolationKind.CHECK
        return IntegrityViolationKind.UNKNOWN


__all__: "list[str]" = ["MySQLDialect"]
