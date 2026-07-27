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

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import IntegrityViolationKind, InsertResult


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
        )
        result = await session.execute(stmt)
        return InsertResult(inserted=result.rowcount == 1)

    def classify_integrity_error(
        self, error: BaseException
    ) -> IntegrityViolationKind:
        orig = getattr(error, "orig", None)
        message = str(orig or error).lower()
        # SQLite's unique-constraint message names the constraint column.
        if "unique constraint" in message:
            return IntegrityViolationKind.UNIQUE_CONFLICT
        if "foreign key constraint" in message or "foreignkey" in message:
            return IntegrityViolationKind.FOREIGN_KEY
        if "check constraint" in message:
            return IntegrityViolationKind.CHECK
        return IntegrityViolationKind.UNKNOWN


__all__: "list[str]" = ["SqliteDialect"]
