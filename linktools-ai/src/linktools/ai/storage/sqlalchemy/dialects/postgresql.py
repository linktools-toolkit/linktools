#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PostgreSQL dialect, shared by every SQL store/backend.

PostgreSQL originated the ``INSERT ... ON CONFLICT (col) DO NOTHING`` syntax
SQLite later adopted, so this mirrors the SQLite reference dialect almost
exactly -- the only real difference is error classification, which reads
Postgres's SQLSTATE codes instead of SQLite's message text."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import IntegrityViolationKind, InsertResult

# Postgres SQLSTATE codes (asyncpg exposes these as error.sqlstate; psycopg2
# as error.orig.pgcode; psycopg3 as error.orig.sqlstate / .diag.sqlstate_).
_UNIQUE_VIOLATION = "23505"
_FOREIGN_KEY_VIOLATION = "23503"
_CHECK_VIOLATION = "23514"


class PostgreSQLDialect:
    """PostgreSQL dialect. ``insert_ignore_conflict`` issues an ``ON CONFLICT
    (...) DO NOTHING`` insert into the given model's table; the resulting
    ``rowcount`` signals whether the row landed (1) or was skipped because a
    row with the same unique-column values already existed (0)."""

    @property
    def name(self) -> str:
        return "postgresql"

    async def insert_ignore_conflict(
        self,
        session: Any,
        *,
        model: type,
        values: "Mapping[str, Any]",
        index_elements: "Sequence[str]",
    ) -> InsertResult:
        from sqlalchemy.dialects.postgresql import insert

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
        code = (
            getattr(orig, "sqlstate", None)
            or getattr(orig, "pgcode", None)
            or getattr(getattr(orig, "diag", None), "sqlstate", None)
        )
        if code == _UNIQUE_VIOLATION:
            return IntegrityViolationKind.UNIQUE_CONFLICT
        if code == _FOREIGN_KEY_VIOLATION:
            return IntegrityViolationKind.FOREIGN_KEY
        if code == _CHECK_VIOLATION:
            return IntegrityViolationKind.CHECK
        # Fall back to message sniffing for drivers that don't expose a
        # SQLSTATE the same way (or wrap it differently).
        message = str(orig or error).lower()
        if "unique constraint" in message or "duplicate key" in message:
            return IntegrityViolationKind.UNIQUE_CONFLICT
        if "foreign key constraint" in message:
            return IntegrityViolationKind.FOREIGN_KEY
        if "check constraint" in message:
            return IntegrityViolationKind.CHECK
        return IntegrityViolationKind.UNKNOWN


__all__: "list[str]" = ["PostgreSQLDialect"]
