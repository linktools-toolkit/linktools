#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite reference dialect for the storage kernel's object backend.

The ONE concrete dialect core ships. Uses SQLite's ``INSERT ... ON CONFLICT
(key_hash) DO NOTHING`` so a conflicting insert returns ``inserted=False``
with NO exception -- safe under a surrounding UoW transaction (aiosqlite
commits a SAVEPOINT immediately, so a savepoint-based recovery would poison
UoW rollback, which is why the ON CONFLICT path is mandatory under SQLite).

Downstreams with a different engine (their own vendor) implement
:class:`SqlAlchemyObjectDialect` themselves and run the kernel conformance
suite in their own CI. Core deliberately ships no MySQL / PostgreSQL /
Oracle / etc. dialect; those belong to the downstream that owns the engine."""

from __future__ import annotations

from typing import Any, Mapping

from .base import IntegrityViolationKind, InsertResult


class SqliteObjectDialect:
    """The SQLite reference dialect. ``insert_current_if_absent`` issues an
    ``ON CONFLICT (key_hash) DO NOTHING`` insert into ``storage_objects``;
    the resulting ``rowcount`` signals whether the row landed (1) or was
    skipped because a row with the same key_hash already existed (0)."""

    @property
    def name(self) -> str:
        return "sqlite"

    async def insert_current_if_absent(
        self,
        session: Any,
        *,
        values: "Mapping[str, Any]",
    ) -> InsertResult:
        from sqlalchemy.dialects.sqlite import insert

        # Late import: keep this module importable without SQLAlchemy + the
        # backend models; the dialect is reached only when the storage kernel
        # actually wires it into a SqlAlchemyObjectBackend.
        from ...backends.sqlalchemy.models import StorageObjectRow

        stmt = (
            insert(StorageObjectRow)
            .values(**values)
            .on_conflict_do_nothing(index_elements=["key_hash"])
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


__all__: "list[str]" = ["SqliteObjectDialect"]
