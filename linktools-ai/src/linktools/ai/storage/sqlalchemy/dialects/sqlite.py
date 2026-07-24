#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite dialect strategy.

Conflict-detecting INSERT uses SQLite's ``ON CONFLICT DO NOTHING``: a conflicting
insert reports ``rowcount == 0`` with NO exception, so the surrounding UoW
transaction is never poisoned and there is no SAVEPOINT to leak. (aiosqlite
commits a SAVEPOINT immediately, so the MySQL-style savepoint recovery would
break UoW rollback here.) Classification maps the column-based ``UNIQUE
constraint failed: ai_assets.path_hash`` message for the rare path that still
raises.
"""

from .base import IntegrityViolationKind


class SqliteDialectStrategy:
    name = "sqlite"

    def classify_integrity_error(self, error: BaseException) -> IntegrityViolationKind:
        orig = getattr(error, "orig", None)
        message = str(orig or error)
        if "ai_assets.path_hash" in message:
            return IntegrityViolationKind.ASSET_KEY
        if "ai_asset_idempotency.key" in message:
            return IntegrityViolationKind.IDEMPOTENCY_KEY
        return IntegrityViolationKind.OTHER

    async def execute_conflict_insert(
        self, session, table, values, *, index_elements
    ) -> bool:
        from sqlalchemy.dialects.sqlite import insert

        stmt = (
            insert(table)
            .values(**values)
            .on_conflict_do_nothing(index_elements=index_elements)
        )
        result = await session.execute(stmt)
        return result.rowcount == 0


__all__: "list[str]" = ["SqliteDialectStrategy"]
