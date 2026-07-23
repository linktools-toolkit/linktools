#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MySQL dialect strategy: classifies only duplicate-key violations (driver
error 1062) by parsing the named constraint out of the message. Every other
driver error is OTHER -- INSERT IGNORE is never used because it would swallow
non-unique-constraint failures."""

from ..models import ASSET_IDEMPOTENCY_CONSTRAINT, ASSET_PATH_CONSTRAINT
from .base import IntegrityViolationKind

# MySQL ER_DUP_ENTRY: a unique/primary key collision.
_MYSQL_DUPLICATE_KEY_CODE = 1062


def _driver_error_code(orig: BaseException) -> "int | None":
    # aiomysql / pymysql errors carry the integer code as the first arg.
    args = getattr(orig, "args", None)
    code = args[0] if args else None
    return code if isinstance(code, int) else None


class MySqlDialectStrategy:
    name = "mysql"

    def classify_integrity_error(self, error: BaseException) -> IntegrityViolationKind:
        orig = getattr(error, "orig", None)
        if _driver_error_code(orig) != _MYSQL_DUPLICATE_KEY_CODE:
            return IntegrityViolationKind.OTHER
        message = str(orig or error)
        if ASSET_PATH_CONSTRAINT in message:
            return IntegrityViolationKind.ASSET_KEY
        if ASSET_IDEMPOTENCY_CONSTRAINT in message:
            return IntegrityViolationKind.IDEMPOTENCY_KEY
        return IntegrityViolationKind.OTHER

    async def execute_conflict_insert(
        self, session, table, values, *, index_elements
    ) -> bool:
        # MySQL has no ON CONFLICT clause and INSERT IGNORE is forbidden (it
        # swallows non-unique failures). Use a plain INSERT inside a SAVEPOINT:
        # a unique-key collision rolls back to the savepoint (the outer
        # transaction stays usable) and classification decides whether it is a
        # known conflict (return True) or an unrelated violation (re-raise).
        # MySQL SAVEPOINTs are transactionally sound (unlike aiosqlite's).
        from sqlalchemy import insert
        from sqlalchemy.exc import IntegrityError

        stmt = insert(table).values(**values)
        try:
            async with session.begin_nested():
                await session.execute(stmt)
        except IntegrityError as exc:
            if self.classify_integrity_error(exc) is IntegrityViolationKind.OTHER:
                raise
            return True
        return False


__all__: "list[str]" = ["MySqlDialectStrategy"]
