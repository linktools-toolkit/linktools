#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PostgreSQL dialect strategy: classifies unique-violation IntegrityErrors by
the constraint name the driver attaches. psycopg exposes it as
``error.orig.diag.constraint_name``; asyncpg's wording puts the constraint name
in the message text, so the named constraint is matched either way. Only the
two asset-domain named constraints are recognized."""

from ..models import ASSET_IDEMPOTENCY_CONSTRAINT, ASSET_PATH_CONSTRAINT
from .base import IntegrityViolationKind


class PostgreSqlDialectStrategy:
    name = "postgresql"

    def classify_integrity_error(self, error: BaseException) -> IntegrityViolationKind:
        orig = getattr(error, "orig", None)
        diag = getattr(orig, "diag", None)
        constraint = getattr(diag, "constraint_name", None) if diag else None
        message = str(orig or error)
        if constraint == ASSET_PATH_CONSTRAINT or ASSET_PATH_CONSTRAINT in message:
            return IntegrityViolationKind.ASSET_KEY
        if (
            constraint == ASSET_IDEMPOTENCY_CONSTRAINT
            or ASSET_IDEMPOTENCY_CONSTRAINT in message
        ):
            return IntegrityViolationKind.IDEMPOTENCY_KEY
        return IntegrityViolationKind.OTHER

    async def execute_conflict_insert(
        self, session, table, values, *, index_elements
    ) -> bool:
        # PostgreSQL supports ON CONFLICT DO NOTHING, which signals a conflict
        # via rowcount without raising -- safe under a surrounding transaction.
        from sqlalchemy.dialects.postgresql import insert

        stmt = (
            insert(table)
            .values(**values)
            .on_conflict_do_nothing(index_elements=index_elements)
        )
        result = await session.execute(stmt)
        return result.rowcount == 0


__all__: "list[str]" = ["PostgreSqlDialectStrategy"]
