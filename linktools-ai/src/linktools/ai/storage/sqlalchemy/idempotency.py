#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyIdempotencyStore: DB-backed IdempotencyStore (Protocol in
tool/idempotency.py). Mirrors SqlAlchemyApprovalStore's structure:
``session_factory: Callable[[], AsyncSession]`` constructor, ``_as_utc``
helper for aiosqlite's naive-datetime round-trip, and the
``_execute_in_session`` UoW hook so the store can participate in cross-store
transactions through SqlAlchemyStorage.transaction().

``reserve`` handles the race via the unique (scope, key) constraint:
INSERT a RESERVED row; on IntegrityError (concurrent insert from another
process) SELECT the winner and hash-check it. This is the multi-process
equivalent of FileIdempotencyStore's asyncio.Lock -- both backends enforce
"at most one RESERVED per (scope, key)" but the SQL backend does it via the
schema rather than an in-process lock."""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .models import ToolIdempotencyRow
from ...errors import IdempotencyConflictError
from ...tool.idempotency import IdempotencyRecord, IdempotencyStatus


def _as_utc(dt: "datetime | None") -> "datetime | None":
    # SQLite (aiosqlite) round-trips datetimes as naive values regardless of
    # tzinfo, so reattach UTC on read to match the timezone-aware datetimes
    # IdempotencyRecord is constructed with everywhere else.
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=timezone.utc)


def _row_to_record(row: ToolIdempotencyRow) -> IdempotencyRecord:
    return IdempotencyRecord(
        id=row.id,
        scope=row.scope,
        key=row.key,
        request_hash=row.request_hash,
        status=IdempotencyStatus(row.status),
        result=None if row.result_json is None else json.loads(row.result_json),
        error=row.error_text,
        created_at=_as_utc(row.created_at),
        completed_at=_as_utc(row.completed_at),
    )


class SqlAlchemyIdempotencyStore:
    """Multi-process IdempotencyStore backed by SQLAlchemy/AsyncSession.

    Mirrors SqlAlchemyApprovalStore: ``session_factory`` constructor,
    optional shared ``session`` for UoW mode (every method reuses it instead
    of opening its own transaction). The unique (scope, key) constraint
    backs the reserve() race: a duplicate insert raises IntegrityError,
    which we translate into a SELECT of the existing row + hash check."""

    def __init__(
        self,
        *,
        session_factory: "Callable[[], AsyncSession]",
        session: "AsyncSession | None" = None,
    ) -> None:
        self._session_factory = session_factory
        # UoW mode: when set, every method uses this shared session directly
        # and does NOT open its own session or call session.begin() -- the
        # UoW owns the transaction. None means normal mode.
        self._session = session

    async def _execute_in_session(self, fn):
        """Run ``fn(session)`` in own transaction (normal mode) or against
        the shared session (UoW mode). See SqlAlchemyRunStore."""
        if self._session is not None:
            result = await fn(self._session)
            await self._session.flush()
            return result
        async with self._session_factory() as session:
            async with session.begin():
                return await fn(session)

    # -- read ----------------------------------------------------------

    async def get(self, scope: str, key: str) -> "IdempotencyRecord | None":
        async def _do(session):
            result = await session.execute(
                select(ToolIdempotencyRow).where(
                    ToolIdempotencyRow.scope == scope,
                    ToolIdempotencyRow.key == key,
                )
            )
            row = result.scalar_one_or_none()
            return None if row is None else _row_to_record(row)
        return await self._execute_in_session(_do)

    # -- write ---------------------------------------------------------

    async def reserve(
        self,
        scope: str,
        key: str,
        request_hash: str,
        *,
        expires_at: "datetime | None" = None,
    ) -> "IdempotencyRecord | None":
        """INSERT a fresh RESERVED row; on the (scope, key) unique-constraint
        collision, SELECT the existing row. Hash-mismatch on the existing row
        raises IdempotencyConflictError; hash-match returns the existing row
        so the caller can branch on status. Returns None only when the
        INSERT succeeds (truly fresh reservation)."""
        now = datetime.now(timezone.utc)
        record_id = str(uuid.uuid4())

        def _row():
            return ToolIdempotencyRow(
                id=record_id,
                scope=scope,
                key=key,
                request_hash=request_hash,
                status=IdempotencyStatus.RESERVED.value,
                result_json=None,
                error_text=None,
                created_at=now,
                completed_at=None,
                expires_at=expires_at,
            )

        try:
            if self._session is not None:
                # §7.6/P0-7: in UoW mode, isolate the INSERT in a SAVEPOINT
                # (session.begin_nested()) so a unique-constraint collision
                # only rolls back this savepoint -- NOT the whole shared
                # transaction. Without this, IntegrityError on flush() marks
                # the outer AsyncSession's transaction as needing rollback,
                # poisoning every other write the surrounding UnitOfWork made
                # (tool idempotency would silently break approval/run/event
                # writes sharing the same transaction).
                async with self._session.begin_nested():
                    self._session.add(_row())
                    await self._session.flush()
            else:
                async def _insert(session):
                    session.add(_row())
                await self._execute_in_session(_insert)
            return None
        except IntegrityError as exc:
            # Concurrent insert won the race (or this is a true duplicate).
            # The savepoint (UoW mode) or the failed transaction (normal mode)
            # already rolled back just this insert -- the surrounding
            # transaction (if any) is still usable. SELECT the existing row to
            # surface the domain-correct answer (existing record or
            # hash-conflict) so the caller sees the same outcome either
            # backend would produce.
            existing = await self.get(scope, key)
            if existing is None:
                # Should be impossible (the IntegrityError proves the row
                # exists), but don't mask the original error if it happens.
                raise
            if existing.request_hash != request_hash:
                raise IdempotencyConflictError(
                    f"idempotency key {key!r} reused with a different request"
                ) from exc
            return existing

    async def complete(self, scope: str, key: str, result: Any) -> None:
        async def _do(session):
            result_q = await session.execute(
                select(ToolIdempotencyRow).where(
                    ToolIdempotencyRow.scope == scope,
                    ToolIdempotencyRow.key == key,
                )
            )
            row = result_q.scalar_one_or_none()
            if row is None:
                return
            row.status = IdempotencyStatus.COMPLETED.value
            row.result_json = json.dumps(result, default=str)
            row.error_text = None
            row.completed_at = datetime.now(timezone.utc)
        await self._execute_in_session(_do)

    async def fail(self, scope: str, key: str, error: str) -> None:
        async def _do(session):
            result_q = await session.execute(
                select(ToolIdempotencyRow).where(
                    ToolIdempotencyRow.scope == scope,
                    ToolIdempotencyRow.key == key,
                )
            )
            row = result_q.scalar_one_or_none()
            if row is None:
                return
            row.status = IdempotencyStatus.FAILED.value
            row.result_json = None
            row.error_text = error
            row.completed_at = datetime.now(timezone.utc)
        await self._execute_in_session(_do)
