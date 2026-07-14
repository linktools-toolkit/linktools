#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemySessionStore: DB-backed SessionStore.

Concurrency boundary: ``append_messages``
is the sequence authority -- it reads MAX(sequence) for the session and
assigns fresh ones inside the inserting transaction.

* **Normal mode** (``session=None``, the common case -- each call opens its
  own session + transaction): safe under concurrent appenders. A unique
  ``(session_id, sequence)` collision from two callers racing to reserve the
  same next sequence raises ``IntegrityError``, which is caught and retried
  (up to 8 attempts, mirroring ``SqlAlchemyEventStore.append``) -- the retry
  re-reads MAX(sequence), which now reflects the winner's committed row.

* **Explicit UnitOfWork mode** (``session=<shared session>``, e.g. inside
  ``AgentRunner``'s pause-path transaction): a SINGLE attempt only. A
  sequence-conflict ``IntegrityError`` here cannot be retried (the shared
  transaction would need to roll back and restart from the caller's
  perspective, not just this store's), and -- per the same aiosqlite
  SAVEPOINT limitation documented in ``storage/sqlalchemy/approval.py``
  (a released SAVEPOINT does not reliably participate in a later outer
  rollback) -- is NOT wrapped in ``session.begin_nested()`` either. **Two
  concurrent tasks appending to the SAME session within the SAME explicit
  UnitOfWork are therefore not supported** and will surface as an
  unhandled ``IntegrityError`` propagating out of the transaction. This is a
  deliberate scope boundary, not an oversight: every current caller
  (AgentRunner, SwarmRunner) drives at most one UoW-mode session append per
  transaction, so the scenario does not arise in practice. If a future
  caller needs multiple concurrent UoW-mode writers to the same session, the
  fix is a dedicated ``ai_session_counters(session_id, next_sequence)`` table
  with an atomic ``UPDATE ... RETURNING`` (mirroring how ``claim_task``
  allocates), NOT a broader retry loop."""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from .models import SessionMessageRow, SessionRow
from ...errors import SessionError, SessionSequenceConflictError
from ...session.models import (
    MessageRole,
    NewSessionMessage,
    SessionMessage,
    SessionRecord,
    SessionStatus,
)


def _as_utc(dt: "datetime | None") -> "datetime | None":
    # SQLite (aiosqlite) round-trips datetimes as naive values regardless of the
    # tzinfo they were stored with, so reattach UTC on read to match the
    # timezone-aware datetimes SessionRecord/SessionMessage are constructed with
    # everywhere else.
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=timezone.utc)


def _row_to_record(row: SessionRow) -> SessionRecord:
    return SessionRecord(
        id=row.id,
        parent_id=row.parent_id,
        user_id=row.user_id,
        tenant_id=row.tenant_id,
        status=SessionStatus(row.status),
        version=row.version,
        created_at=_as_utc(row.created_at),
        updated_at=_as_utc(row.updated_at),
        metadata=json.loads(row.metadata_json),
    )


def _row_to_message(row: SessionMessageRow) -> SessionMessage:
    return SessionMessage(
        id=row.id,
        session_id=row.session_id,
        sequence=row.sequence,
        role=MessageRole(row.role),
        content=json.loads(row.content_json),
        run_id=row.run_id,
        created_at=_as_utc(row.created_at),
        metadata=json.loads(row.metadata_json),
    )


class SqlAlchemySessionStore:
    def __init__(
        self,
        *,
        session_factory: "Callable[[], AsyncSession]",
        session: "AsyncSession | None" = None,
    ) -> None:
        self._session_factory = session_factory
        # UoW mode: when set, every method uses this shared session directly and
        # does NOT open its own session or call session.begin() -- the UoW owns
        # the transaction. None means normal mode (own session + transaction).
        self._session = session

    async def _execute_in_session(self, fn):
        """Run ``fn(session)`` in own transaction (normal mode) or against the
        shared session (UoW mode). See SqlAlchemyRunStore._execute_in_session."""
        if self._session is not None:
            result = await fn(self._session)
            await self._session.flush()
            return result
        async with self._session_factory() as session:
            async with session.begin():
                return await fn(session)

    async def create(self, session: SessionRecord) -> SessionRecord:
        async def _do(db_session):
            db_session.add(
                SessionRow(
                    id=session.id,
                    parent_id=session.parent_id,
                    user_id=session.user_id,
                    tenant_id=session.tenant_id,
                    status=session.status.value,
                    version=session.version,
                    created_at=session.created_at,
                    updated_at=session.updated_at,
                    metadata_json=json.dumps(dict(session.metadata)),
                )
            )

        await self._execute_in_session(_do)
        return session

    async def get(self, session_id: str) -> "SessionRecord | None":
        async def _do(db_session):
            result = await db_session.execute(
                select(SessionRow).where(SessionRow.id == session_id)
            )
            row = result.scalar_one_or_none()
            return None if row is None else _row_to_record(row)

        return await self._execute_in_session(_do)

    async def _append_one_batch(
        self,
        session: AsyncSession,
        session_id: str,
        messages: "tuple[NewSessionMessage, ...]",
    ) -> "tuple[SessionMessage, ...]":
        # reserve the next sequence(s) inside the inserting
        # transaction -- read MAX(sequence) for the session, assign
        # contiguously, insert. Mirrors SqlAlchemyEventStore._append_one.
        result = await session.execute(
            select(func.max(SessionMessageRow.sequence)).where(
                SessionMessageRow.session_id == session_id
            )
        )
        next_seq = (result.scalar() or 0) + 1
        persisted = []
        for offset, message in enumerate(messages):
            sequence = next_seq + offset
            now = datetime.now(timezone.utc)
            row_id = str(uuid.uuid4())
            session.add(
                SessionMessageRow(
                    id=row_id,
                    session_id=session_id,
                    sequence=sequence,
                    role=message.role.value,
                    content_json=json.dumps(message.content),
                    run_id=message.run_id,
                    created_at=now,
                    metadata_json=json.dumps(dict(message.metadata)),
                )
            )
            persisted.append(
                SessionMessage(
                    id=row_id,
                    session_id=session_id,
                    sequence=sequence,
                    role=message.role,
                    content=message.content,
                    run_id=message.run_id,
                    created_at=now,
                    metadata=message.metadata,
                )
            )
        await session.flush()
        return tuple(persisted)

    async def append_messages(
        self,
        session_id: str,
        messages: "tuple[NewSessionMessage, ...]",
    ) -> "tuple[SessionMessage, ...]":
        if not messages:
            return ()
        if self._session is not None:
            # UoW mode: single attempt -- a sequence-conflict IntegrityError
            # would poison the shared transaction, and within one unit there
            # is no concurrent appender to race against.
            return await self._append_one_batch(self._session, session_id, messages)
        last_exc: "BaseException | None" = None
        for _ in range(8):
            try:
                async with self._session_factory() as session:
                    async with session.begin():
                        return await self._append_one_batch(
                            session, session_id, messages
                        )
            except IntegrityError as exc:
                # Unique (session_id, sequence) collision -- a concurrent
                # append reserved the same sequence first. Retry to re-read MAX.
                last_exc = exc
                await asyncio.sleep(0)
                continue
            except OperationalError as exc:
                if "database is locked" not in str(exc).lower():
                    raise
                last_exc = exc
                await asyncio.sleep(0.01)
                continue
        raise SessionSequenceConflictError(
            f"could not reserve a unique message sequence for session {session_id!r} "
            f"after repeated conflicts"
        ) from last_exc

    async def list_messages(
        self, session_id: str, *, after_sequence: int = 0, limit: int = 1000
    ) -> "tuple[SessionMessage, ...]":
        async def _do(db_session):
            result = await db_session.execute(
                select(SessionMessageRow)
                .where(
                    SessionMessageRow.session_id == session_id,
                    SessionMessageRow.sequence > after_sequence,
                )
                .order_by(SessionMessageRow.sequence.asc())
                .limit(limit)
            )
            return tuple(_row_to_message(row) for row in result.scalars())

        return await self._execute_in_session(_do)

    async def update(
        self,
        session_id: str,
        *,
        status: "SessionStatus | None" = None,
        metadata: "Mapping[str, Any] | None" = None,
    ) -> SessionRecord:
        async def _do(db_session):
            result = await db_session.execute(
                select(SessionRow).where(SessionRow.id == session_id)
            )
            row = result.scalar_one_or_none()
            if row is None:
                raise SessionError(f"session not found: {session_id}")
            if status is not None:
                row.status = status.value
            if metadata is not None:
                row.metadata_json = json.dumps(dict(metadata))
            row.version = row.version + 1
            row.updated_at = datetime.now(timezone.utc)
            await db_session.flush()
            return _row_to_record(row)

        return await self._execute_in_session(_do)
