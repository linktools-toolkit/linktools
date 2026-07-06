#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemySessionStore: DB-backed SessionStore."""

import json
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import SessionMessageRow, SessionRow
from ...errors import SessionError
from ...session.models import MessageRole, SessionMessage, SessionRecord, SessionStatus


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
        id=row.id, parent_id=row.parent_id, status=SessionStatus(row.status), version=row.version,
        created_at=_as_utc(row.created_at), updated_at=_as_utc(row.updated_at), metadata=json.loads(row.metadata_json),
    )


def _row_to_message(row: SessionMessageRow) -> SessionMessage:
    return SessionMessage(
        id=row.id, session_id=row.session_id, sequence=row.sequence, role=MessageRole(row.role),
        content=json.loads(row.content_json), run_id=row.run_id, created_at=_as_utc(row.created_at),
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
            db_session.add(SessionRow(
                id=session.id, parent_id=session.parent_id, status=session.status.value, version=session.version,
                created_at=session.created_at, updated_at=session.updated_at, metadata_json=json.dumps(dict(session.metadata)),
            ))
        await self._execute_in_session(_do)
        return session

    async def get(self, session_id: str) -> "SessionRecord | None":
        async def _do(db_session):
            result = await db_session.execute(select(SessionRow).where(SessionRow.id == session_id))
            row = result.scalar_one_or_none()
            return None if row is None else _row_to_record(row)
        return await self._execute_in_session(_do)

    async def append_messages(self, session_id: str, messages: "tuple[SessionMessage, ...]") -> None:
        async def _do(db_session):
            for message in messages:
                db_session.add(SessionMessageRow(
                    id=message.id, session_id=message.session_id, sequence=message.sequence, role=message.role.value,
                    content_json=json.dumps(message.content), run_id=message.run_id, created_at=message.created_at,
                    metadata_json=json.dumps(dict(message.metadata)),
                ))
        await self._execute_in_session(_do)

    async def list_messages(self, session_id: str, *, after_sequence: int = 0, limit: int = 1000) -> "tuple[SessionMessage, ...]":
        async def _do(db_session):
            result = await db_session.execute(
                select(SessionMessageRow)
                .where(SessionMessageRow.session_id == session_id, SessionMessageRow.sequence > after_sequence)
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
            result = await db_session.execute(select(SessionRow).where(SessionRow.id == session_id))
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
