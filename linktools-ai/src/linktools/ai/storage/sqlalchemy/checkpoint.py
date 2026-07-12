#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyCheckpointStore: DB-backed CheckpointStore, keyed by (run_id, sequence)."""

import json
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import RunCheckpointRow
from ...run.models import RunCheckpoint


def _as_utc(dt: "datetime | None") -> "datetime | None":
    # SQLite (aiosqlite) round-trips datetimes as naive values regardless of the
    # tzinfo they were stored with, so reattach UTC on read to match the
    # timezone-aware datetimes RunRecord is constructed with everywhere else.
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=timezone.utc)


def _row_to_checkpoint(row: RunCheckpointRow) -> RunCheckpoint:
    return RunCheckpoint(
        id=row.id,
        run_id=row.run_id,
        sequence=row.sequence,
        format=row.format,
        schema_version=row.schema_version,
        payload=row.payload,
        created_at=_as_utc(row.created_at),
        metadata=json.loads(row.metadata_json),
    )


class SqlAlchemyCheckpointStore:
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

    async def save(self, checkpoint: RunCheckpoint) -> None:
        async def _do(session):
            session.add(
                RunCheckpointRow(
                    id=checkpoint.id,
                    run_id=checkpoint.run_id,
                    sequence=checkpoint.sequence,
                    format=checkpoint.format,
                    schema_version=checkpoint.schema_version,
                    payload=checkpoint.payload,
                    created_at=checkpoint.created_at,
                    metadata_json=json.dumps(dict(checkpoint.metadata)),
                )
            )

        await self._execute_in_session(_do)

    async def latest(self, run_id: str) -> "RunCheckpoint | None":
        async def _do(session):
            result = await session.execute(
                select(RunCheckpointRow)
                .where(RunCheckpointRow.run_id == run_id)
                .order_by(RunCheckpointRow.sequence.desc())
                .limit(1)
            )
            row = result.scalar_one_or_none()
            return None if row is None else _row_to_checkpoint(row)

        return await self._execute_in_session(_do)

    async def get(self, checkpoint_id: str) -> "RunCheckpoint | None":
        async def _do(session):
            result = await session.execute(
                select(RunCheckpointRow).where(RunCheckpointRow.id == checkpoint_id)
            )
            row = result.scalar_one_or_none()
            return None if row is None else _row_to_checkpoint(row)

        return await self._execute_in_session(_do)
