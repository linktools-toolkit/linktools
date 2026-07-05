#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyCheckpointStore: DB-backed CheckpointStore, keyed by (run_id, sequence)."""

import json
from datetime import datetime
from typing import Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import RunCheckpointRow
from ...run.models import RunCheckpoint


def _row_to_checkpoint(row: RunCheckpointRow) -> RunCheckpoint:
    return RunCheckpoint(
        id=row.id, run_id=row.run_id, sequence=row.sequence, format=row.format,
        schema_version=row.schema_version, payload=row.payload, created_at=row.created_at,
        metadata=json.loads(row.metadata_json),
    )


class SqlAlchemyCheckpointStore:
    def __init__(self, *, session_factory: "Callable[[], AsyncSession]") -> None:
        self._session_factory = session_factory

    async def save(self, checkpoint: RunCheckpoint) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                session.add(RunCheckpointRow(
                    id=checkpoint.id, run_id=checkpoint.run_id, sequence=checkpoint.sequence,
                    format=checkpoint.format, schema_version=checkpoint.schema_version, payload=checkpoint.payload,
                    created_at=checkpoint.created_at, metadata_json=json.dumps(dict(checkpoint.metadata)),
                ))

    async def latest(self, run_id: str) -> "RunCheckpoint | None":
        async with self._session_factory() as session:
            result = await session.execute(
                select(RunCheckpointRow).where(RunCheckpointRow.run_id == run_id).order_by(RunCheckpointRow.sequence.desc()).limit(1)
            )
            row = result.scalar_one_or_none()
            return None if row is None else _row_to_checkpoint(row)

    async def get(self, checkpoint_id: str) -> "RunCheckpoint | None":
        async with self._session_factory() as session:
            result = await session.execute(select(RunCheckpointRow).where(RunCheckpointRow.id == checkpoint_id))
            row = result.scalar_one_or_none()
            return None if row is None else _row_to_checkpoint(row)
