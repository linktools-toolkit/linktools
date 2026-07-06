#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyRunStore: DB-backed RunStore. transition() runs read-check-mutate-
commit in one transaction, with the WHERE version = :expected_version clause
providing the optimistic-concurrency check atomically."""

import json
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import RunRow
from ...errors import InvalidRunTransitionError, RunConflictError, RunNotFoundError
from ...run.models import ALLOWED_RUN_TRANSITIONS, RunErrorInfo, RunInput, RunnableType, RunRecord, RunResult, RunStatus


def _as_utc(dt: "datetime | None") -> "datetime | None":
    # SQLite (aiosqlite) round-trips datetimes as naive values regardless of the
    # tzinfo they were stored with, so reattach UTC on read to match the
    # timezone-aware datetimes RunRecord is constructed with everywhere else.
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=timezone.utc)


def _row_to_record(row: RunRow) -> RunRecord:
    return RunRecord(
        id=row.id, root_run_id=row.root_run_id, parent_run_id=row.parent_run_id,
        session_id=row.session_id, runnable_id=row.runnable_id, runnable_type=RunnableType(row.runnable_type),
        status=RunStatus(row.status), input=RunInput(**json.loads(row.input_json)),
        result=None if row.result_json is None else RunResult(**json.loads(row.result_json)),
        error=None if row.error_json is None else RunErrorInfo(**json.loads(row.error_json)),
        version=row.version, created_at=_as_utc(row.created_at), started_at=_as_utc(row.started_at),
        finished_at=_as_utc(row.finished_at), metadata=json.loads(row.metadata_json),
    )


class SqlAlchemyRunStore:
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
        shared session (UoW mode). In UoW mode the shared transaction is already
        open, so we only flush after fn to make the writes visible to subsequent
        reads inside the same unit without committing."""
        if self._session is not None:
            result = await fn(self._session)
            await self._session.flush()
            return result
        async with self._session_factory() as session:
            async with session.begin():
                return await fn(session)

    async def create(self, run: RunRecord) -> RunRecord:
        async def _do(session):
            session.add(RunRow(
                id=run.id, root_run_id=run.root_run_id, parent_run_id=run.parent_run_id,
                session_id=run.session_id, runnable_id=run.runnable_id, runnable_type=run.runnable_type.value,
                status=run.status.value, input_json=json.dumps({"prompt": run.input.prompt, "metadata": dict(run.input.metadata)}),
                result_json=None, error_json=None, version=run.version, created_at=run.created_at,
                started_at=run.started_at, finished_at=run.finished_at, metadata_json=json.dumps(dict(run.metadata)),
            ))
        await self._execute_in_session(_do)
        return run

    async def get(self, run_id: str) -> "RunRecord | None":
        async def _do(session):
            result = await session.execute(select(RunRow).where(RunRow.id == run_id))
            row = result.scalar_one_or_none()
            return None if row is None else _row_to_record(row)
        return await self._execute_in_session(_do)

    async def transition(
        self,
        run_id: str,
        target: RunStatus,
        *,
        expected_version: int,
        result: "RunResult | None" = None,
        error: "RunErrorInfo | None" = None,
    ) -> RunRecord:
        async def _do(session):
            query_result = await session.execute(select(RunRow).where(RunRow.id == run_id))
            row = query_result.scalar_one_or_none()
            if row is None:
                raise RunNotFoundError(f"run not found: {run_id}")
            if row.version != expected_version:
                raise RunConflictError(f"expected version {expected_version}, found {row.version}")
            current_status = RunStatus(row.status)
            if target not in ALLOWED_RUN_TRANSITIONS.get(current_status, frozenset()):
                raise InvalidRunTransitionError(f"cannot transition {current_status} -> {target}")
            now = datetime.now(timezone.utc)
            if target == RunStatus.RUNNING and row.started_at is None:
                row.started_at = now
            if target in (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED):
                row.finished_at = now
            row.status = target.value
            row.version = row.version + 1
            if result is not None:
                row.result_json = json.dumps({"output": result.output, "token_usage": dict(result.token_usage), "metadata": dict(result.metadata)})
            if error is not None:
                row.error_json = json.dumps({"error_type": error.error_type, "message": error.message, "detail": dict(error.detail)})
            await session.flush()
            return _row_to_record(row)
        return await self._execute_in_session(_do)

    async def list_children(self, run_id: str) -> "tuple[RunRecord, ...]":
        async def _do(session):
            result = await session.execute(select(RunRow).where(RunRow.parent_run_id == run_id))
            return tuple(_row_to_record(row) for row in result.scalars())
        return await self._execute_in_session(_do)
