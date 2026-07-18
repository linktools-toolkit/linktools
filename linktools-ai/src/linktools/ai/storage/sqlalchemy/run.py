#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyRunStore: DB-backed RunStore. transition() runs read-check-mutate-
commit in one transaction, with the WHERE version = :expected_version clause
providing the optimistic-concurrency check atomically."""

import json
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import false, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import RunRow
from ...errors import InvalidRunTransitionError, RunConflictError, RunNotFoundError
from ...run.models import (
    ALLOWED_RUN_TRANSITIONS,
    RunErrorInfo,
    RunInput,
    RunnableType,
    RunRecord,
    RunResult,
    RunStatus,
)


def _as_utc(dt: "datetime | None") -> "datetime | None":
    # SQLite (aiosqlite) round-trips datetimes as naive values regardless of the
    # tzinfo they were stored with, so reattach UTC on read to match the
    # timezone-aware datetimes RunRecord is constructed with everywhere else.
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=timezone.utc)


def _row_to_record(row: RunRow) -> RunRecord:
    return RunRecord(
        id=row.id,
        root_run_id=row.root_run_id,
        parent_run_id=row.parent_run_id,
        session_id=row.session_id,
        runnable_id=row.runnable_id,
        runnable_type=RunnableType(row.runnable_type),
        status=RunStatus(row.status),
        input=RunInput(**json.loads(row.input_json)),
        result=None
        if row.result_json is None
        else RunResult(**json.loads(row.result_json)),
        error=None
        if row.error_json is None
        else RunErrorInfo(**json.loads(row.error_json)),
        version=row.version,
        created_at=_as_utc(row.created_at),
        started_at=_as_utc(row.started_at),
        finished_at=_as_utc(row.finished_at),
        metadata=json.loads(row.metadata_json),
        cancel_requested_at=_as_utc(row.cancel_requested_at),
        cancel_requested_by=row.cancel_requested_by,
        cancel_reason=row.cancel_reason,
        worker_id=row.worker_id,
        execution_token=row.execution_token,
        heartbeat_at=_as_utc(row.heartbeat_at),
        manifest_id=row.manifest_id,
        resumability=row.resumability,
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
            session.add(
                RunRow(
                    id=run.id,
                    root_run_id=run.root_run_id,
                    parent_run_id=run.parent_run_id,
                    session_id=run.session_id,
                    runnable_id=run.runnable_id,
                    runnable_type=run.runnable_type.value,
                    status=run.status.value,
                    input_json=json.dumps(
                        {
                            "prompt": run.input.prompt,
                            "metadata": dict(run.input.metadata),
                        }
                    ),
                    result_json=None,
                    error_json=None,
                    version=run.version,
                    created_at=run.created_at,
                    started_at=run.started_at,
                    finished_at=run.finished_at,
                    metadata_json=json.dumps(dict(run.metadata)),
                    cancel_requested_at=run.cancel_requested_at,
                    cancel_requested_by=run.cancel_requested_by,
                    cancel_reason=run.cancel_reason,
                    worker_id=run.worker_id,
                    execution_token=run.execution_token,
                    heartbeat_at=run.heartbeat_at,
                    manifest_id=run.manifest_id,
                    resumability=run.resumability,
                )
            )

        await self._execute_in_session(_do)
        return run

    async def get(self, run_id: str) -> "RunRecord | None":
        async def _do(session):
            result = await session.execute(select(RunRow).where(RunRow.id == run_id))
            row = result.scalar_one_or_none()
            return None if row is None else _row_to_record(row)

        return await self._execute_in_session(_do)

    async def claim_execution(self, run_id: str, *, worker_id: str, execution_token: str) -> RunRecord:
        now = datetime.now(timezone.utc)
        async def _do(session):
            stmt = update(RunRow).where(RunRow.id == run_id).where(
                (RunRow.execution_token.is_(None)) | (RunRow.execution_token == execution_token)
            ).values(worker_id=worker_id, execution_token=execution_token, heartbeat_at=now)
            result = await session.execute(stmt)
            if result.rowcount == 0:
                raise RunConflictError("run execution is already fenced by another worker")
        await self._execute_in_session(_do)
        record = await self.get(run_id)
        if record is None:
            raise RunNotFoundError(f"run not found: {run_id}")
        return record

    async def heartbeat_execution(self, run_id: str, *, worker_id: str, execution_token: str) -> RunRecord:
        now = datetime.now(timezone.utc)
        async def _do(session):
            result = await session.execute(update(RunRow).where(
                RunRow.id == run_id, RunRow.worker_id == worker_id,
                RunRow.execution_token == execution_token,
            ).values(heartbeat_at=now))
            if result.rowcount == 0:
                raise RunConflictError("run execution heartbeat fencing failed")
        await self._execute_in_session(_do)
        record = await self.get(run_id)
        if record is None:
            raise RunNotFoundError(f"run not found: {run_id}")
        return record

    async def transition(
        self,
        run_id: str,
        target: RunStatus,
        *,
        expected_version: int,
        result: "RunResult | None" = None,
        error: "RunErrorInfo | None" = None,
        cancel_requested_at: "datetime | None" = None,
        cancel_requested_by: "str | None" = None,
        cancel_reason: "str | None" = None,
    ) -> RunRecord:
        # a Python read-then-compare-then-flush is forbidden for core state
        # updates -- two concurrent transactions can both SELECT the same
        # version under READ COMMITTED and both then unconditionally write,
        # silently losing one update. The WHERE id=... AND version=:expected
        # clause on the UPDATE itself is the atomic race-decider: at most one
        # concurrent transition can ever match it (rowcount == 1).
        #
        # Valid source statuses are derived from ALLOWED_RUN_TRANSITIONS so the
        # UPDATE's WHERE also enforces the transition-legality check
        # atomically, not just the version.
        valid_sources = tuple(
            source.value
            for source, targets in ALLOWED_RUN_TRANSITIONS.items()
            if target in targets
        )
        now = datetime.now(timezone.utc)
        values: "dict" = {"status": target.value, "version": RunRow.version + 1}
        if target == RunStatus.RUNNING:
            # started_at is set only the first time RUNNING is reached; a
            # resume (WAITING_APPROVAL -> RUNNING) must not clobber it.
            values["started_at"] = func.coalesce(RunRow.started_at, now)
        if target in (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED):
            values["finished_at"] = now
        if result is not None:
            values["result_json"] = json.dumps(
                {
                    "output": result.output,
                    "token_usage": dict(result.token_usage),
                    "metadata": dict(result.metadata),
                }
            )
        if error is not None:
            values["error_json"] = json.dumps(
                {
                    "error_type": error.error_type,
                    "message": error.message,
                    "detail": dict(error.detail),
                }
            )
        # Cancel-request audit: applied when the transition carries a cancel
        # request (set on entering CANCELLING/CANCELLED via a Principal). Once
        # set, later transitions pass None and the column is not touched,
        # preserving the audit trail across the CANCELLING -> CANCELLED handoff.
        if cancel_requested_at is not None:
            values["cancel_requested_at"] = cancel_requested_at
        if cancel_requested_by is not None:
            values["cancel_requested_by"] = cancel_requested_by
        if cancel_reason is not None:
            values["cancel_reason"] = cancel_reason

        async def _do(session):
            stmt = (
                update(RunRow)
                .where(RunRow.id == run_id)
                .where(RunRow.version == expected_version)
                .where(RunRow.status.in_(valid_sources) if valid_sources else false())
                .values(**values)
            )
            result_proxy = await session.execute(stmt)
            if result_proxy.rowcount == 0:
                # WHERE didn't match: discriminate missing / stale-version /
                # illegal-transition so the caller sees the right error class.
                query_result = await session.execute(
                    select(RunRow).where(RunRow.id == run_id)
                )
                row = query_result.scalar_one_or_none()
                if row is None:
                    raise RunNotFoundError(f"run not found: {run_id}")
                if row.version != expected_version:
                    raise RunConflictError(
                        f"expected version {expected_version}, found {row.version}"
                    )
                current_status = RunStatus(row.status)
                raise InvalidRunTransitionError(
                    f"cannot transition {current_status} -> {target}"
                )
            query_result = await session.execute(
                select(RunRow).where(RunRow.id == run_id)
            )
            row = query_result.scalar_one()
            return _row_to_record(row)

        return await self._execute_in_session(_do)

    async def list_children(self, run_id: str) -> "tuple[RunRecord, ...]":
        async def _do(session):
            result = await session.execute(
                select(RunRow).where(RunRow.parent_run_id == run_id)
            )
            return tuple(_row_to_record(row) for row in result.scalars())

        return await self._execute_in_session(_do)
