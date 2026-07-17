#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyEvalStore: DB-backed EvalStore.

Shares one ``session_factory`` with the other SQL stores (and supports a bound
session for the UnitOfWork). Mirrors the task store's session handling and
datetime normalization (aiosqlite round-trips datetimes as naive UTC)."""

import json
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...evaluation.models import EvalResult, EvalRun, EvalRunStatus, EvalTarget
from ...evaluation.store import (
    EvalResultConflictError,
    EvalRunNotFoundError,
)
from ...json import from_jsonable, to_jsonable
from .models import EvalResultRow, EvalRunRow


def _as_utc(value: "datetime | None") -> "datetime | None":
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _store_dt(value: "datetime | None") -> "datetime | None":
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _run_envelope(run: EvalRun) -> str:
    return json.dumps(
        {
            "target": to_jsonable(run.target),
            "baseline_target": to_jsonable(run.baseline_target),
            "metadata": dict(run.metadata),
        }
    )


def _row_to_run(row: EvalRunRow) -> EvalRun:
    env = json.loads(row.data_json)
    baseline = env.get("baseline_target")
    return EvalRun(
        id=row.id,
        suite_id=row.suite_id,
        target=from_jsonable(EvalTarget, env["target"]),
        status=EvalRunStatus(row.status),
        baseline_target=from_jsonable(EvalTarget, baseline) if baseline else None,
        created_at=_as_utc(row.created_at),
        started_at=_as_utc(row.started_at),
        finished_at=_as_utc(row.finished_at),
        metadata=env.get("metadata", {}),
    )


def _result_envelope(result: EvalResult) -> str:
    return json.dumps({"scores": dict(result.scores), "metrics": dict(result.metrics)})


def _row_to_result(row: EvalResultRow) -> EvalResult:
    env = json.loads(row.data_json)
    return EvalResult(
        id=row.id,
        eval_run_id=row.eval_run_id,
        case_id=row.case_id,
        run_id=row.run_id,
        job_id=row.job_id,
        task_id=row.task_id,
        output_artifact_id=row.output_artifact_id,
        snapshot_artifact_id=row.snapshot_artifact_id,
        scores=env.get("scores", {}),
        metrics=env.get("metrics", {}),
        error_type=row.error_type,
        error_message=row.error_message,
    )


class SqlAlchemyEvalStore:
    def __init__(
        self,
        *,
        session_factory: "Callable[[], AsyncSession]",
        session: "AsyncSession | None" = None,
    ) -> None:
        self._session_factory = session_factory
        self._session = session

    async def _in_session(self, action):
        if self._session is not None:
            result = await action(self._session)
            await self._session.flush()
            return result
        async with self._session_factory() as session:
            try:
                result = await action(session)
                await session.commit()
                return result
            except Exception:
                await session.rollback()
                raise

    async def create_run(self, run: EvalRun) -> EvalRun:
        async def do(session: AsyncSession) -> EvalRun:
            existing = await session.get(EvalRunRow, run.id)
            if existing is not None:
                raise EvalResultConflictError(f"eval run already exists: {run.id}")
            session.add(
                EvalRunRow(
                    id=run.id,
                    suite_id=run.suite_id,
                    status=run.status.value,
                    created_at=_store_dt(run.created_at),
                    started_at=_store_dt(run.started_at),
                    finished_at=_store_dt(run.finished_at),
                    data_json=_run_envelope(run),
                )
            )
            return run

        return await self._in_session(do)

    async def get_run(self, run_id: str) -> "EvalRun | None":
        async def do(session: AsyncSession):
            row = await session.get(EvalRunRow, run_id)
            return _row_to_run(row) if row is not None else None

        return await self._in_session(do)

    async def transition_run(
        self,
        run_id: str,
        *,
        status: EvalRunStatus,
        started_at: "datetime | None" = None,
        finished_at: "datetime | None" = None,
    ) -> EvalRun:
        async def do(session: AsyncSession) -> EvalRun:
            row = await session.get(EvalRunRow, run_id)
            if row is None:
                raise EvalRunNotFoundError(f"eval run not found: {run_id}")
            row.status = status.value
            if started_at is not None:
                row.started_at = _store_dt(started_at)
            if finished_at is not None:
                row.finished_at = _store_dt(finished_at)
            return _row_to_run(row)

        return await self._in_session(do)

    async def append_result(self, result: EvalResult) -> EvalResult:
        async def do(session: AsyncSession) -> EvalResult:
            existing = await session.get(EvalResultRow, result.id)
            if existing is not None:
                raise EvalResultConflictError(
                    f"eval result already exists: {result.id}"
                )
            session.add(
                EvalResultRow(
                    id=result.id,
                    eval_run_id=result.eval_run_id,
                    case_id=result.case_id,
                    run_id=result.run_id,
                    job_id=result.job_id,
                    task_id=result.task_id,
                    output_artifact_id=result.output_artifact_id,
                    snapshot_artifact_id=result.snapshot_artifact_id,
                    error_type=result.error_type,
                    error_message=result.error_message,
                    data_json=_result_envelope(result),
                )
            )
            return result

        return await self._in_session(do)

    async def list_results(self, run_id: str) -> "tuple[EvalResult, ...]":
        async def do(session: AsyncSession):
            rows = (
                (
                    await session.execute(
                        select(EvalResultRow)
                        .where(EvalResultRow.eval_run_id == run_id)
                        .order_by(EvalResultRow.id)
                    )
                )
                .scalars()
                .all()
            )
            return tuple(_row_to_result(r) for r in rows)

        return await self._in_session(do)

    async def get_result(self, result_id: str) -> "EvalResult | None":
        async def do(session: AsyncSession):
            row = await session.get(EvalResultRow, result_id)
            return _row_to_result(row) if row is not None else None

        return await self._in_session(do)


__all__: "list[str]" = ["SqlAlchemyEvalStore"]
