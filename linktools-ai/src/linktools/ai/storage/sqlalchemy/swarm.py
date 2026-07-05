#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemySwarmStore: DB-backed SwarmStore (the Protocol in
swarm/store.py). Mirrors SqlAlchemyRunStore's structure:
`session_factory: Callable[[], AsyncSession]` constructor, `_as_utc` helper for
aiosqlite's naive-datetime round-trip, and read-check-mutate-commit transactions.

The SQL-specific behaviour lives in ``claim_task`` and ``reclaim_expired_tasks``:

* ``claim_task`` issues a real ``UPDATE ... SET status='claimed' WHERE id=:tid
  AND status='pending'`` and checks ``rowcount`` — the WHERE clause is the atomic
  optimistic claim that makes the loser of a concurrent race observe 0 rows.
* ``reclaim_expired_tasks`` issues a real ``UPDATE ... SET status='pending' WHERE
  status='claimed' AND lease_expires_at < :now``, which the single-process
  FileSwarmStore cannot do (it returns ``()`` unconditionally).
"""

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Callable

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import SwarmRunRow, SwarmTaskRow
from ...errors import (
    InvalidSwarmTransitionError,
    SwarmConflictError,
    SwarmRunNotFoundError,
    SwarmTaskNotFoundError,
)
from ...run.models import RunErrorInfo, RunResult
from ...swarm.models import (
    ALLOWED_SWARM_TRANSITIONS,
    SwarmRun,
    SwarmStatus,
    SwarmTask,
    SwarmTaskStatus,
    TaskInput,
    TokenUsage,
)


def _as_utc(dt: "datetime | None") -> "datetime | None":
    # SQLite (aiosqlite) round-trips datetimes as naive values regardless of the
    # tzinfo they were stored with, so reattach UTC on read to match the
    # timezone-aware datetimes SwarmRun/SwarmTask are constructed with everywhere
    # else.
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=timezone.utc)


def _row_to_run(row: SwarmRunRow) -> SwarmRun:
    cost = Decimal(row.total_cost)
    return SwarmRun(
        id=row.id,
        run_id=row.run_id,
        round=row.round,
        status=SwarmStatus(row.status),
        version=row.version,
        token_usage=TokenUsage(
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            total_cost=cost,
        ),
        cost=cost,
        created_at=_as_utc(row.created_at),
        updated_at=_as_utc(row.updated_at),
        metadata=json.loads(row.metadata_json),
    )


def _row_to_task(row: SwarmTaskRow) -> SwarmTask:
    return SwarmTask(
        id=row.id,
        swarm_run_id=row.swarm_run_id,
        parent_task_id=row.parent_task_id,
        assigned_agent_id=row.assigned_agent_id,
        description=row.description,
        status=SwarmTaskStatus(row.status),
        dependencies=tuple(json.loads(row.dependencies_json)),
        input=TaskInput(**json.loads(row.input_json)),
        result=None if row.result_json is None else RunResult(**json.loads(row.result_json)),
        error=None if row.error_json is None else RunErrorInfo(**json.loads(row.error_json)),
        attempts=row.attempts,
        version=row.version,
        claimed_at=_as_utc(row.claimed_at),
        lease_expires_at=_as_utc(row.lease_expires_at),
        created_at=_as_utc(row.created_at),
        updated_at=_as_utc(row.updated_at),
    )


def _result_to_json(result: RunResult) -> str:
    return json.dumps({
        "output": result.output,
        "token_usage": dict(result.token_usage),
        "metadata": dict(result.metadata),
    })


def _error_to_json(error: RunErrorInfo) -> str:
    return json.dumps({
        "error_type": error.error_type,
        "message": error.message,
        "detail": dict(error.detail),
    })


class SqlAlchemySwarmStore:
    """Multi-process SwarmStore backed by SQLAlchemy/AsyncSession.

    Optimistic concurrency on ``update_run`` mirrors ``SqlAlchemyRunStore.transition``
    (read-check-mutate-commit in one transaction). ``claim_task`` goes further:
    the actual claim is a SQL ``UPDATE ... WHERE status='pending'`` whose
    ``rowcount`` is the atomic race-decider, so two concurrent workers cannot
    claim the same task.
    """

    def __init__(self, *, session_factory: "Callable[[], AsyncSession]") -> None:
        self._session_factory = session_factory

    # -- run lifecycle -------------------------------------------------

    async def create_run(self, run: SwarmRun) -> SwarmRun:
        async with self._session_factory() as session:
            async with session.begin():
                session.add(SwarmRunRow(
                    id=run.id,
                    run_id=run.run_id,
                    round=run.round,
                    status=run.status.value,
                    version=run.version,
                    input_tokens=run.token_usage.input_tokens,
                    output_tokens=run.token_usage.output_tokens,
                    total_cost=str(run.cost),
                    created_at=run.created_at,
                    updated_at=run.updated_at,
                    metadata_json=json.dumps(dict(run.metadata)),
                ))
        return run

    async def get_run(self, swarm_run_id: str) -> "SwarmRun | None":
        async with self._session_factory() as session:
            result = await session.execute(
                select(SwarmRunRow).where(SwarmRunRow.id == swarm_run_id)
            )
            row = result.scalar_one_or_none()
            return None if row is None else _row_to_run(row)

    async def update_run(
        self,
        swarm_run_id: str,
        *,
        expected_version: int,
        status: "SwarmStatus | None" = None,
        round: "int | None" = None,
        token_usage: "TokenUsage | None" = None,
        cost: "Decimal | None" = None,
        metadata: "dict | None" = None,
    ) -> SwarmRun:
        async with self._session_factory() as session:
            async with session.begin():
                query_result = await session.execute(
                    select(SwarmRunRow).where(SwarmRunRow.id == swarm_run_id)
                )
                row = query_result.scalar_one_or_none()
                if row is None:
                    raise SwarmRunNotFoundError(f"swarm run not found: {swarm_run_id}")
                if row.version != expected_version:
                    raise SwarmConflictError(
                        f"expected version {expected_version}, found {row.version}"
                    )
                if status is not None and status.value != row.status:
                    current = SwarmStatus(row.status)
                    if status not in ALLOWED_SWARM_TRANSITIONS.get(current, frozenset()):
                        raise InvalidSwarmTransitionError(
                            f"cannot transition {current} -> {status}"
                        )
                    row.status = status.value
                if round is not None:
                    row.round = round
                if token_usage is not None:
                    row.input_tokens = token_usage.input_tokens
                    row.output_tokens = token_usage.output_tokens
                if cost is not None:
                    row.total_cost = str(cost)
                if metadata is not None:
                    row.metadata_json = json.dumps(metadata)
                row.version = row.version + 1
                row.updated_at = datetime.now(timezone.utc)
                await session.flush()
                return _row_to_run(row)

    # -- task lifecycle ------------------------------------------------

    async def create_task(self, task: SwarmTask) -> SwarmTask:
        async with self._session_factory() as session:
            async with session.begin():
                session.add(SwarmTaskRow(
                    id=task.id,
                    swarm_run_id=task.swarm_run_id,
                    parent_task_id=task.parent_task_id,
                    assigned_agent_id=task.assigned_agent_id,
                    description=task.description,
                    status=task.status.value,
                    dependencies_json=json.dumps(list(task.dependencies)),
                    input_json=json.dumps({
                        "prompt": task.input.prompt,
                        "metadata": dict(task.input.metadata),
                    }),
                    result_json=None if task.result is None else _result_to_json(task.result),
                    error_json=None if task.error is None else _error_to_json(task.error),
                    attempts=task.attempts,
                    version=task.version,
                    claimed_at=task.claimed_at,
                    lease_expires_at=task.lease_expires_at,
                    created_at=task.created_at,
                    updated_at=task.updated_at,
                ))
        return task

    async def list_tasks(
        self, swarm_run_id: str, *, status: "SwarmTaskStatus | None" = None
    ) -> "tuple[SwarmTask, ...]":
        async with self._session_factory() as session:
            query = select(SwarmTaskRow).where(SwarmTaskRow.swarm_run_id == swarm_run_id)
            if status is not None:
                query = query.where(SwarmTaskRow.status == status.value)
            query = query.order_by(SwarmTaskRow.created_at)
            result = await session.execute(query)
            return tuple(_row_to_task(row) for row in result.scalars())

    async def claim_task(
        self, swarm_run_id: str, agent_id: str, *, lease_seconds: "float | None" = None
    ) -> "SwarmTask | None":
        async with self._session_factory() as session:
            async with session.begin():
                # Snapshot pending candidates ordered by created_at. FOR UPDATE
                # SKIP LOCKED is a no-op on SQLite (aiosqlite) but is the correct
                # row-locking clause for Postgres, where concurrent workers each
                # skip rows the others have locked.
                candidates_result = await session.execute(
                    select(SwarmTaskRow)
                    .where(SwarmTaskRow.swarm_run_id == swarm_run_id)
                    .where(SwarmTaskRow.status == SwarmTaskStatus.PENDING.value)
                    .order_by(SwarmTaskRow.created_at)
                    .with_for_update(skip_locked=True)
                )
                candidates = candidates_result.scalars().all()
                for candidate in candidates:
                    # Dependencies gate: every dependency task must be SUCCEEDED.
                    deps = json.loads(candidate.dependencies_json)
                    deps_ok = True
                    for dep_id in deps:
                        dep_result = await session.execute(
                            select(SwarmTaskRow.status).where(SwarmTaskRow.id == dep_id)
                        )
                        dep_status = dep_result.scalar_one_or_none()
                        if dep_status != SwarmTaskStatus.SUCCEEDED.value:
                            deps_ok = False
                            break
                    if not deps_ok:
                        continue
                    now = datetime.now(timezone.utc)
                    lease_expires = (
                        None if lease_seconds is None
                        else now + timedelta(seconds=lease_seconds)
                    )
                    # Atomic optimistic claim: the WHERE status='pending' clause
                    # makes the UPDATE hit 0 rows if another worker raced us to
                    # this task between our SELECT and UPDATE. rowcount is the
                    # race-decider.
                    claim_result = await session.execute(
                        update(SwarmTaskRow)
                        .where(SwarmTaskRow.id == candidate.id)
                        .where(SwarmTaskRow.status == SwarmTaskStatus.PENDING.value)
                        .values(
                            status=SwarmTaskStatus.CLAIMED.value,
                            assigned_agent_id=agent_id,
                            claimed_at=now,
                            lease_expires_at=lease_expires,
                            version=SwarmTaskRow.version + 1,
                            updated_at=now,
                        )
                    )
                    if claim_result.rowcount == 1:
                        # version was set via SQL expression (column + 1); refresh
                        # to repopulate the in-memory row from the DB post-update.
                        await session.refresh(candidate)
                        return _row_to_task(candidate)
                    # rowcount == 0: another worker claimed it; try next candidate.
                return None

    async def complete_task(self, task_id: str, result: RunResult) -> SwarmTask:
        async with self._session_factory() as session:
            async with session.begin():
                query_result = await session.execute(
                    select(SwarmTaskRow).where(SwarmTaskRow.id == task_id)
                )
                row = query_result.scalar_one_or_none()
                if row is None:
                    raise SwarmTaskNotFoundError(f"swarm task not found: {task_id}")
                now = datetime.now(timezone.utc)
                row.status = SwarmTaskStatus.SUCCEEDED.value
                row.result_json = _result_to_json(result)
                row.version = row.version + 1
                row.updated_at = now
                await session.flush()
                return _row_to_task(row)

    async def fail_task(self, task_id: str, error: RunErrorInfo) -> SwarmTask:
        async with self._session_factory() as session:
            async with session.begin():
                query_result = await session.execute(
                    select(SwarmTaskRow).where(SwarmTaskRow.id == task_id)
                )
                row = query_result.scalar_one_or_none()
                if row is None:
                    raise SwarmTaskNotFoundError(f"swarm task not found: {task_id}")
                now = datetime.now(timezone.utc)
                row.status = SwarmTaskStatus.FAILED.value
                row.error_json = _error_to_json(error)
                row.attempts = row.attempts + 1
                row.version = row.version + 1
                row.updated_at = now
                await session.flush()
                return _row_to_task(row)

    async def reclaim_expired_tasks(self, swarm_run_id: str) -> "tuple[SwarmTask, ...]":
        async with self._session_factory() as session:
            async with session.begin():
                now = datetime.now(timezone.utc)
                expired_result = await session.execute(
                    select(SwarmTaskRow)
                    .where(SwarmTaskRow.swarm_run_id == swarm_run_id)
                    .where(SwarmTaskRow.status == SwarmTaskStatus.CLAIMED.value)
                    .where(SwarmTaskRow.lease_expires_at < now)
                )
                expired = expired_result.scalars().all()
                reclaimed: list = []
                for row in expired:
                    row.status = SwarmTaskStatus.PENDING.value
                    row.assigned_agent_id = None
                    row.claimed_at = None
                    row.lease_expires_at = None
                    row.version = row.version + 1
                    row.updated_at = now
                    reclaimed.append(_row_to_task(row))
                await session.flush()
                return tuple(reclaimed)
