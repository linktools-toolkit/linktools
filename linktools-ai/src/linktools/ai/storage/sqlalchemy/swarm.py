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
  FilesystemSwarmStore cannot do (it returns ``()`` unconditionally).
"""

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Callable

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import SwarmRunRow, SwarmTaskAttemptRow, SwarmTaskRow
from ...errors import (
    InvalidSwarmTransitionError,
    SwarmConflictError,
    SwarmRunNotFoundError,
    SwarmTaskNotFoundError,
)
from ...run.models import RunErrorInfo, RunResult
from ...swarm.models import (
    ALLOWED_SWARM_TRANSITIONS,
    AttemptStatus,
    SwarmRun,
    SwarmStatus,
    SwarmTask,
    SwarmTaskAttempt,
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
        result=None
        if row.result_json is None
        else RunResult(**json.loads(row.result_json)),
        error=None
        if row.error_json is None
        else RunErrorInfo(**json.loads(row.error_json)),
        attempts=row.attempts,
        version=row.version,
        claimed_at=_as_utc(row.claimed_at),
        lease_expires_at=_as_utc(row.lease_expires_at),
        created_at=_as_utc(row.created_at),
        updated_at=_as_utc(row.updated_at),
        # getattr covers rows written before the column was added (raw
        # SQL rows from an old DB won't have it; SQLAlchemy unmapped-column
        # access raises AttributeError).
        active_run_id=getattr(row, "active_run_id", None),
    )


def _result_to_json(result: RunResult) -> str:
    return json.dumps(
        {
            "output": result.output,
            "token_usage": dict(result.token_usage),
            "metadata": dict(result.metadata),
        }
    )


def _error_to_json(error: RunErrorInfo) -> str:
    return json.dumps(
        {
            "error_type": error.error_type,
            "message": error.message,
            "detail": dict(error.detail),
        }
    )


def _row_to_attempt(row: SwarmTaskAttemptRow) -> SwarmTaskAttempt:
    return SwarmTaskAttempt(
        id=row.id,
        task_id=row.task_id,
        run_id=row.run_id,
        agent_id=row.agent_id,
        attempt=row.attempt,
        status=AttemptStatus(row.status),
        started_at=_as_utc(row.started_at),
        finished_at=_as_utc(row.finished_at),
        error=None
        if row.error_json is None
        else RunErrorInfo(**json.loads(row.error_json)),
    )


class SqlAlchemySwarmStore:
    """Multi-process SwarmStore backed by SQLAlchemy/AsyncSession.

    Optimistic concurrency on ``update_run`` mirrors ``SqlAlchemyRunStore.transition``
    (read-check-mutate-commit in one transaction). ``claim_task`` goes further:
    the actual claim is a SQL ``UPDATE ... WHERE status='pending'`` whose
    ``rowcount`` is the atomic race-decider, so two concurrent workers cannot
    claim the same task.
    """

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

    # -- run lifecycle -------------------------------------------------

    async def create_run(self, run: SwarmRun) -> SwarmRun:
        async def _do(session):
            session.add(
                SwarmRunRow(
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
                )
            )

        await self._execute_in_session(_do)
        return run

    async def get_run(self, swarm_run_id: str) -> "SwarmRun | None":
        async def _do(session):
            result = await session.execute(
                select(SwarmRunRow).where(SwarmRunRow.id == swarm_run_id)
            )
            row = result.scalar_one_or_none()
            return None if row is None else _row_to_run(row)

        return await self._execute_in_session(_do)

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
        # DB-level CAS via UPDATE ... WHERE version=:expected,
        # mirroring SqlAlchemyRunStore.transition -- a Python read-check-mutate
        # pattern is not safe under concurrent updates on a real (non-SQLite)
        # backend. When a status change is requested, the WHERE clause also
        # restricts to the set of source statuses ALLOWED_SWARM_TRANSITIONS
        # permits, so the legality check is enforced atomically too.
        valid_sources = None
        if status is not None:
            # A "same status" update (e.g. bumping token_usage without a real
            # transition) is always allowed; a genuine transition additionally
            # requires ``status`` to be a legal target of the source per
            # ALLOWED_SWARM_TRANSITIONS.
            valid_sources = tuple(
                source.value
                for source in SwarmStatus
                if source == status
                or status in ALLOWED_SWARM_TRANSITIONS.get(source, frozenset())
            )

        async def _do(session):
            values: "dict" = {
                "version": SwarmRunRow.version + 1,
                "updated_at": datetime.now(timezone.utc),
            }
            if status is not None:
                values["status"] = status.value
            if round is not None:
                values["round"] = round
            if token_usage is not None:
                values["input_tokens"] = token_usage.input_tokens
                values["output_tokens"] = token_usage.output_tokens
            if cost is not None:
                values["total_cost"] = str(cost)
            if metadata is not None:
                values["metadata_json"] = json.dumps(metadata)

            stmt = (
                update(SwarmRunRow)
                .where(SwarmRunRow.id == swarm_run_id)
                .where(SwarmRunRow.version == expected_version)
            )
            if valid_sources is not None:
                stmt = stmt.where(SwarmRunRow.status.in_(valid_sources))
            stmt = stmt.values(**values)
            result_proxy = await session.execute(stmt)
            if result_proxy.rowcount == 0:
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
                current = SwarmStatus(row.status)
                raise InvalidSwarmTransitionError(
                    f"cannot transition {current} -> {status}"
                )
            query_result = await session.execute(
                select(SwarmRunRow).where(SwarmRunRow.id == swarm_run_id)
            )
            row = query_result.scalar_one()
            return _row_to_run(row)

        return await self._execute_in_session(_do)

    # -- task lifecycle ------------------------------------------------

    async def create_task(self, task: SwarmTask) -> SwarmTask:
        async def _do(session):
            session.add(
                SwarmTaskRow(
                    id=task.id,
                    swarm_run_id=task.swarm_run_id,
                    parent_task_id=task.parent_task_id,
                    assigned_agent_id=task.assigned_agent_id,
                    description=task.description,
                    status=task.status.value,
                    dependencies_json=json.dumps(list(task.dependencies)),
                    input_json=json.dumps(
                        {
                            "prompt": task.input.prompt,
                            "metadata": dict(task.input.metadata),
                        }
                    ),
                    result_json=None
                    if task.result is None
                    else _result_to_json(task.result),
                    error_json=None
                    if task.error is None
                    else _error_to_json(task.error),
                    attempts=task.attempts,
                    version=task.version,
                    claimed_at=task.claimed_at,
                    lease_expires_at=task.lease_expires_at,
                    created_at=task.created_at,
                    updated_at=task.updated_at,
                    active_run_id=task.active_run_id,
                )
            )

        await self._execute_in_session(_do)
        return task

    async def list_tasks(
        self, swarm_run_id: str, *, status: "SwarmTaskStatus | None" = None
    ) -> "tuple[SwarmTask, ...]":
        async def _do(session):
            query = select(SwarmTaskRow).where(
                SwarmTaskRow.swarm_run_id == swarm_run_id
            )
            if status is not None:
                query = query.where(SwarmTaskRow.status == status.value)
            query = query.order_by(SwarmTaskRow.created_at)
            result = await session.execute(query)
            return tuple(_row_to_task(row) for row in result.scalars())

        return await self._execute_in_session(_do)

    async def claim_task(
        self, swarm_run_id: str, agent_id: str, *, lease_seconds: "float | None" = None
    ) -> "SwarmTask | None":
        async def _do(session):
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
                    None
                    if lease_seconds is None
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

        return await self._execute_in_session(_do)

    async def set_active_run(
        self, task_id: str, run_id: str, *, expected_version: int
    ) -> SwarmTask:
        # Status guard added alongside expected_version (mirrors
        # complete_task/fail_task's own fencing): the strategy calls this
        # with the freshly-minted child RunRecord id right after claim_task
        # (task is CLAIMED). version alone is sound for pure optimistic
        # concurrency -- every mutating write (claim_task, complete_task,
        # fail_task, reclaim_expired_tasks, renew_lease) bumps it, so a
        # version match already implies no other write interleaved. The
        # explicit status == CLAIMED check is defense-in-depth (a clearer
        # SwarmConflictError message, and a second line of defense if some
        # future write path ever forgot to bump version) rather than a fix
        # for a currently reachable race.
        async def _do(session):
            result = await session.execute(
                update(SwarmTaskRow)
                .where(SwarmTaskRow.id == task_id)
                .where(SwarmTaskRow.version == expected_version)
                .where(SwarmTaskRow.status == SwarmTaskStatus.CLAIMED.value)
                .values(
                    active_run_id=run_id,
                    version=SwarmTaskRow.version + 1,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            if result.rowcount == 0:
                # Missing, version mismatch, or not CLAIMED -- read to discriminate.
                query_result = await session.execute(
                    select(SwarmTaskRow).where(SwarmTaskRow.id == task_id)
                )
                row = query_result.scalar_one_or_none()
                if row is None:
                    raise SwarmTaskNotFoundError(f"swarm task not found: {task_id}")
                if row.version != expected_version:
                    raise SwarmConflictError(
                        f"expected version {expected_version}, found {row.version}"
                    )
                raise SwarmConflictError(
                    f"task {task_id} is not claimed (status={row.status})"
                )
            query_result = await session.execute(
                select(SwarmTaskRow).where(SwarmTaskRow.id == task_id)
            )
            row = query_result.scalar_one()
            return _row_to_task(row)

        return await self._execute_in_session(_do)

    async def complete_task(
        self,
        task_id: str,
        result: RunResult,
        *,
        expected_version: int,
        active_run_id: "str | None" = None,
    ) -> SwarmTask:
        # expected_version is now
        # MANDATORY -- there is no unconditional fallback. DB-level
        # CAS via UPDATE ... WHERE version=:expected AND status='claimed'
        # AND (active_run_id IS NULL OR active_run_id=:active_run_id): a
        # worker whose lease already expired and was reclaimed to a new
        # owner cannot overwrite the new owner's progress with a stale
        # completion; a task no longer CLAIMED (already completed/failed by
        # a racing writer) is not silently re-completed; and (when the
        # caller supplies active_run_id) a worker driving a since-superseded
        # child Run cannot complete the task even if its version still
        # happened to match.
        async def _do(session):
            now = datetime.now(timezone.utc)
            stmt = (
                update(SwarmTaskRow)
                .where(SwarmTaskRow.id == task_id)
                .where(SwarmTaskRow.version == expected_version)
                .where(SwarmTaskRow.status == SwarmTaskStatus.CLAIMED.value)
            )
            if active_run_id is not None:
                stmt = stmt.where(SwarmTaskRow.active_run_id == active_run_id)
            stmt = stmt.values(
                status=SwarmTaskStatus.SUCCEEDED.value,
                result_json=_result_to_json(result),
                version=SwarmTaskRow.version + 1,
                updated_at=now,
            )
            result_proxy = await session.execute(stmt)
            if result_proxy.rowcount == 0:
                query_result = await session.execute(
                    select(SwarmTaskRow).where(SwarmTaskRow.id == task_id)
                )
                row = query_result.scalar_one_or_none()
                if row is None:
                    raise SwarmTaskNotFoundError(f"swarm task not found: {task_id}")
                if row.version != expected_version:
                    raise SwarmConflictError(
                        f"expected version {expected_version}, found {row.version}"
                    )
                if row.status != SwarmTaskStatus.CLAIMED.value:
                    raise SwarmConflictError(
                        f"task {task_id} is not claimed (status={row.status})"
                    )
                raise SwarmConflictError(
                    f"task {task_id} active_run_id mismatch: expected {active_run_id!r}, "
                    f"found {row.active_run_id!r}"
                )
            query_result = await session.execute(
                select(SwarmTaskRow).where(SwarmTaskRow.id == task_id)
            )
            return _row_to_task(query_result.scalar_one())

        return await self._execute_in_session(_do)

    async def fail_task(
        self,
        task_id: str,
        error: RunErrorInfo,
        *,
        expected_version: int,
        active_run_id: "str | None" = None,
    ) -> SwarmTask:
        # same mandatory fencing as complete_task.
        async def _do(session):
            now = datetime.now(timezone.utc)
            stmt = (
                update(SwarmTaskRow)
                .where(SwarmTaskRow.id == task_id)
                .where(SwarmTaskRow.version == expected_version)
                .where(SwarmTaskRow.status == SwarmTaskStatus.CLAIMED.value)
            )
            if active_run_id is not None:
                stmt = stmt.where(SwarmTaskRow.active_run_id == active_run_id)
            stmt = stmt.values(
                status=SwarmTaskStatus.FAILED.value,
                error_json=_error_to_json(error),
                attempts=SwarmTaskRow.attempts + 1,
                version=SwarmTaskRow.version + 1,
                updated_at=now,
            )
            result_proxy = await session.execute(stmt)
            if result_proxy.rowcount == 0:
                query_result = await session.execute(
                    select(SwarmTaskRow).where(SwarmTaskRow.id == task_id)
                )
                row = query_result.scalar_one_or_none()
                if row is None:
                    raise SwarmTaskNotFoundError(f"swarm task not found: {task_id}")
                if row.version != expected_version:
                    raise SwarmConflictError(
                        f"expected version {expected_version}, found {row.version}"
                    )
                if row.status != SwarmTaskStatus.CLAIMED.value:
                    raise SwarmConflictError(
                        f"task {task_id} is not claimed (status={row.status})"
                    )
                raise SwarmConflictError(
                    f"task {task_id} active_run_id mismatch: expected {active_run_id!r}, "
                    f"found {row.active_run_id!r}"
                )
            query_result = await session.execute(
                select(SwarmTaskRow).where(SwarmTaskRow.id == task_id)
            )
            return _row_to_task(query_result.scalar_one())

        return await self._execute_in_session(_do)

    async def reclaim_expired_tasks(self, swarm_run_id: str) -> "tuple[SwarmTask, ...]":
        # A select-then-loop-mutate here raced against a concurrent
        # renew_lease/complete_task/fail_task -- two overlapping
        # reclaim_expired_tasks calls could both select the same expired rows
        # before either commits, or a reclaim could stomp a lease a worker
        # legitimately renewed a moment ago. A single bulk
        # UPDATE ... WHERE status='claimed' AND lease_expires_at<:now
        # re-evaluates BOTH conditions atomically in the database at UPDATE
        # time, so a row a concurrent renew_lease just pushed past ``now`` is
        # simply not matched -- there is no read-then-write gap to race in.
        async def _do(session):
            now = datetime.now(timezone.utc)
            stmt = (
                update(SwarmTaskRow)
                .where(SwarmTaskRow.swarm_run_id == swarm_run_id)
                .where(SwarmTaskRow.status == SwarmTaskStatus.CLAIMED.value)
                .where(SwarmTaskRow.lease_expires_at < now)
                .values(
                    status=SwarmTaskStatus.PENDING.value,
                    assigned_agent_id=None,
                    claimed_at=None,
                    lease_expires_at=None,
                    version=SwarmTaskRow.version + 1,
                    updated_at=now,
                )
                .returning(SwarmTaskRow.id)
            )
            reclaimed_ids = [row.id for row in (await session.execute(stmt))]
            if not reclaimed_ids:
                return ()
            query_result = await session.execute(
                select(SwarmTaskRow).where(SwarmTaskRow.id.in_(reclaimed_ids))
            )
            return tuple(_row_to_task(row) for row in query_result.scalars())

        return await self._execute_in_session(_do)

    # -- lease renewal --------------------------------

    async def renew_lease(
        self, task_id: str, *, expected_version: int, lease_seconds: float
    ) -> SwarmTask:
        # UPDATE ... WHERE id=:tid AND version=:expected AND status='claimed':
        # the WHERE clauses make both the optimistic-concurrency check and the
        # CLAIMED-only guard atomic. rowcount == 0 means either missing, stale
        # version, or wrong status; the trailing SELECT discriminates so the
        # caller gets the right error class/message.
        async def _do(session):
            new_lease = datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
            result = await session.execute(
                update(SwarmTaskRow)
                .where(SwarmTaskRow.id == task_id)
                .where(SwarmTaskRow.version == expected_version)
                .where(SwarmTaskRow.status == SwarmTaskStatus.CLAIMED.value)
                .values(
                    lease_expires_at=new_lease,
                    version=SwarmTaskRow.version + 1,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            if result.rowcount == 0:
                query_result = await session.execute(
                    select(SwarmTaskRow).where(SwarmTaskRow.id == task_id)
                )
                row = query_result.scalar_one_or_none()
                if row is None:
                    raise SwarmTaskNotFoundError(f"swarm task not found: {task_id}")
                if row.version != expected_version:
                    raise SwarmConflictError(
                        f"expected version {expected_version}, found {row.version}"
                    )
                raise InvalidSwarmTransitionError(
                    f"renew_lease requires CLAIMED, task {task_id} is {row.status}"
                )
            query_result = await session.execute(
                select(SwarmTaskRow).where(SwarmTaskRow.id == task_id)
            )
            row = query_result.scalar_one()
            return _row_to_task(row)

        return await self._execute_in_session(_do)

    # -- attempts -------------------------------------

    async def record_attempt(self, attempt: SwarmTaskAttempt) -> SwarmTaskAttempt:
        # Upsert keyed on attempt.id. SQLite/SQLAlchemy has no native upsert
        # across dialects, so emulate: try to find the row, INSERT if missing,
        # UPDATE all mutable columns if present. The strategy writes the RUNNING
        # row before the worker call and the SUCCEEDED|FAILED row after with the
        # same id, so this path always sees exactly one prior row on the update.
        async def _do(session):
            query_result = await session.execute(
                select(SwarmTaskAttemptRow).where(SwarmTaskAttemptRow.id == attempt.id)
            )
            row = query_result.scalar_one_or_none()
            error_json = (
                None if attempt.error is None else _error_to_json(attempt.error)
            )
            if row is None:
                session.add(
                    SwarmTaskAttemptRow(
                        id=attempt.id,
                        task_id=attempt.task_id,
                        run_id=attempt.run_id,
                        agent_id=attempt.agent_id,
                        attempt=attempt.attempt,
                        status=attempt.status.value,
                        started_at=attempt.started_at,
                        finished_at=attempt.finished_at,
                        error_json=error_json,
                    )
                )
            else:
                row.status = attempt.status.value
                row.finished_at = attempt.finished_at
                row.error_json = error_json
            await session.flush()
            return attempt

        return await self._execute_in_session(_do)

    async def list_attempts(self, task_id: str) -> "tuple[SwarmTaskAttempt, ...]":
        async def _do(session):
            result = await session.execute(
                select(SwarmTaskAttemptRow)
                .where(SwarmTaskAttemptRow.task_id == task_id)
                .order_by(SwarmTaskAttemptRow.started_at, SwarmTaskAttemptRow.attempt)
            )
            return tuple(_row_to_attempt(row) for row in result.scalars())

        return await self._execute_in_session(_do)
