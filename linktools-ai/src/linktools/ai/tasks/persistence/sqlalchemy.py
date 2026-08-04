#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SQLAlchemy TaskStore with database-side fencing.

Every state transition is a single conditional UPDATE whose WHERE clause
re-checks the previously-read (status, owner, fence) so a stale fence or a
racing writer updates zero rows; rowcount != 1 is a StorageConflictError. The
fence column never resets, so a stale owner can never re-win. create_plan
inserts ``plan`` and every node execution in one transaction; any failure
rolls the whole batch back."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, DateTime, Integer, String, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from linktools.core import environ

from ...errors import StorageConflictError, UsageRegressionError
from ...execution.domain import RunError
from ...json import JsonValue
from ...storage.coordination.lease import Lease, claim, is_expired, renew
from ...storage.database import CoordinationScope
from ...storage.sqlalchemy.base import Base
from ...storage.sqlalchemy.conventions import TABLE_PREFIX
from ...storage.sqlalchemy.dialects import resolve_dialect
from ..codec import decode_plan, encode_plan
from ..models import (
    TaskExecution,
    TaskPlan,
    TaskStatus,
    TaskUsage,
    apply_usage_revision,
)

logger = environ.get_logger("ai.tasks.persistence.sqlalchemy")

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from ...storage.sqlalchemy.dialects import SqlAlchemyDialect


class PlanRow(Base):
    __tablename__ = f"{TABLE_PREFIX}task_plans"
    plan_id: "Mapped[str]" = mapped_column(String(255), unique=True)
    payload: "Mapped[dict[str, Any]]" = mapped_column(JSON)


class ExecutionRow(Base):
    __tablename__ = f"{TABLE_PREFIX}task_executions"
    execution_id: "Mapped[str]" = mapped_column(String(255), unique=True)
    plan_id: "Mapped[str]" = mapped_column(
        String(255), index=True
    )
    node_id: "Mapped[str]" = mapped_column(String(255))
    status: "Mapped[str]" = mapped_column(String(32))
    owner: "Mapped[str | None]" = mapped_column(String(255))
    fence: "Mapped[int]" = mapped_column(Integer, default=0)
    attempt: "Mapped[int]" = mapped_column(Integer, default=0)
    active_run_id: "Mapped[str | None]" = mapped_column(String(255), nullable=True)
    result: "Mapped[Any]" = mapped_column(JSON, nullable=True)
    error: "Mapped[Any]" = mapped_column(JSON, nullable=True)
    blocked_by: "Mapped[Any]" = mapped_column(JSON, nullable=False, default=list)
    terminal_reason: "Mapped[str | None]" = mapped_column(String(255), nullable=True)
    usage: "Mapped[Any]" = mapped_column(JSON, nullable=False, default=dict)
    usage_revision: "Mapped[int]" = mapped_column(Integer, default=0)
    lease_expires_at: "Mapped[Any]" = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: "Mapped[Any]" = mapped_column(DateTime(timezone=True))
    updated_at: "Mapped[Any]" = mapped_column(DateTime(timezone=True))


class SqlAlchemyTaskBackend:
    coordination_scope = CoordinationScope.SHARED_DATABASE

    def __init__(
        self, session_factory, *, dialect: "SqlAlchemyDialect | None" = None
    ) -> None:
        self.session_factory = session_factory
        self._dialect = dialect

    async def initialize_storage(self, engine: "AsyncEngine") -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            await connection.execute(
                text(
                    f"CREATE UNIQUE INDEX IF NOT EXISTS "
                    f"ix_{TABLE_PREFIX}task_exec_plan_node "
                    f"ON {TABLE_PREFIX}task_executions (plan_id, node_id)"
                )
            )
            await connection.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS "
                    f"ix_{TABLE_PREFIX}task_exec_plan_status "
                    f"ON {TABLE_PREFIX}task_executions (plan_id, status)"
                )
            )

    async def _dialect_for(self, session: "AsyncSession") -> "SqlAlchemyDialect":
        if self._dialect is None:
            self._dialect = resolve_dialect(session)
        return self._dialect

    @staticmethod
    async def _plan_row(session: "AsyncSession", plan_id: str):
        return await session.scalar(select(PlanRow).where(PlanRow.plan_id == plan_id))

    @staticmethod
    async def _execution_row(
        session: "AsyncSession", execution_id: str
    ) -> "ExecutionRow | None":
        return await session.scalar(
            select(ExecutionRow).where(ExecutionRow.execution_id == execution_id)
        )

    @staticmethod
    def _plan(row: PlanRow) -> TaskPlan:
        return decode_plan(row.payload)

    @staticmethod
    def _execution(row: ExecutionRow) -> TaskExecution:
        return TaskExecution(
            id=row.execution_id,
            plan_id=row.plan_id,
            node_id=row.node_id,
            status=TaskStatus(row.status),
            lease=Lease(row.owner, row.fence, row.lease_expires_at),
            attempt=row.attempt,
            active_run_id=row.active_run_id,
            result=row.result,
            error=_decode_error(row.error),
            blocked_by=tuple(row.blocked_by or ()),
            terminal_reason=row.terminal_reason,
            usage=_decode_usage(row.usage),
            usage_revision=row.usage_revision,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    async def create_plan(
        self,
        plan: TaskPlan,
        executions: "tuple[TaskExecution, ...]",
    ) -> None:
        payload = encode_plan(plan)
        try:
            async with self.session_factory() as session:
                async with session.begin():
                    dialect = await self._dialect_for(session)
                    result = await dialect.insert_ignore_conflict(
                        session,
                        model=PlanRow,
                        values={"plan_id": plan.id, "payload": payload},
                        index_elements=("plan_id",),
                    )
                    if not result.inserted:
                        raise StorageConflictError(
                            f"task plan {plan.id!r} already exists"
                        )
                    for execution in executions:
                        session.add(self._execution_row_for(execution))
                    await session.flush()
        except IntegrityError as exc:
            raise StorageConflictError(
                f"task plan {plan.id!r} or a node already exists"
            ) from exc

    @staticmethod
    def _execution_row_for(execution: TaskExecution) -> ExecutionRow:
        return ExecutionRow(
            execution_id=execution.id,
            plan_id=execution.plan_id,
            node_id=execution.node_id,
            status=execution.status.value,
            owner=execution.lease.owner,
            fence=execution.lease.fence,
            attempt=execution.attempt,
            active_run_id=execution.active_run_id,
            result=execution.result,
            error=_encode_error(execution.error),
            blocked_by=list(execution.blocked_by),
            terminal_reason=execution.terminal_reason,
            usage=_encode_usage(execution.usage),
            usage_revision=execution.usage_revision,
            lease_expires_at=execution.lease.expires_at,
            created_at=execution.created_at,
            updated_at=execution.updated_at,
        )

    async def get_plan(self, plan_id: str) -> "TaskPlan | None":
        async with self.session_factory() as session:
            row = await self._plan_row(session, plan_id)
            return None if row is None else self._plan(row)

    async def list_executions(self, plan_id: str) -> "tuple[TaskExecution, ...]":
        async with self.session_factory() as session:
            result = await session.scalars(
                select(ExecutionRow)
                .where(ExecutionRow.plan_id == plan_id)
                .order_by(ExecutionRow.created_at)
            )
            return tuple(self._execution(row) for row in result)

    async def get_execution(self, execution_id: str) -> "TaskExecution | None":
        async with self.session_factory() as session:
            row = await self._execution_row(session, execution_id)
            return None if row is None else self._execution(row)

    async def claim_ready(
        self,
        execution_id: str,
        *,
        owner: str,
        duration: timedelta,
    ) -> TaskExecution:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._execution_row(session, execution_id)
                if row is None:
                    raise StorageConflictError(
                        f"task execution {execution_id!r} not found"
                    )
                now = datetime.now(timezone.utc)
                current_lease = Lease(row.owner, row.fence, row.lease_expires_at)
                if row.status != TaskStatus.READY.value:
                    raise StorageConflictError(
                        f"task {execution_id!r} is {row.status}, not claimable"
                    )
                new_lease = claim(
                    current_lease,
                    owner=owner,
                    now=now,
                    duration=duration,
                )
                result = await session.execute(
                    update(ExecutionRow)
                    .where(
                        ExecutionRow.execution_id == execution_id,
                        ExecutionRow.status == row.status,
                        ExecutionRow.fence == row.fence,
                    )
                    .values(
                        status=TaskStatus.CLAIMED.value,
                        owner=new_lease.owner,
                        fence=new_lease.fence,
                        lease_expires_at=new_lease.expires_at,
                        attempt=1,
                        updated_at=now,
                    )
                )
                if result.rowcount != 1:
                    raise StorageConflictError("task claim lost a race")
                row.status = TaskStatus.CLAIMED.value
                row.owner = new_lease.owner
                row.fence = new_lease.fence
                row.lease_expires_at = new_lease.expires_at
                row.attempt = 1
                row.updated_at = now
                return self._execution(row)

    async def take_over_expired_claim_for_reconcile(
        self,
        execution_id: str,
        *,
        owner: str,
        now: datetime,
        duration: timedelta,
    ) -> TaskExecution:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._execution_row(session, execution_id)
                if row is None:
                    raise StorageConflictError(f"task execution {execution_id!r} not found")
                current_lease = Lease(row.owner, row.fence, row.lease_expires_at)
                if row.status != TaskStatus.CLAIMED.value or not is_expired(current_lease, now):
                    raise StorageConflictError("task is not an expired CLAIMED execution")
                new_lease = claim(current_lease, owner=owner, now=now, duration=duration)
                result = await session.execute(
                    update(ExecutionRow)
                    .where(
                        ExecutionRow.execution_id == execution_id,
                        ExecutionRow.status == TaskStatus.CLAIMED.value,
                        ExecutionRow.owner == row.owner,
                        ExecutionRow.fence == row.fence,
                        ExecutionRow.lease_expires_at <= now,
                    )
                    .execution_options(synchronize_session=False)
                    .values(owner=owner, fence=new_lease.fence, lease_expires_at=new_lease.expires_at, updated_at=now)
                )
                if result.rowcount != 1:
                    raise StorageConflictError("expired task reconcile lost the race")
                row.owner = owner
                row.fence = new_lease.fence
                row.lease_expires_at = new_lease.expires_at
                row.updated_at = now
                return self._execution(row)

    async def bind_child_run(
        self,
        execution_id: str,
        *,
        owner: str,
        fence: int,
        child_run_id: str,
    ) -> TaskExecution:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._claimed_row(session, execution_id, owner, fence)
                if row.active_run_id is not None and row.active_run_id != child_run_id:
                    raise StorageConflictError(
                        f"task {execution_id!r} already bound to a different child run"
                    )
                now = datetime.now(timezone.utc)
                result = await session.execute(
                    self._claimed_guard(
                        execution_id,
                        owner,
                        fence,
                        now,
                    ).values(
                        active_run_id=child_run_id,
                        updated_at=now,
                    )
                )
                if result.rowcount != 1:
                    raise StorageConflictError("task bind lost a race")
                row.active_run_id = child_run_id
                row.updated_at = now
                return self._execution(row)

    async def renew(
        self,
        execution_id: str,
        *,
        owner: str,
        fence: int,
        duration: timedelta,
    ) -> TaskExecution:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._claimed_row(session, execution_id, owner, fence)
                now = datetime.now(timezone.utc)
                new_lease = renew(
                    Lease(row.owner, row.fence, row.lease_expires_at),
                    owner=owner,
                    fence=fence,
                    now=now,
                    duration=duration,
                )
                result = await session.execute(
                    self._claimed_guard(
                        execution_id,
                        owner,
                        fence,
                        now,
                    ).values(
                        lease_expires_at=new_lease.expires_at,
                        updated_at=now,
                    )
                )
                if result.rowcount != 1:
                    raise StorageConflictError("task renew lost a race")
                row.lease_expires_at = new_lease.expires_at
                row.updated_at = now
                return self._execution(row)

    async def record_claimed_usage(
        self,
        execution_id: str,
        *,
        owner: str,
        fence: int,
        snapshot_revision: int,
        usage: TaskUsage,
    ) -> TaskExecution:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._claimed_row(session, execution_id, owner, fence)
                old_usage = _decode_usage(row.usage)
                revision, merged, changed = apply_usage_revision(
                    current_revision=row.usage_revision,
                    current_usage=old_usage,
                    incoming_revision=snapshot_revision,
                    incoming_usage=usage,
                )
                if not changed:
                    return self._execution(row)
                now = datetime.now(timezone.utc)
                result = await session.execute(
                    self._claimed_guard(
                        execution_id,
                        owner,
                        fence,
                        now,
                        expected_revision=row.usage_revision,
                    ).values(
                        usage=_encode_usage(merged),
                        usage_revision=revision,
                        updated_at=now,
                    )
                )
                if result.rowcount != 1:
                    current_row = await self._execution_row(session, execution_id)
                    if current_row is None:
                        raise StorageConflictError(
                            f"task execution {execution_id!r} disappeared"
                        )
                    current = self._execution(current_row)
                    if current.status is not TaskStatus.CLAIMED:
                        raise StorageConflictError(
                            f"task {execution_id!r} is no longer claimed"
                        )
                    _, _, changed_again = apply_usage_revision(
                        current_revision=current.usage_revision,
                        current_usage=current.usage,
                        incoming_revision=snapshot_revision,
                        incoming_usage=usage,
                    )
                    if not changed_again:
                        return current
                    raise StorageConflictError("task usage update lost a race")
                row.usage = _encode_usage(merged)
                row.usage_revision = revision
                row.updated_at = now
                return self._execution(row)

    async def complete(
        self,
        execution_id: str,
        *,
        owner: str,
        fence: int,
        result: "JsonValue",
        snapshot_revision: int,
        usage: TaskUsage,
    ) -> TaskExecution:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._claimed_row(session, execution_id, owner, fence)
                revision, merged, _ = _terminal_usage(
                    row, snapshot_revision=snapshot_revision, usage=usage
                )
                now = datetime.now(timezone.utc)
                outcome = await session.execute(
                    self._claimed_guard(
                        execution_id,
                        owner,
                        fence,
                        now,
                        expected_revision=row.usage_revision,
                    ).values(
                        status=TaskStatus.COMPLETED.value,
                        result=result,
                        usage=_encode_usage(merged),
                        usage_revision=revision,
                        lease_expires_at=None,
                        updated_at=now,
                    )
                )
                if outcome.rowcount != 1:
                    raise StorageConflictError("task complete lost a race")
                row.status = TaskStatus.COMPLETED.value
                row.result = result
                row.usage = _encode_usage(merged)
                row.usage_revision = revision
                row.lease_expires_at = None
                row.updated_at = now
                return self._execution(row)

    async def fail(
        self,
        execution_id: str,
        *,
        owner: str,
        fence: int,
        error: RunError,
        snapshot_revision: int,
        usage: TaskUsage,
    ) -> TaskExecution:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._claimed_row(session, execution_id, owner, fence)
                revision, merged, _ = _terminal_usage(
                    row, snapshot_revision=snapshot_revision, usage=usage
                )
                now = datetime.now(timezone.utc)
                outcome = await session.execute(
                    self._claimed_guard(
                        execution_id,
                        owner,
                        fence,
                        now,
                        expected_revision=row.usage_revision,
                    ).values(
                        status=TaskStatus.FAILED.value,
                        error=_encode_error(error),
                        usage=_encode_usage(merged),
                        usage_revision=revision,
                        lease_expires_at=None,
                        updated_at=now,
                    )
                )
                if outcome.rowcount != 1:
                    raise StorageConflictError("task fail lost a race")
                row.status = TaskStatus.FAILED.value
                row.error = _encode_error(error)
                row.usage = _encode_usage(merged)
                row.usage_revision = revision
                row.lease_expires_at = None
                row.updated_at = now
                return self._execution(row)

    async def skip(
        self,
        execution_id: str,
        *,
        blocked_by: "tuple[str, ...]",
        reason: str,
    ) -> TaskExecution:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._execution_row(session, execution_id)
                if row is None:
                    raise StorageConflictError(
                        f"task execution {execution_id!r} not found"
                    )
                if row.status == TaskStatus.SKIPPED.value:
                    return self._execution(row)
                if row.status != TaskStatus.READY.value:
                    raise StorageConflictError(
                        f"task {execution_id!r} is {row.status}, cannot skip"
                    )
                now = datetime.now(timezone.utc)
                outcome = await session.execute(
                    update(ExecutionRow)
                    .where(
                        ExecutionRow.execution_id == execution_id,
                        ExecutionRow.status == TaskStatus.READY.value,
                    )
                    .values(
                        status=TaskStatus.SKIPPED.value,
                        blocked_by=list(blocked_by),
                        terminal_reason=reason,
                        updated_at=now,
                    )
                )
                if outcome.rowcount != 1:
                    raise StorageConflictError("task skip lost a race")
                row.status = TaskStatus.SKIPPED.value
                row.blocked_by = list(blocked_by)
                row.terminal_reason = reason
                row.updated_at = now
                return self._execution(row)

    async def cancel_ready(
        self,
        execution_id: str,
        *,
        reason: str,
    ) -> TaskExecution:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._execution_row(session, execution_id)
                if row is None:
                    raise StorageConflictError(
                        f"task execution {execution_id!r} not found"
                    )
                if row.status in _TERMINAL_VALUES:
                    return self._execution(row)
                if row.status != TaskStatus.READY.value:
                    raise StorageConflictError(
                        f"task {execution_id!r} is {row.status}, use cancel_claimed"
                    )
                now = datetime.now(timezone.utc)
                outcome = await session.execute(
                    update(ExecutionRow)
                    .where(
                        ExecutionRow.execution_id == execution_id,
                        ExecutionRow.status == TaskStatus.READY.value,
                    )
                    .values(
                        status=TaskStatus.CANCELLED.value,
                        terminal_reason=reason,
                        updated_at=now,
                    )
                )
                if outcome.rowcount != 1:
                    raise StorageConflictError("task cancel_ready lost a race")
                row.status = TaskStatus.CANCELLED.value
                row.terminal_reason = reason
                row.updated_at = now
                return self._execution(row)

    async def cancel_claimed(
        self,
        execution_id: str,
        *,
        owner: str,
        fence: int,
        reason: str,
        snapshot_revision: int,
        usage: TaskUsage,
    ) -> TaskExecution:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._claimed_row(session, execution_id, owner, fence)
                revision, merged, _ = _terminal_usage(
                    row, snapshot_revision=snapshot_revision, usage=usage
                )
                now = datetime.now(timezone.utc)
                outcome = await session.execute(
                    self._claimed_guard(
                        execution_id,
                        owner,
                        fence,
                        now,
                        expected_revision=row.usage_revision,
                    ).values(
                        status=TaskStatus.CANCELLED.value,
                        terminal_reason=reason,
                        usage=_encode_usage(merged),
                        usage_revision=revision,
                        lease_expires_at=None,
                        updated_at=now,
                    )
                )
                if outcome.rowcount != 1:
                    raise StorageConflictError("task cancel_claimed lost a race")
                row.status = TaskStatus.CANCELLED.value
                row.terminal_reason = reason
                row.usage = _encode_usage(merged)
                row.usage_revision = revision
                row.lease_expires_at = None
                row.updated_at = now
                return self._execution(row)

    async def _claimed_row(
        self,
        session: "AsyncSession",
        execution_id: str,
        owner: str,
        fence: int,
    ) -> ExecutionRow:
        row = await self._execution_row(session, execution_id)
        if row is None:
            raise StorageConflictError(f"task execution {execution_id!r} not found")
        if row.status != TaskStatus.CLAIMED.value:
            raise StorageConflictError(
                f"task {execution_id!r} is {row.status}, not claimed"
            )
        if row.owner != owner or row.fence != fence:
            raise StorageConflictError("stale fence for task execution")
        expires_at = row.lease_expires_at
        if expires_at is not None and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at is None or expires_at <= datetime.now(timezone.utc):
            raise StorageConflictError("task execution lease expired")
        return row

    @staticmethod
    def _claimed_guard(
        execution_id: str,
        owner: str,
        fence: int,
        now: datetime,
        *,
        expected_revision: "int | None" = None,
    ):
        revision_clause = (
            ()
            if expected_revision is None
            else (ExecutionRow.usage_revision == expected_revision,)
        )
        return (
            update(ExecutionRow)
            .where(
                ExecutionRow.execution_id == execution_id,
                ExecutionRow.status == TaskStatus.CLAIMED.value,
                ExecutionRow.owner == owner,
                ExecutionRow.fence == fence,
                ExecutionRow.lease_expires_at > now,
                *revision_clause,
            )
            .execution_options(synchronize_session=False)
        )


_TERMINAL_VALUES = frozenset(
    {
        TaskStatus.COMPLETED.value,
        TaskStatus.FAILED.value,
        TaskStatus.SKIPPED.value,
        TaskStatus.CANCELLED.value,
    }
)


def _encode_usage(usage: TaskUsage) -> "dict[str, JsonValue]":
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_cost": (
            None if usage.total_cost is None else format(usage.total_cost, "f")
        ),
        "cache_write_tokens": usage.cache_write_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
    }


def _decode_usage(value: "Any") -> TaskUsage:
    if not value:
        return TaskUsage()
    total_cost = value.get("total_cost")
    return TaskUsage(
        input_tokens=int(value.get("input_tokens", 0)),
        output_tokens=int(value.get("output_tokens", 0)),
        total_cost=Decimal(total_cost) if total_cost is not None else None,
        cache_write_tokens=int(value.get("cache_write_tokens", 0)),
        cache_read_tokens=int(value.get("cache_read_tokens", 0)),
    )


def _terminal_usage(
    row: ExecutionRow, *, snapshot_revision: int, usage: TaskUsage
) -> "tuple[int, TaskUsage, bool]":
    if snapshot_revision < row.usage_revision:
        raise UsageRegressionError("terminal snapshot revision decreased")
    return apply_usage_revision(
        current_revision=row.usage_revision,
        current_usage=_decode_usage(row.usage),
        incoming_revision=snapshot_revision,
        incoming_usage=usage,
    )


def _encode_error(error: "RunError | None") -> "dict[str, JsonValue] | None":
    if error is None:
        return None
    return {
        "error_type": error.error_type,
        "message": error.message,
        "detail": error.detail,
    }


def _decode_error(value: "Any") -> "RunError | None":
    if not value:
        return None
    return RunError(
        error_type=value["error_type"],
        message=value["message"],
        detail=value.get("detail"),
    )


__all__ = ["SqlAlchemyTaskBackend"]
