#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SQLAlchemy TaskStore with database-side fencing."""

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, DateTime, Integer, String, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from ...storage.sqlalchemy.base import Base
from ...storage.database import CoordinationScope
from ...storage.sqlalchemy.conventions import TABLE_PREFIX
from ...errors import StorageConflictError
from ...storage.coordination.lease import Lease, assert_active, claim, release, renew
from ..models import TaskExecution, TaskNode, TaskPlan, TaskStatus

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


class PlanRow(Base):
    __tablename__ = f"{TABLE_PREFIX}task_plans"
    plan_id: "Mapped[str]" = mapped_column(String(255), unique=True)
    payload: "Mapped[dict[str, Any]]" = mapped_column(JSON)


class ExecutionRow(Base):
    __tablename__ = f"{TABLE_PREFIX}task_executions"
    execution_id: "Mapped[str]" = mapped_column(String(255), unique=True)
    plan_id: "Mapped[str]" = mapped_column(String(255), index=True)
    node_id: "Mapped[str]" = mapped_column(String(255))
    status: "Mapped[str]" = mapped_column(String(32), index=True)
    owner: "Mapped[str | None]" = mapped_column(String(255))
    fence: "Mapped[int]" = mapped_column(Integer, default=0)
    attempt: "Mapped[int]" = mapped_column(Integer, default=0)
    result: "Mapped[Any]" = mapped_column(JSON, nullable=True)
    error: "Mapped[Any]" = mapped_column(JSON, nullable=True)
    lease_expires_at: "Mapped[Any]" = mapped_column(DateTime(timezone=True), nullable=True)


class SqlAlchemyTaskBackend:
    coordination_scope = CoordinationScope.SHARED_DATABASE

    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory

    async def initialize_storage(self, engine: "AsyncEngine") -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    @staticmethod
    async def _plan_row(session, plan_id: str):
        return await session.scalar(
            select(PlanRow).where(PlanRow.plan_id == plan_id)
        )

    @staticmethod
    async def _execution_row(session, execution_id: str):
        return await session.scalar(
            select(ExecutionRow).where(ExecutionRow.execution_id == execution_id)
        )

    @staticmethod
    def _plan(row: PlanRow) -> TaskPlan:
        return TaskPlan(row.plan_id, tuple(TaskNode(**node) for node in row.payload["nodes"]))

    @staticmethod
    def _execution(row: ExecutionRow) -> TaskExecution:
        return TaskExecution(row.execution_id, row.plan_id, row.node_id, TaskStatus(row.status), Lease(row.owner, row.fence, row.lease_expires_at), row.attempt, row.result, row.error, row.created_at, row.updated_at)

    @staticmethod
    async def _update_claimed(session, row: ExecutionRow, **values: object) -> ExecutionRow:
        result = await session.execute(
            update(ExecutionRow)
            .where(
                ExecutionRow.execution_id == row.execution_id,
                ExecutionRow.status == TaskStatus.CLAIMED.value,
                ExecutionRow.owner == row.owner,
                ExecutionRow.fence == row.fence,
                ExecutionRow.lease_expires_at == row.lease_expires_at,
            )
            .values(**values)
        )
        if result.rowcount != 1:
            raise StorageConflictError("task claim changed concurrently")
        # The CAS matched exactly one row at the held state, so the UPDATE's
        # values are now the row's true state -- reflect them onto the loaded
        # ORM object and build the record from it, skipping a re-SELECT.
        for key, value in values.items():
            setattr(row, key, value)
        return row

    async def save_plan(self, plan: TaskPlan) -> None:
        payload = {"nodes": [asdict(node) for node in plan.nodes]}
        for attempt in range(2):
            try:
                async with self.session_factory() as session:
                    async with session.begin():
                        result = await session.execute(
                            update(PlanRow)
                            .where(PlanRow.plan_id == plan.id)
                            .values(payload=payload)
                        )
                        if result.rowcount == 0:
                            session.add(PlanRow(plan_id=plan.id, payload=payload))
                            await session.flush()
                return
            except IntegrityError:
                if attempt:
                    raise StorageConflictError("task plan write conflict")

    async def get_plan(self, plan_id: str) -> "TaskPlan | None":
        async with self.session_factory() as session:
            row = await self._plan_row(session, plan_id)
            return None if row is None else self._plan(row)

    async def add_execution(self, execution: TaskExecution) -> None:
        try:
            async with self.session_factory() as session:
                async with session.begin():
                    if await self._execution_row(session, execution.id) is not None:
                        raise StorageConflictError("task execution already exists")
                    session.add(ExecutionRow(execution_id=execution.id, plan_id=execution.plan_id, node_id=execution.node_id, status=execution.status.value, owner=execution.owner, fence=execution.fence, attempt=execution.attempt, result=execution.result, error=execution.error, lease_expires_at=execution.lease.expires_at))
                    await session.flush()
        except IntegrityError as exc:
            raise StorageConflictError("task execution already exists") from exc

    create_execution = add_execution

    async def get_execution(self, execution_id: str) -> "TaskExecution | None":
        async with self.session_factory() as session:
            row = await self._execution_row(session, execution_id)
            return None if row is None else self._execution(row)

    async def claim(self, execution_id: str, *, owner: str, duration: timedelta = timedelta(minutes=5)) -> TaskExecution:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._execution_row(session, execution_id)
                if row is None:
                    raise KeyError(execution_id)
                if row.status == TaskStatus.COMPLETED.value:
                    return self._execution(row)
                if row.status == TaskStatus.FAILED.value:
                    raise StorageConflictError("failed task cannot be claimed")
                now = datetime.now(timezone.utc)
                lease = claim(
                    Lease(row.owner, row.fence, row.lease_expires_at),
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
                        owner=lease.owner,
                        fence=lease.fence,
                        lease_expires_at=lease.expires_at,
                        attempt=row.attempt + 1,
                        updated_at=now,
                    )
                )
                if result.rowcount != 1:
                    raise StorageConflictError("task claim conflict")
                # CAS matched at the held state: reflect the UPDATE's values onto
                # the loaded row and build the record without a re-SELECT.
                row.status = TaskStatus.CLAIMED.value
                row.owner = lease.owner
                row.fence = lease.fence
                row.lease_expires_at = lease.expires_at
                row.attempt = row.attempt + 1
                row.updated_at = now
                return self._execution(row)

    async def renew(self, execution_id: str, *, owner: str, fence: int, duration: timedelta = timedelta(minutes=5)) -> TaskExecution:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._execution_row(session, execution_id)
                if row is None:
                    raise KeyError(execution_id)
                now = datetime.now(timezone.utc)
                lease = renew(
                    Lease(row.owner, row.fence, row.lease_expires_at),
                    owner=owner,
                    fence=fence,
                    now=now,
                    duration=duration,
                )
                updated = await self._update_claimed(
                    session,
                    row,
                    lease_expires_at=lease.expires_at,
                    updated_at=now,
                )
                return self._execution(updated)

    async def complete(self, execution_id: str, *, owner: str, fence: int, result: object) -> TaskExecution:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._execution_row(session, execution_id)
                if row is None:
                    raise KeyError(execution_id)
                now = datetime.now(timezone.utc)
                assert_active(Lease(row.owner, row.fence, row.lease_expires_at), owner=owner, fence=fence, now=now)
                if row.status != TaskStatus.CLAIMED.value:
                    raise StorageConflictError("task is not claimed")
                lease = release(Lease(row.owner, row.fence, row.lease_expires_at))
                updated = await self._update_claimed(
                    session,
                    row,
                    status=TaskStatus.COMPLETED.value,
                    result=result,
                    owner=lease.owner,
                    lease_expires_at=lease.expires_at,
                    updated_at=now,
                )
                return self._execution(updated)

    async def fail(
        self,
        execution_id: str,
        *,
        owner: str,
        fence: int,
        retry: bool = False,
        error: object = None,
    ) -> TaskExecution:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._execution_row(session, execution_id)
                if row is None:
                    raise KeyError(execution_id)
                now = datetime.now(timezone.utc)
                assert_active(Lease(row.owner, row.fence, row.lease_expires_at), owner=owner, fence=fence, now=now)
                if row.status != TaskStatus.CLAIMED.value:
                    raise StorageConflictError("task is not claimed")
                lease = release(Lease(row.owner, row.fence, row.lease_expires_at))
                updated = await self._update_claimed(
                    session,
                    row,
                    status=TaskStatus.READY.value if retry else TaskStatus.FAILED.value,
                    owner=lease.owner,
                    lease_expires_at=lease.expires_at,
                    error=error,
                    updated_at=now,
                )
                return self._execution(updated)
