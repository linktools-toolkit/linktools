"""SQLAlchemy TaskStore with database-side fencing."""

from dataclasses import asdict
from typing import Any

from sqlalchemy import JSON, Integer, String, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from ...storage.sqlalchemy.base import Base
from ...storage.sqlalchemy.conventions import TABLE_PREFIX
from ..models import TaskExecution, TaskNode, TaskPlan


class PlanRow(Base):
    __tablename__ = f"{TABLE_PREFIX}task_plans"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class ExecutionRow(Base):
    __tablename__ = f"{TABLE_PREFIX}task_executions"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    plan_id: Mapped[str] = mapped_column(String(255), index=True)
    node_id: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), index=True)
    owner: Mapped[str | None] = mapped_column(String(255))
    fence: Mapped[int] = mapped_column(Integer, default=0)
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    result: Mapped[Any] = mapped_column(JSON, nullable=True)


class SqlAlchemyTaskStore:
    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory

    async def initialize_storage(self, engine) -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    @staticmethod
    def _plan(row: PlanRow) -> TaskPlan:
        return TaskPlan(row.id, tuple(TaskNode(**node) for node in row.payload["nodes"]))

    @staticmethod
    def _execution(row: ExecutionRow) -> TaskExecution:
        return TaskExecution(row.id, row.plan_id, row.node_id, row.status, row.owner, row.fence, row.attempt, row.result)

    async def save_plan(self, plan: TaskPlan) -> None:
        payload = {"nodes": [asdict(node) for node in plan.nodes]}
        async with self.session_factory() as session:
            async with session.begin():
                row = await session.get(PlanRow, plan.id, with_for_update=True)
                if row is None:
                    session.add(PlanRow(id=plan.id, payload=payload))
                else:
                    row.payload = payload

    async def get_plan(self, plan_id: str) -> TaskPlan | None:
        async with self.session_factory() as session:
            row = await session.get(PlanRow, plan_id)
            return None if row is None else self._plan(row)

    async def add_execution(self, execution: TaskExecution) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                if await session.get(ExecutionRow, execution.id) is not None:
                    raise ValueError("task execution already exists")
                session.add(ExecutionRow(id=execution.id, plan_id=execution.plan_id, node_id=execution.node_id, status=execution.status, owner=execution.owner, fence=execution.fence, attempt=execution.attempt, result=execution.result))

    create_execution = add_execution

    async def get_execution(self, execution_id: str) -> TaskExecution | None:
        async with self.session_factory() as session:
            row = await session.get(ExecutionRow, execution_id)
            return None if row is None else self._execution(row)

    async def claim(self, execution_id: str, *, owner: str) -> TaskExecution:
        async with self.session_factory() as session:
            async with session.begin():
                row = await session.get(ExecutionRow, execution_id, with_for_update=True)
                if row is None:
                    raise KeyError(execution_id)
                if row.status == "completed":
                    return self._execution(row)
                row.status, row.owner = "claimed", owner
                row.fence, row.attempt = row.fence + 1, row.attempt + 1
                return self._execution(row)

    async def renew(self, execution_id: str, *, owner: str, fence: int) -> TaskExecution:
        async with self.session_factory() as session:
            row = await session.get(ExecutionRow, execution_id)
            if row is None or row.owner != owner or row.fence != fence:
                raise ValueError("task fence conflict")
            return self._execution(row)

    async def complete(self, execution_id: str, *, owner: str, fence: int, result: object) -> TaskExecution:
        async with self.session_factory() as session:
            async with session.begin():
                row = await session.get(ExecutionRow, execution_id, with_for_update=True)
                if row is None or row.owner != owner or row.fence != fence:
                    raise ValueError("task fence conflict")
                row.status, row.result = "completed", result
                return self._execution(row)

    async def fail(self, execution_id: str, *, owner: str, fence: int, retry: bool = False) -> TaskExecution:
        async with self.session_factory() as session:
            async with session.begin():
                row = await session.get(ExecutionRow, execution_id, with_for_update=True)
                if row is None or row.owner != owner or row.fence != fence:
                    raise ValueError("task fence conflict")
                row.status = "ready" if retry else "failed"
                return self._execution(row)
