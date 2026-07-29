"""Structural task-store contract implemented directly by backends."""

from datetime import timedelta
from typing import Protocol

from ..storage.database import CoordinationScope
from .models import TaskExecution, TaskPlan


class TaskStore(Protocol):
    coordination_scope: CoordinationScope

    async def save_plan(self, plan: TaskPlan) -> None: ...

    async def get_plan(self, plan_id: str) -> TaskPlan | None: ...

    async def get_execution(
        self, execution_id: str
    ) -> TaskExecution | None: ...

    async def add_execution(self, execution: TaskExecution) -> None: ...

    async def claim(
        self,
        execution_id: str,
        *,
        owner: str,
        duration: timedelta = timedelta(minutes=5),
    ) -> TaskExecution: ...

    async def renew(
        self,
        execution_id: str,
        *,
        owner: str,
        fence: int,
        duration: timedelta = timedelta(minutes=5),
    ) -> TaskExecution: ...

    async def complete(
        self, execution_id: str, *, owner: str, fence: int, result: object
    ) -> TaskExecution: ...

    async def fail(
        self,
        execution_id: str,
        *,
        owner: str,
        fence: int,
        retry: bool = False,
        error: object = None,
    ) -> TaskExecution: ...


__all__ = ["TaskStore"]
