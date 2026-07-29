"""Task backend contract and composed Store."""

from datetime import timedelta
from typing import Protocol

from .models import TaskExecution, TaskPlan


class TaskBackend(Protocol):
    async def get_plan(self, plan_id: str) -> TaskPlan | None: ...
    async def get_execution(self, execution_id: str) -> TaskExecution | None: ...
    async def save_plan(self, plan: TaskPlan) -> None: ...
    async def add_execution(self, execution: TaskExecution) -> None: ...
    async def claim(self, execution_id: str, *, owner: str, duration: timedelta = ...) -> TaskExecution: ...
    async def renew(self, execution_id: str, *, owner: str, fence: int, duration: timedelta = ...) -> TaskExecution: ...
    async def complete(self, execution_id: str, *, owner: str, fence: int, result: object) -> TaskExecution: ...
    async def fail(
        self,
        execution_id: str,
        *,
        owner: str,
        fence: int,
        retry: bool = False,
        error: object = None,
    ) -> TaskExecution: ...


class TaskStore:
    def __init__(self, backend: TaskBackend) -> None:
        self._backend = backend

    @property
    def backend(self) -> TaskBackend:
        return self._backend

    async def initialize_storage(self, *args: object) -> None:
        await self._backend.initialize_storage(*args)

    async def get_plan(self, plan_id: str) -> TaskPlan | None:
        return await self._backend.get_plan(plan_id)

    async def get_execution(self, execution_id: str) -> TaskExecution | None:
        return await self._backend.get_execution(execution_id)

    async def save_plan(self, plan: TaskPlan) -> None:
        await self._backend.save_plan(plan)

    async def add_execution(self, execution: TaskExecution) -> None:
        await self._backend.add_execution(execution)

    async def claim(self, execution_id: str, *, owner: str, duration: timedelta = timedelta(minutes=5)) -> TaskExecution:
        return await self._backend.claim(execution_id, owner=owner, duration=duration)

    async def renew(self, execution_id: str, *, owner: str, fence: int, duration: timedelta = timedelta(minutes=5)) -> TaskExecution:
        return await self._backend.renew(execution_id, owner=owner, fence=fence, duration=duration)

    async def complete(self, execution_id: str, *, owner: str, fence: int, result: object) -> TaskExecution:
        return await self._backend.complete(
            execution_id,
            owner=owner,
            fence=fence,
            result=result,
        )

    async def fail(
        self,
        execution_id: str,
        *,
        owner: str,
        fence: int,
        retry: bool = False,
        error: object = None,
    ) -> TaskExecution:
        return await self._backend.fail(
            execution_id,
            owner=owner,
            fence=fence,
            retry=retry,
            error=error,
        )


__all__ = ["TaskBackend", "TaskStore"]
