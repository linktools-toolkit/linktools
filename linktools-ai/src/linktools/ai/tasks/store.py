"""Task backend contract and composed Store."""

from datetime import timedelta
from typing import Protocol

from ..storage.composition import StorageComposition
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
        self._storage = StorageComposition(primary=backend)

    @property
    def backend(self) -> TaskBackend:
        return self._storage.primary

    async def initialize_storage(self, *args: object) -> None:
        await self._storage.initialize(*args)

    async def get_plan(self, plan_id: str) -> TaskPlan | None:
        return await self._storage.primary.get_plan(plan_id)

    async def get_execution(self, execution_id: str) -> TaskExecution | None:
        return await self._storage.primary.get_execution(execution_id)

    async def save_plan(self, plan: TaskPlan) -> None:
        await self._storage.primary.save_plan(plan)

    async def add_execution(self, execution: TaskExecution) -> None:
        await self._storage.primary.add_execution(execution)

    async def claim(self, execution_id: str, *, owner: str, duration: timedelta = timedelta(minutes=5)) -> TaskExecution:
        return await self._storage.primary.claim(execution_id, owner=owner, duration=duration)

    async def renew(self, execution_id: str, *, owner: str, fence: int, duration: timedelta = timedelta(minutes=5)) -> TaskExecution:
        return await self._storage.primary.renew(execution_id, owner=owner, fence=fence, duration=duration)

    async def complete(self, execution_id: str, *, owner: str, fence: int, result: object) -> TaskExecution:
        return await self._storage.primary.complete(
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
        return await self._storage.primary.fail(
            execution_id,
            owner=owner,
            fence=fence,
            retry=retry,
            error=error,
        )


__all__ = ["TaskBackend", "TaskStore"]
