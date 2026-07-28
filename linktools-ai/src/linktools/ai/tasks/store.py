"""Small task protocols with one claim/lease/attempt state machine."""

from typing import Protocol

from .models import TaskExecution, TaskNode, TaskPlan


class TaskReader(Protocol):
    async def get_plan(self, plan_id: str) -> TaskPlan | None: ...
    async def get_execution(self, execution_id: str) -> TaskExecution | None: ...


class TaskPlanner(Protocol):
    async def save_plan(self, plan: TaskPlan) -> None: ...
    async def add_execution(self, execution: TaskExecution) -> None: ...


class TaskClaimer(Protocol):
    async def claim(self, execution_id: str, *, owner: str) -> TaskExecution: ...
    async def renew(self, execution_id: str, *, owner: str, fence: int) -> TaskExecution: ...


class TaskCommitter(Protocol):
    async def complete(self, execution_id: str, *, owner: str, fence: int, result: object) -> TaskExecution: ...
    async def fail(self, execution_id: str, *, owner: str, fence: int, retry: bool = False) -> TaskExecution: ...


class TaskPort(TaskReader, TaskPlanner, TaskClaimer, TaskCommitter, Protocol):
    pass


class TaskStore:
    def __init__(self, backend: TaskPort) -> None:
        self.backend = backend

    def __getattr__(self, name: str):
        return getattr(self.backend, name)
