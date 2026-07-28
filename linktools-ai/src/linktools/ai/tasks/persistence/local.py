"""Single-process TaskStore."""

from dataclasses import replace

from ..models import TaskExecution, TaskPlan


class LocalTaskStore:
    def __init__(self) -> None:
        self._plans: dict[str, TaskPlan] = {}
        self._executions: dict[str, TaskExecution] = {}

    async def save_plan(self, plan: TaskPlan) -> None:
        self._plans[plan.id] = plan

    async def get_plan(self, plan_id: str) -> TaskPlan | None:
        return self._plans.get(plan_id)

    async def get_execution(self, execution_id: str) -> TaskExecution | None:
        return self._executions.get(execution_id)

    async def add_execution(self, execution: TaskExecution) -> None:
        self._executions[execution.id] = execution

    create_execution = add_execution

    async def claim(self, execution_id: str, *, owner: str) -> TaskExecution:
        current = self._executions[execution_id]
        claimed = replace(current, status="claimed", owner=owner, fence=current.fence + 1, attempt=current.attempt + 1)
        self._executions[execution_id] = claimed
        return claimed

    async def renew(self, execution_id: str, *, owner: str, fence: int) -> TaskExecution:
        current = self._executions[execution_id]
        if current.owner != owner or current.fence != fence:
            raise ValueError("task fence conflict")
        return current

    async def complete(self, execution_id: str, *, owner: str, fence: int, result: object) -> TaskExecution:
        current = await self.renew(execution_id, owner=owner, fence=fence)
        completed = replace(current, status="completed", result=result)
        self._executions[execution_id] = completed
        return completed

    async def fail(self, execution_id: str, *, owner: str, fence: int, retry: bool = False) -> TaskExecution:
        current = await self.renew(execution_id, owner=owner, fence=fence)
        failed = replace(current, status="ready" if retry else "failed")
        self._executions[execution_id] = failed
        return failed
