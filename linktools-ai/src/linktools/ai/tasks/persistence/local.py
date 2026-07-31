#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Single-process TaskStore with shared lease semantics."""


from dataclasses import replace
from datetime import datetime, timedelta, timezone
from ...storage.coordination.lease import Lease, assert_active, claim, renew
from ...storage.database import CoordinationScope
from ...errors import StorageConflictError
from ..models import TaskStatus

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import TaskExecution, TaskPlan

class LocalTaskBackend:
    coordination_scope = CoordinationScope.PROCESS

    async def initialize_storage(self) -> None:
        return None

    def __init__(self) -> None:
        self._plans: "dict[str, TaskPlan]" = {}
        self._executions: "dict[str, TaskExecution]" = {}

    async def save_plan(self, plan: "TaskPlan") -> None:
        self._plans[plan.id] = plan

    async def get_plan(self, plan_id: str) -> "TaskPlan | None":
        return self._plans.get(plan_id)

    async def get_execution(self, execution_id: str) -> "TaskExecution | None":
        return self._executions.get(execution_id)

    async def add_execution(self, execution: "TaskExecution") -> None:
        self._executions[execution.id] = execution

    create_execution = add_execution

    async def claim(self, execution_id: str, *, owner: str, duration: timedelta = timedelta(minutes=5)) -> "TaskExecution":
        current = self._executions[execution_id]
        now = datetime.now(timezone.utc)
        if current.status is TaskStatus.COMPLETED:
            return current
        lease = claim(current.lease, owner=owner, now=now, duration=duration)
        updated = replace(current, status=TaskStatus.CLAIMED, lease=lease, attempt=current.attempt + 1, updated_at=now)
        self._executions[execution_id] = updated
        return updated

    async def renew(self, execution_id: str, *, owner: str, fence: int, duration: timedelta = timedelta(minutes=5)) -> "TaskExecution":
        current = self._executions[execution_id]
        now = datetime.now(timezone.utc)
        updated = replace(current, lease=renew(current.lease, owner=owner, fence=fence, now=now, duration=duration), updated_at=now)
        self._executions[execution_id] = updated
        return updated

    async def complete(self, execution_id: str, *, owner: str, fence: int, result: object) -> "TaskExecution":
        current = await self.renew(execution_id, owner=owner, fence=fence)
        now = datetime.now(timezone.utc)
        updated = replace(current, status=TaskStatus.COMPLETED, result=result, lease=Lease(current.lease.owner, current.lease.fence, None), updated_at=now)
        self._executions[execution_id] = updated
        return updated

    async def fail(self, execution_id: str, *, owner: str, fence: int, retry: bool = False, error: object = None) -> "TaskExecution":
        current = await self.renew(execution_id, owner=owner, fence=fence)
        now = datetime.now(timezone.utc)
        updated = replace(current, status=TaskStatus.READY if retry else TaskStatus.FAILED, error=error, lease=Lease(current.lease.owner, current.lease.fence, None), updated_at=now)
        self._executions[execution_id] = updated
        return updated


__all__ = ["LocalTaskBackend"]
