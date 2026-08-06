#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Task Fence and retry coordination."""

from linktools.core import environ

from ...domain.task import TaskExecution, TaskPlan

logger = environ.get_logger("ai.application.services.task")


class TaskService:
    """Apply domain transitions through an injected TaskRepository."""

    def __init__(self, repository: object, query: "object | None" = None) -> None:
        self._repository = repository
        self._query = query or repository

    async def submit(self, task: TaskPlan) -> object:
        task.validate()
        logger.info("task plan submitted id=%s tasks=%s", task.plan_id, len(task.tasks))
        return await self._repository.submit(task)

    async def inspect(self, task_id: str) -> object:
        return await self._query.get(task_id)

    async def list(self, job_id: "str | None", limit: int, cursor: "str | None") -> object:
        return await self._query.list(job_id, min(limit, 200), cursor)

    async def claim(self, task_id: str, owner: str, now: object, duration: object) -> object:
        return await self._repository.claim(task_id, owner, now, duration)

    async def renew(self, execution: TaskExecution, owner: str, fence: int, now: object, duration: object) -> object:
        return await self._repository.renew(execution.task_id, owner, fence, now, duration)

    async def complete(self, execution: TaskExecution, owner: str, fence: int, now: object, result: object) -> object:
        return await self._repository.complete(execution.task_id, owner, fence, now, result)

    async def fail(self, execution: TaskExecution, owner: str, fence: int, now: object, error: str) -> object:
        return await self._repository.fail(execution.task_id, owner, fence, now, error)

    async def retry(self, task_id: str) -> object:
        return await self._repository.retry(task_id)

    async def cancel(self, task_id: str, reason: str) -> object:
        return await self._query.cancel(task_id, reason)


__all__ = ["TaskService"]
