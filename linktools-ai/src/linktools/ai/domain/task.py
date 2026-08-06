#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Task DAG, fencing and aggregate invariants."""

from datetime import datetime, timedelta, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ..foundation.errors import ErrorCode, LinktoolsAIError


class TaskStatus(StrEnum):
    READY = "READY"
    CLAIMED = "CLAIMED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RetryPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_attempts: int = Field(default=1, ge=1)
    backoff_seconds: float = Field(default=0, ge=0)


class TaskNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    dependencies: "tuple[str, ...]" = ()
    required: bool = True
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)


class TaskPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_id: str
    tasks: "tuple[TaskNode, ...]"

    def validate(self) -> None:
        ids = [task.task_id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise LinktoolsAIError(ErrorCode.TASK_DAG_INVALID, "task ids must be unique")
        known = set(ids)
        if any(dependency not in known for task in self.tasks for dependency in task.dependencies):
            raise LinktoolsAIError(ErrorCode.TASK_DAG_INVALID, "task dependency is unknown")
        state: "dict[str, int]" = {}
        graph = {task.task_id: task.dependencies for task in self.tasks}

        def visit(task_id: str) -> None:
            if state.get(task_id) == 1:
                raise LinktoolsAIError(ErrorCode.TASK_DAG_INVALID, "task graph contains a cycle")
            if state.get(task_id) == 2:
                return
            state[task_id] = 1
            for dependency in graph[task_id]:
                visit(dependency)
            state[task_id] = 2

        for task_id in ids:
            visit(task_id)

    def ready(self, completed: "frozenset[str]") -> "tuple[str, ...]":
        self.validate()
        return tuple(sorted(task.task_id for task in self.tasks if task.task_id not in completed and set(task.dependencies) <= completed))


class TaskExecution(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    status: TaskStatus = TaskStatus.READY
    attempt: int = Field(default=0, ge=0)
    fence: int = Field(default=0, ge=0)
    lease_owner: "str | None" = None
    lease_expires_at: "datetime | None" = None
    result_ref: "str | None" = None
    error: "str | None" = None
    next_ready_at: "datetime | None" = None

    def claim(self, owner: str, now: datetime, duration: timedelta) -> "TaskExecution":
        if self.status == TaskStatus.COMPLETED:
            return self
        if self.next_ready_at is not None:
            ready_at = self.next_ready_at if self.next_ready_at.tzinfo else self.next_ready_at.replace(tzinfo=timezone.utc)
            current = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
            if ready_at > current:
                raise LinktoolsAIError(ErrorCode.TASK_NOT_READY, "task retry backoff has not elapsed")
        if self.status == TaskStatus.CLAIMED and self._active(now):
            raise LinktoolsAIError(ErrorCode.TASK_NOT_READY, "task already claimed")
        if self.status not in {TaskStatus.READY, TaskStatus.FAILED, TaskStatus.CLAIMED}:
            raise LinktoolsAIError(ErrorCode.TASK_NOT_READY, "task is not claimable")
        return self.model_copy(update={"status": TaskStatus.CLAIMED, "attempt": self.attempt + 1, "fence": self.fence + 1, "lease_owner": owner, "lease_expires_at": now + duration, "next_ready_at": None})

    def renew(self, owner: str, fence: int, now: datetime, duration: timedelta) -> "TaskExecution":
        self._assert_fence(owner, fence, now)
        return self.model_copy(update={"lease_expires_at": now + duration})

    def complete(self, owner: str, fence: int, now: datetime, result_ref: "str | None" = None) -> "TaskExecution":
        if self.status == TaskStatus.COMPLETED:
            return self
        self._assert_fence(owner, fence, now)
        return self.model_copy(update={"status": TaskStatus.COMPLETED, "result_ref": result_ref, "lease_owner": None, "lease_expires_at": None})

    def fail(self, owner: str, fence: int, now: datetime, error: str) -> "TaskExecution":
        if self.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
            return self
        self._assert_fence(owner, fence, now)
        return self.model_copy(update={"status": TaskStatus.FAILED, "error": error, "lease_owner": None, "lease_expires_at": None})

    def recover(self, now: datetime) -> "TaskExecution":
        if self.status != TaskStatus.CLAIMED or self._active(now):
            return self
        return self.model_copy(update={"status": TaskStatus.READY, "lease_owner": None, "lease_expires_at": None})

    def retry(self, now: datetime, max_attempts: int, backoff_seconds: float = 0) -> "TaskExecution":
        """Return a failed task to READY when its retry policy permits it."""
        if self.status is TaskStatus.COMPLETED:
            return self
        if self.status is not TaskStatus.FAILED:
            raise LinktoolsAIError(ErrorCode.TASK_NOT_READY, "only failed tasks can be retried")
        if max_attempts < 1 or self.attempt >= max_attempts:
            return self
        return self.model_copy(
            update={
                "status": TaskStatus.READY,
                "lease_owner": None,
                "lease_expires_at": None,
                "next_ready_at": now + timedelta(seconds=max(0, backoff_seconds)),
            }
        )

    def _active(self, now: datetime) -> bool:
        if self.lease_expires_at is None:
            return False
        expiry = self.lease_expires_at if self.lease_expires_at.tzinfo else self.lease_expires_at.replace(tzinfo=timezone.utc)
        current = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        return expiry > current

    def _assert_fence(self, owner: str, fence: int, now: datetime) -> None:
        if self.status != TaskStatus.CLAIMED or self.lease_owner != owner or self.fence != fence or not self._active(now):
            raise LinktoolsAIError(ErrorCode.TASK_FENCE_STALE, "task fencing token is stale")


class Job(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    plan: TaskPlan
    executions: "tuple[TaskExecution, ...]" = ()

    def aggregate(self) -> str:
        statuses = {execution.task_id: execution.status for execution in self.executions}
        required = {task.task_id for task in self.plan.tasks if task.required}
        completed = {task_id for task_id, status in statuses.items() if status is TaskStatus.COMPLETED}
        failed_required = {
            task_id
            for task_id, status in statuses.items()
            if status is TaskStatus.FAILED and task_id in required
        }
        if required <= completed:
            return "SUCCEEDED"
        if failed_required:
            return "FAILED"
        return "RUNNING"


class Swarm(BaseModel):
    model_config = ConfigDict(frozen=True)

    swarm_id: str
    node_ids: "tuple[str, ...]"
    max_concurrency: int = Field(gt=0)
    max_depth: int = Field(gt=0)
    max_nodes: int = Field(gt=0)

    def validate(self) -> None:
        if len(self.node_ids) > self.max_nodes or len(set(self.node_ids)) != len(self.node_ids):
            raise LinktoolsAIError(ErrorCode.TASK_DAG_INVALID, "swarm node limit or uniqueness violated")

    def aggregate(self, results: "dict[str, object]") -> "tuple[object | None, ...]":
        self.validate()
        return tuple(results.get(node_id) for node_id in self.node_ids)


__all__ = ["Job", "RetryPolicy", "Swarm", "TaskExecution", "TaskNode", "TaskPlan", "TaskStatus"]
