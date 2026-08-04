#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Single-process TaskStore with shared lease semantics.

Every read-modify-write over the execution dict holds the same asyncio.Lock, so
the lease helpers (pure functions) can mutate-in-place safely. Multi-process
deployments must use the shared-database backend instead."""

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import MappingProxyType

from ...errors import StorageConflictError
from ...storage.coordination.lease import Lease, claim, is_expired, renew
from ...storage.database import CoordinationScope
from ..models import TaskExecution, TaskPlan, TaskStatus, TaskUsage

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...execution.domain import RunErrorInfo
    from ...json import JsonValue


class LocalTaskBackend:
    coordination_scope = CoordinationScope.PROCESS

    def __init__(self) -> None:
        self._plans: "dict[str, TaskPlan]" = {}
        self._executions: "dict[str, TaskExecution]" = {}
        self._lock = asyncio.Lock()

    async def initialize_storage(self) -> None:
        return None

    async def create_plan(
        self,
        plan: TaskPlan,
        executions: "tuple[TaskExecution, ...]",
    ) -> None:
        async with self._lock:
            if plan.id in self._plans:
                raise StorageConflictError(f"task plan {plan.id!r} already exists")
            seen: "set[str]" = set()
            for execution in executions:
                if execution.id in seen:
                    raise StorageConflictError(
                        f"duplicate execution id {execution.id!r} in batch"
                    )
                if execution.id in self._executions:
                    raise StorageConflictError(
                        f"task execution {execution.id!r} already exists"
                    )
                seen.add(execution.id)
            self._plans[plan.id] = plan
            for execution in executions:
                self._executions[execution.id] = execution

    async def get_plan(self, plan_id: str) -> "TaskPlan | None":
        async with self._lock:
            return self._plans.get(plan_id)

    async def list_executions(self, plan_id: str) -> "tuple[TaskExecution, ...]":
        async with self._lock:
            rows = [e for e in self._executions.values() if e.plan_id == plan_id]
            rows.sort(key=lambda e: e.created_at)
            return tuple(rows)

    async def get_execution(self, execution_id: str) -> "TaskExecution | None":
        async with self._lock:
            return self._executions.get(execution_id)

    async def claim_ready(
        self,
        execution_id: str,
        *,
        owner: str,
        duration: timedelta,
    ) -> TaskExecution:
        async with self._lock:
            current = self._require(execution_id)
            now = datetime.now(timezone.utc)
            if current.status is not TaskStatus.READY:
                raise StorageConflictError(
                    f"task {execution_id!r} is {current.status.value}, not claimable"
                )
            new_lease = claim(current.lease, owner=owner, now=now, duration=duration)
            updated = replace(
                current,
                status=TaskStatus.CLAIMED,
                lease=new_lease,
                attempt=1,
                updated_at=now,
            )
            self._executions[execution_id] = updated
            return updated

    async def take_over_expired_claim_for_reconcile(
        self,
        execution_id: str,
        *,
        owner: str,
        now: datetime,
        duration: timedelta,
    ) -> TaskExecution:
        async with self._lock:
            current = self._require(execution_id)
            if current.status is not TaskStatus.CLAIMED or not is_expired(
                current.lease, now
            ):
                raise StorageConflictError("task is not an expired CLAIMED execution")
            updated = replace(
                current,
                lease=claim(current.lease, owner=owner, now=now, duration=duration),
                updated_at=now,
            )
            self._executions[execution_id] = updated
            return updated

    async def bind_child_run(
        self,
        execution_id: str,
        *,
        owner: str,
        fence: int,
        child_run_id: str,
    ) -> TaskExecution:
        async with self._lock:
            current = self._require(execution_id)
            self._assert_claimed(current, owner, fence)
            if current.active_run_id is not None and current.active_run_id != child_run_id:
                raise StorageConflictError(
                    f"task {execution_id!r} already bound to a different child run"
                )
            now = datetime.now(timezone.utc)
            updated = replace(
                current,
                active_run_id=child_run_id,
                updated_at=now,
            )
            self._executions[execution_id] = updated
            return updated

    async def renew(
        self,
        execution_id: str,
        *,
        owner: str,
        fence: int,
        duration: timedelta,
    ) -> TaskExecution:
        async with self._lock:
            current = self._require(execution_id)
            self._assert_claimed(current, owner, fence)
            now = datetime.now(timezone.utc)
            new_lease = renew(
                current.lease, owner=owner, fence=fence, now=now, duration=duration
            )
            updated = replace(current, lease=new_lease, updated_at=now)
            self._executions[execution_id] = updated
            return updated

    async def complete(
        self,
        execution_id: str,
        *,
        owner: str,
        fence: int,
        result: "JsonValue",
        usage: TaskUsage,
    ) -> TaskExecution:
        async with self._lock:
            current = self._require(execution_id)
            self._assert_claimed(current, owner, fence)
            now = datetime.now(timezone.utc)
            updated = replace(
                current,
                status=TaskStatus.COMPLETED,
                result=_freeze(result),
                usage=usage,
                lease=Lease(current.lease.owner, current.lease.fence, None),
                updated_at=now,
            )
            self._executions[execution_id] = updated
            return updated

    async def fail(
        self,
        execution_id: str,
        *,
        owner: str,
        fence: int,
        error: "RunErrorInfo",
        usage: TaskUsage,
    ) -> TaskExecution:
        async with self._lock:
            current = self._require(execution_id)
            self._assert_claimed(current, owner, fence)
            now = datetime.now(timezone.utc)
            updated = replace(
                current,
                status=TaskStatus.FAILED,
                error=error,
                usage=usage,
                lease=Lease(current.lease.owner, current.lease.fence, None),
                updated_at=now,
            )
            self._executions[execution_id] = updated
            return updated

    async def skip(
        self,
        execution_id: str,
        *,
        blocked_by: "tuple[str, ...]",
        reason: str,
    ) -> TaskExecution:
        async with self._lock:
            current = self._require(execution_id)
            if current.status is TaskStatus.SKIPPED:
                return current
            if current.status is not TaskStatus.READY:
                raise StorageConflictError(
                    f"task {execution_id!r} is {current.status.value}, cannot skip"
                )
            now = datetime.now(timezone.utc)
            updated = replace(
                current,
                status=TaskStatus.SKIPPED,
                blocked_by=blocked_by,
                terminal_reason=reason,
                updated_at=now,
            )
            self._executions[execution_id] = updated
            return updated

    async def cancel_ready(
        self,
        execution_id: str,
        *,
        reason: str,
    ) -> TaskExecution:
        async with self._lock:
            current = self._require(execution_id)
            if current.status in (TaskStatus.CANCELLED, TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED):
                return current
            if current.status is not TaskStatus.READY:
                raise StorageConflictError(
                    f"task {execution_id!r} is {current.status.value}, use cancel_claimed"
                )
            now = datetime.now(timezone.utc)
            updated = replace(
                current,
                status=TaskStatus.CANCELLED,
                terminal_reason=reason,
                updated_at=now,
            )
            self._executions[execution_id] = updated
            return updated

    async def cancel_claimed(
        self,
        execution_id: str,
        *,
        owner: str,
        fence: int,
        reason: str,
        usage: TaskUsage,
    ) -> TaskExecution:
        async with self._lock:
            current = self._require(execution_id)
            self._assert_claimed(current, owner, fence)
            now = datetime.now(timezone.utc)
            updated = replace(
                current,
                status=TaskStatus.CANCELLED,
                terminal_reason=reason,
                usage=usage,
                lease=Lease(current.lease.owner, current.lease.fence, None),
                updated_at=now,
            )
            self._executions[execution_id] = updated
            return updated

    def _require(self, execution_id: str) -> TaskExecution:
        current = self._executions.get(execution_id)
        if current is None:
            raise StorageConflictError(f"task execution {execution_id!r} not found")
        return current

    def _assert_claimed(
        self, execution: TaskExecution, owner: str, fence: int
    ) -> None:
        if execution.status is not TaskStatus.CLAIMED:
            raise StorageConflictError(
                f"task {execution.id!r} is {execution.status.value}, not claimed"
            )
        if execution.lease.owner != owner or execution.lease.fence != fence:
            raise StorageConflictError("stale fence for task execution")
        if execution.lease.expires_at is None or execution.lease.expires_at <= datetime.now(timezone.utc):
            raise StorageConflictError("task execution lease expired")


def _freeze(result: "JsonValue") -> "JsonValue":
    if isinstance(result, dict):
        return MappingProxyType({k: _freeze(v) for k, v in result.items()})
    if isinstance(result, list):
        return tuple(_freeze(v) for v in result)
    return result


__all__ = ["LocalTaskBackend"]
