#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Structural task-store contract implemented directly by backends.

Every state transition is a single conditional update: the caller supplies the
owner and fence it read, the backend re-checks them in the WHERE clause, and a
rowcount that is not exactly one is a StorageConflictError. READY->CLAIMED uses
claim_ready; CLAIMED->{COMPLETED,FAILED,CANCELLED} and CLAIMED renewals all
verify owner+fence. skip/cancel_ready move READY nodes straight to a terminal
without a lease."""

from datetime import timedelta
from datetime import datetime
from typing import Protocol

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..storage.database import CoordinationScope
    from .models import TaskExecution, TaskPlan, TaskUsage
    from ..execution.domain import RunErrorInfo
    from ..json import JsonValue


class TaskStore(Protocol):
    coordination_scope: "CoordinationScope"

    async def create_plan(
        self,
        plan: "TaskPlan",
        executions: "tuple[TaskExecution, ...]",
    ) -> None:
        """Atomically insert ``plan`` and every node's initial READY execution.
        Any failure rolls the whole batch back; an existing plan_id is a conflict
        (never an upsert overwrite)."""
        ...

    async def get_plan(self, plan_id: str) -> "TaskPlan | None": ...

    async def list_executions(self, plan_id: str) -> "tuple[TaskExecution, ...]": ...

    async def get_execution(self, execution_id: str) -> "TaskExecution | None": ...

    async def claim_ready(
        self,
        execution_id: str,
        *,
        owner: str,
        duration: timedelta,
    ) -> "TaskExecution":
        """READY -> CLAIMED. Expired CLAIMED rows use the reconcile operation;
        this method never starts a node a second time."""
        ...

    async def bind_child_run(
        self,
        execution_id: str,
        *,
        owner: str,
        fence: int,
        child_run_id: str,
    ) -> "TaskExecution":
        """Record the deterministic child RunRecord id on a CLAIMED execution
        without leaving CLAIMED. Verifies owner+fence. Idempotent if the same
        child_run_id is re-bound; a different id is a conflict."""
        ...

    async def renew(
        self,
        execution_id: str,
        *,
        owner: str,
        fence: int,
        duration: timedelta,
    ) -> "TaskExecution":
        """Extend the lease on a CLAIMED execution, verifying owner+fence."""
        ...

    async def record_claimed_usage(
        self,
        execution_id: str,
        *,
        owner: str,
        fence: int,
        snapshot_revision: int,
        usage: "TaskUsage",
    ) -> "TaskExecution":
        """Persist monotonic usage without changing any task lifecycle field."""
        ...

    async def complete(
        self,
        execution_id: str,
        *,
        owner: str,
        fence: int,
        result: "JsonValue",
        snapshot_revision: int,
        usage: "TaskUsage",
    ) -> "TaskExecution":
        """CLAIMED -> COMPLETED with the child's structured result and usage.
        Releases the lease (owner retained for audit). Verifies owner+fence."""
        ...

    async def fail(
        self,
        execution_id: str,
        *,
        owner: str,
        fence: int,
        error: "RunErrorInfo",
        snapshot_revision: int,
        usage: "TaskUsage",
    ) -> "TaskExecution":
        """CLAIMED -> FAILED with a structured error and the real usage produced
        before the failure. Releases the lease. Verifies owner+fence."""
        ...

    async def skip(
        self,
        execution_id: str,
        *,
        blocked_by: "tuple[str, ...]",
        reason: str,
    ) -> "TaskExecution":
        """READY -> SKIPPED with the blocking node ids. No lease involved. If the
        node already left READY, the current state is returned unchanged only
        when it is already SKIPPED; any other status is a conflict."""
        ...

    async def cancel_ready(
        self,
        execution_id: str,
        *,
        reason: str,
    ) -> "TaskExecution":
        """READY -> CANCELLED with a terminal_reason. No lease involved. Terminal
        nodes are returned unchanged; CLAIMED nodes are a conflict (use
        cancel_claimed)."""
        ...

    async def cancel_claimed(
        self,
        execution_id: str,
        *,
        owner: str,
        fence: int,
        reason: str,
        snapshot_revision: int,
        usage: "TaskUsage",
    ) -> "TaskExecution":
        """CLAIMED -> CANCELLED with the reason and the real usage produced so
        far. Releases the lease. Verifies owner+fence."""
        ...

    async def take_over_expired_claim_for_reconcile(
        self,
        execution_id: str,
        *,
        owner: str,
        now: datetime,
        duration: timedelta,
    ) -> "TaskExecution":
        """Take over only an expired CLAIMED execution for child reconciliation.
        The attempt and active child id are retained; the caller must not launch
        a new child after this operation."""
        ...


__all__ = ["TaskStore"]
