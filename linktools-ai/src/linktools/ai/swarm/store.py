#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmStore Protocol: persistence contract for SwarmRun/SwarmTask.
Method signatures are this phase's concrete resolution of the spec's `(...)`
ellipses (section 22). Two backends implement it: FileSwarmStore (single-process)
and SqlAlchemySwarmStore (multi-process via atomic optimistic claim)."""

from typing import Any, Protocol, runtime_checkable

from ..run.models import RunErrorInfo, RunResult
from .models import SwarmRun, SwarmStatus, SwarmTask, SwarmTaskAttempt, SwarmTaskStatus


@runtime_checkable
class SwarmStore(Protocol):
    async def create_run(self, run: SwarmRun) -> SwarmRun: ...

    async def get_run(self, swarm_run_id: str) -> "SwarmRun | None": ...

    async def update_run(
        self,
        swarm_run_id: str,
        *,
        expected_version: int,
        status: "SwarmStatus | None" = None,
        round: "int | None" = None,
        token_usage: "Any | None" = None,
        cost: "Any | None" = None,
        metadata: "dict | None" = None,
    ) -> SwarmRun: ...

    async def create_task(self, task: SwarmTask) -> SwarmTask: ...

    async def claim_task(
        self, swarm_run_id: str, agent_id: str, *, lease_seconds: "float | None" = None
    ) -> "SwarmTask | None": ...

    async def set_active_run(
        self, task_id: str, run_id: str, *, expected_version: int
    ) -> SwarmTask:
        """Record the freshly-minted child RunRecord id on the task. Called by
        strategy._run_task immediately after a successful claim_task with the
        new uuid4 run_id it generated for this execution. Bumps the task
        version (optimistic concurrency on the claim's returned version). On
        retry the same task gets a NEW run_id here, so active_run_id always
        points at the most recent execution's child Run."""
        ...

    async def complete_task(self, task_id: str, result: RunResult) -> SwarmTask: ...

    async def fail_task(self, task_id: str, error: RunErrorInfo) -> SwarmTask: ...

    async def list_tasks(
        self, swarm_run_id: str, *, status: "SwarmTaskStatus | None" = None
    ) -> "tuple[SwarmTask, ...]": ...

    async def reclaim_expired_tasks(self, swarm_run_id: str) -> "tuple[SwarmTask, ...]": ...

    # -- attempts (review doc §19.2) --------------------------------------
    #
    # Each (re)execution of a SwarmTask records one SwarmTaskAttempt for audit.
    # ``record_attempt`` is an upsert keyed on ``attempt.id``: the strategy
    # writes status=RUNNING/started_at before invoking the worker, then calls it
    # again with finished_at + SUCCEEDED|FAILED after the worker returns. One
    # attempt row per retry iteration so a 3-try retry leaves a 3-row trail.

    async def record_attempt(self, attempt: SwarmTaskAttempt) -> SwarmTaskAttempt: ...

    async def list_attempts(self, task_id: str) -> "tuple[SwarmTaskAttempt, ...]": ...

    # -- lease renewal (review doc §19.4) ---------------------------------
    #
    # A worker that's still actively executing a long-running task must extend
    # its lease periodically; otherwise a concurrent reclaim/owner-change would
    # reset it. ``renew_lease`` only succeeds on CLAIMED tasks whose current
    # version matches ``expected_version`` (optimistic concurrency, same as
    # set_active_run); on success it pushes ``lease_expires_at`` out by
    # ``lease_seconds`` from now and bumps the task's version.

    async def renew_lease(
        self, task_id: str, *, expected_version: int, lease_seconds: float
    ) -> SwarmTask: ...
