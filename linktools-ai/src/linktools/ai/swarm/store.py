#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmStore Protocol: persistence contract for SwarmRun/SwarmStep.
Two backends implement it: FilesystemSwarmStore (single-process) and
SqlAlchemySwarmStore (multi-process via atomic optimistic claim)."""

from typing import Any, Protocol, runtime_checkable

from ..run.models import RunErrorInfo, RunResult
from .models import SwarmRun, SwarmStatus, SwarmStep, SwarmStepAttempt, SwarmStepStatus


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

    async def create_task(self, task: SwarmStep) -> SwarmStep: ...

    async def claim_task(
        self, swarm_run_id: str, agent_id: str, *, lease_seconds: "float | None" = None
    ) -> "SwarmStep | None": ...

    async def set_active_run(
        self, task_id: str, run_id: str, *, expected_version: int
    ) -> SwarmStep:
        """Record the freshly-minted child RunRecord id on the task. Called by
        strategy._run_task immediately after a successful claim_task with the
        new uuid4 run_id it generated for this execution. Bumps the task
        version (optimistic concurrency on the claim's returned version). On
        retry the same task gets a NEW run_id here, so active_run_id always
        points at the most recent execution's child Run."""
        ...

    async def complete_task(
        self,
        task_id: str,
        result: RunResult,
        *,
        expected_version: int,
        active_run_id: "str | None" = None,
    ) -> SwarmStep:
        """Mark the task SUCCEEDED. ``expected_version`` is now a MANDATORY
        fencing token -- the CLAIMED
        task's version right after set_active_run -- so a worker whose lease
        already expired (and was reclaimed to a new owner) cannot clobber the
        new owner's progress with its own stale completion. The update is
        additionally conditioned on the task still being in CLAIMED status,
        and (when ``active_run_id`` is supplied) on it still matching the
        task's current ``active_run_id`` -- a second fencing dimension so a
        worker driving a since-superseded child Run cannot complete the task
        even if it somehow still held a matching version. There is no
        ``expected_version=None`` bypass."""
        ...

    async def fail_task(
        self,
        task_id: str,
        error: RunErrorInfo,
        *,
        expected_version: int,
        active_run_id: "str | None" = None,
    ) -> SwarmStep:
        """Mark the task FAILED (bumping ``attempts``). Same mandatory
        fencing-token semantics as :meth:`complete_task`."""
        ...

    async def list_tasks(
        self, swarm_run_id: str, *, status: "SwarmStepStatus | None" = None
    ) -> "tuple[SwarmStep, ...]": ...

    async def reclaim_expired_tasks(
        self, swarm_run_id: str
    ) -> "tuple[SwarmStep, ...]": ...

    # -- attempts ---------------------------------------------------------
    #
    # Each (re)execution of a SwarmStep records one SwarmStepAttempt for audit.
    # ``record_attempt`` is an upsert keyed on ``attempt.id``: the strategy
    # writes status=RUNNING/started_at before invoking the worker, then calls it
    # again with finished_at + SUCCEEDED|FAILED after the worker returns. One
    # attempt row per retry iteration so a 3-try retry leaves a 3-row trail.

    async def record_attempt(self, attempt: SwarmStepAttempt) -> SwarmStepAttempt: ...

    async def list_attempts(self, task_id: str) -> "tuple[SwarmStepAttempt, ...]": ...

    # -- lease renewal ----------------------------------------------------
    #
    # A worker that's still actively executing a long-running task must extend
    # its lease periodically; otherwise a concurrent reclaim/owner-change would
    # reset it. ``renew_lease`` only succeeds on CLAIMED tasks whose current
    # version matches ``expected_version`` (optimistic concurrency, same as
    # set_active_run); on success it pushes ``lease_expires_at`` out by
    # ``lease_seconds`` from now and bumps the task's version.

    async def renew_lease(
        self, task_id: str, *, expected_version: int, lease_seconds: float
    ) -> SwarmStep: ...
