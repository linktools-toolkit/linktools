#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmStore Protocol: persistence contract for SwarmRun/SwarmTask.
Method signatures are this phase's concrete resolution of the spec's `(...)`
ellipses (section 22). Two backends implement it: FileSwarmStore (single-process)
and SqlAlchemySwarmStore (multi-process via atomic optimistic claim)."""

from typing import Any, Protocol, runtime_checkable

from ..run.models import RunErrorInfo, RunResult
from .models import SwarmRun, SwarmStatus, SwarmTask, SwarmTaskStatus


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
