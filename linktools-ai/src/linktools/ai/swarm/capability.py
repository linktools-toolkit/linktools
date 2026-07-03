#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmCapability: claim_task/complete_task/fail_task/list_tasks tools over
an injected shared TaskQueue. Mirrors SubagentCapability's shape -- an
AbstractCapability whose get_toolset() forwards to methods on an injected
collaborator, not a place for new orchestration logic."""

from dataclasses import dataclass
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from .protocols import TaskQueue


def _task_to_dict(task: Any) -> "dict[str, Any]":
    return {
        "task_id": task.task_id,
        "payload": task.payload,
        "status": task.status,
        "depends_on": list(task.depends_on),
        "claimed_by": task.claimed_by,
        "result": task.result,
        "error": task.error,
    }


@dataclass
class SwarmCapability(AbstractCapability[None]):
    task_queue: TaskQueue
    agent_id: str

    def get_toolset(self) -> FunctionToolset:
        toolset: FunctionToolset = FunctionToolset()
        task_queue = self.task_queue
        agent_id = self.agent_id

        async def claim_task() -> "dict[str, Any]":
            """Claim the next available task from the shared swarm queue, or {"task_id": None} if none are ready."""
            task = await task_queue.claim(agent_id)
            if task is None:
                return {"task_id": None}
            return _task_to_dict(task)

        async def complete_task(task_id: str, result: Any) -> "dict[str, Any]":
            """Mark a claimed task done with its result."""
            await task_queue.complete(task_id, result)
            return {"ok": True, "task_id": task_id}

        async def fail_task(task_id: str, error: str) -> "dict[str, Any]":
            """Mark a claimed task failed with an error message."""
            await task_queue.fail(task_id, error)
            return {"ok": True, "task_id": task_id}

        async def list_tasks(status: "str | None" = None) -> "list[dict[str, Any]]":
            """List tasks in the shared swarm queue, optionally filtered by status."""
            tasks = await task_queue.list(status)
            return [_task_to_dict(t) for t in tasks]

        for fn in (claim_task, complete_task, fail_task, list_tasks):
            toolset.add_function(fn)
        return toolset
