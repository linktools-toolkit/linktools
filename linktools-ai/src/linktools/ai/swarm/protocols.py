#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TaskQueue Protocol: shared work queue N agent instances claim from.

`Task.depends_on` exists so `claim()` can enforce dependency ordering --
without a data field to check against, "claim() only returns tasks whose
dependencies are done" has no way to be implemented. See SwarmCoordinator
(swarm/coordinator.py) for how multiple agents are pointed at one shared
TaskQueue instance.
"""

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

TaskStatus = Literal["pending", "claimed", "done", "failed"]


@dataclass
class Task:
    task_id: str
    payload: Any
    status: TaskStatus = "pending"
    depends_on: "tuple[str, ...]" = ()
    claimed_by: "str | None" = None
    result: Any = None
    error: "str | None" = None


@runtime_checkable
class TaskQueue(Protocol):
    async def add(self, tasks: "list[Task]") -> None: ...

    async def claim(self, agent_id: str) -> "Task | None": ...

    async def complete(self, task_id: str, result: Any) -> None: ...

    async def fail(self, task_id: str, error: str) -> None: ...

    async def list(self, status: "TaskStatus | None" = None) -> "list[Task]": ...
