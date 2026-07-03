#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local TaskQueue implementations: in-process and file-backed."""

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .protocols import Task, TaskStatus


def _claimable(task: Task, all_tasks: "dict[str, Task]") -> bool:
    if task.status != "pending":
        return False
    return all(
        dep in all_tasks and all_tasks[dep].status == "done"
        for dep in task.depends_on
    )


class InMemoryTaskQueue:
    def __init__(self) -> None:
        self._tasks: "dict[str, Task]" = {}
        self._lock = asyncio.Lock()

    async def add(self, tasks: "list[Task]") -> None:
        async with self._lock:
            for task in tasks:
                self._tasks[task.task_id] = task

    async def claim(self, agent_id: str) -> "Task | None":
        async with self._lock:
            for task in self._tasks.values():
                if _claimable(task, self._tasks):
                    task.status = "claimed"
                    task.claimed_by = agent_id
                    return task
            return None

    async def complete(self, task_id: str, result: Any) -> None:
        async with self._lock:
            task = self._tasks[task_id]
            task.status = "done"
            task.result = result

    async def fail(self, task_id: str, error: str) -> None:
        async with self._lock:
            task = self._tasks[task_id]
            task.status = "failed"
            task.error = error

    async def list(self, status: "TaskStatus | None" = None) -> "list[Task]":
        async with self._lock:
            return [t for t in self._tasks.values() if status is None or t.status == status]


class FileTaskQueue:
    """File-backed TaskQueue: the full task list round-trips through
    `root/tasks.json` on every mutating call. The `asyncio.Lock` serializes
    claim() within this process -- it does NOT make cross-process claims
    atomic, since plain file writes aren't atomic; a cross-process-safe
    implementation is future work (see AgentArtifactStore's own local-only
    scope for the same caveat)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._path = root / "tasks.json"
        self._lock = asyncio.Lock()

    def _read(self) -> "dict[str, Task]":
        if not self._path.exists():
            return {}
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        return {item["task_id"]: Task(**item) for item in raw}

    def _write(self, tasks: "dict[str, Task]") -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = [asdict(task) for task in tasks.values()]
        self._path.write_text(json.dumps(payload), encoding="utf-8")

    async def add(self, tasks: "list[Task]") -> None:
        async with self._lock:
            current = await asyncio.to_thread(self._read)
            for task in tasks:
                current[task.task_id] = task
            await asyncio.to_thread(self._write, current)

    async def claim(self, agent_id: str) -> "Task | None":
        async with self._lock:
            current = await asyncio.to_thread(self._read)
            for task in current.values():
                if _claimable(task, current):
                    task.status = "claimed"
                    task.claimed_by = agent_id
                    await asyncio.to_thread(self._write, current)
                    return task
            return None

    async def complete(self, task_id: str, result: Any) -> None:
        async with self._lock:
            current = await asyncio.to_thread(self._read)
            current[task_id].status = "done"
            current[task_id].result = result
            await asyncio.to_thread(self._write, current)

    async def fail(self, task_id: str, error: str) -> None:
        async with self._lock:
            current = await asyncio.to_thread(self._read)
            current[task_id].status = "failed"
            current[task_id].error = error
            await asyncio.to_thread(self._write, current)

    async def list(self, status: "TaskStatus | None" = None) -> "list[Task]":
        async with self._lock:
            current = await asyncio.to_thread(self._read)
            return [t for t in current.values() if status is None or t.status == status]
