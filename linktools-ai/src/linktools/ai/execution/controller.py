"""In-process ownership and cancellation of active executions."""

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any

from ..errors import ExecutionAlreadyActiveError
from .cancellation import CancellationToken


@dataclass(slots=True)
class ActiveExecution:
    task: asyncio.Task
    token: CancellationToken


class ExecutionControllerRegistry:
    def __init__(self) -> None:
        self._active: dict[str, ActiveExecution] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        execution_id: str,
        active: ActiveExecution,
    ) -> None:
        async with self._lock:
            previous = self._active.get(execution_id)
            if previous is not None and not previous.task.done():
                raise ExecutionAlreadyActiveError(
                    f"execution {execution_id!r} is already active"
                )
            self._active[execution_id] = active

    async def start(
        self,
        execution_id: str,
        coroutine: Coroutine[Any, Any, object],
        token: CancellationToken,
    ) -> asyncio.Task:
        """Atomically reject duplicates before scheduling the new execution."""
        async with self._lock:
            previous = self._active.get(execution_id)
            if previous is not None and not previous.task.done():
                coroutine.close()
                raise ExecutionAlreadyActiveError(
                    f"execution {execution_id!r} is already active"
                )
            task = asyncio.create_task(coroutine)
            self._active[execution_id] = ActiveExecution(task, token)
            return task

    async def cancel(self, execution_id: str) -> bool:
        async with self._lock:
            active = self._active.get(execution_id)
        if active is None:
            return False
        active.token.cancel()
        if not active.task.done():
            active.task.cancel()
        return True

    async def unregister(
        self,
        execution_id: str,
        *,
        task: asyncio.Task,
    ) -> None:
        async with self._lock:
            active = self._active.get(execution_id)
            if active is not None and active.task is task:
                self._active.pop(execution_id, None)

    def get_token(self, execution_id: str) -> CancellationToken | None:
        active = self._active.get(execution_id)
        return active.token if active is not None else None
