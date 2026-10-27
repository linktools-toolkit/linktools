#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ordered, retryable cleanup of resources owned by one Runtime."""

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable


class _RuntimeCloseCoordinator:
    def __init__(
        self, actions: tuple[Callable[[], Awaitable[None]], ...],
        handoff_actions: tuple[Callable[[], Awaitable[None]], ...] = (),
    ) -> None:
        self._actions = deque(actions)
        self._handoff_actions = handoff_actions
        self._cursor = 0
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def handoff(self) -> None:
        if self._task is None and self._cursor == 0:
            self._actions = deque(self._handoff_actions)
        await self.close()

    async def close(self) -> None:
        async with self._lock:
            self._handoff_actions = ()
            if not self._actions:
                return
            task = self._task
            if task is None or task.done():
                task = asyncio.create_task(self._run(), name="linktools-runtime-close")
                self._task = task
        await asyncio.shield(task)

    async def _run(self) -> None:
        while self._actions:
            await self._actions[0]()
            self._actions.popleft()
            self._cursor += 1


__all__ = []
