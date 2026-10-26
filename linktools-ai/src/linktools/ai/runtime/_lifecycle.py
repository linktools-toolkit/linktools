#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ordered, retryable cleanup of resources owned by one Runtime."""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import Generic, Protocol, TypeVar


class _Closeable(Protocol):
    async def close(self) -> None: ...


_ResourceT = TypeVar("_ResourceT", bound=_Closeable)


class _OpenedDomain(Generic[_ResourceT]):
    """Retain an entered domain's manager without using it as its retry handle."""

    def __init__(
        self, resource: _ResourceT, manager: AbstractAsyncContextManager[_ResourceT],
    ) -> None:
        self.resource = resource
        self._manager = manager
        self._exited = False

    async def close(self) -> None:
        if self._exited:
            return
        await self.resource.close()
        await self._manager.__aexit__(None, None, None)
        self._exited = True


class _RuntimeCloseCoordinator:
    def __init__(
        self, actions: tuple[Callable[[], Awaitable[None]], ...],
        handoff_actions: tuple[Callable[[], Awaitable[None]], ...] = (),
    ) -> None:
        self._actions = actions
        self._handoff_actions = handoff_actions
        self._cursor = 0
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def handoff(self) -> None:
        if self._task is None and self._cursor == 0:
            self._actions = self._handoff_actions
        await self.close()

    async def close(self) -> None:
        async with self._lock:
            if self._cursor >= len(self._actions):
                return
            task = self._task
            if task is None or task.done():
                task = asyncio.create_task(self._run(), name="linktools-runtime-close")
                self._task = task
        await asyncio.shield(task)

    async def _run(self) -> None:
        while self._cursor < len(self._actions):
            await self._actions[self._cursor]()
            self._cursor += 1


__all__ = []
