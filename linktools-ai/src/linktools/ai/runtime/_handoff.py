#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local tool quiescence during a Runtime shutdown handoff."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import TypeVar

T = TypeVar("T")


class ToolEntryStopped(asyncio.CancelledError):
    """The tool leaf was never entered."""


class ExecutionHandoffScope:
    def __init__(self) -> None:
        self.stopping = False
        self.cancelled = False
        self._tools: set[asyncio.Task[object]] = set()

    def require_accepting(self) -> None:
        if self.stopping:
            raise ToolEntryStopped

    def cancel_tools(self) -> None:
        self.cancelled = True
        for task in tuple(self._tools):
            task.cancel()

    async def call_tool(self, invoke: Callable[[], Awaitable[T]]) -> T:
        self.require_accepting()
        task = asyncio.create_task(invoke(), name="runtime-tool-settlement")
        self._tools.add(task)
        try:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if not self.stopping or self.cancelled:
                    task.cancel()
                # Once a leaf entered, its durable settlement must finish before
                # the Runtime can relinquish the local producer and its resources.
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        if not self.stopping or self.cancelled:
                            task.cancel()
                    except BaseException:
                        break
                if not task.cancelled():
                    task.result()
                raise
        finally:
            self._tools.discard(task)


execution_handoff_scope: ContextVar[ExecutionHandoffScope | None] = ContextVar(
    "execution_handoff_scope", default=None,
)

__all__ = []
