#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Owned MCP resources for one ACP session."""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .errors import request_error
from .mcp import mcp_spec
from .task_utils import cancel_and_wait, wait_and_observe


logger = logging.getLogger("linktools.ai.acp.mcp_resources")


class McpResourceState(str, Enum):
    NEW = "new"
    CONNECTING = "connecting"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass(slots=True)
class SessionMcpResources:
    descriptors: "tuple[Any, ...]" = ()
    state: McpResourceState = McpResourceState.NEW
    lock: "asyncio.Lock" = field(default_factory=asyncio.Lock)
    connect_task: "asyncio.Task[tuple[Any, ...]] | None" = None
    close_task: "asyncio.Task[None] | None" = None
    pool: Any = None
    _toolsets: "tuple[Any, ...] | None" = None

    async def toolsets(self) -> "tuple[Any, ...]":
        async with self.lock:
            if self.state is McpResourceState.CLOSED:
                raise request_error("session_closed")
            if self.state is McpResourceState.CLOSING:
                raise request_error("session_closing")
            if self.state is McpResourceState.OPEN:
                return self._toolsets or ()
            if self.connect_task is None:
                self.state = McpResourceState.CONNECTING
                self.connect_task = asyncio.create_task(self._connect_once())
            task = self.connect_task
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await wait_and_observe(task, timeout=10)
            raise

    async def _connect_once(self) -> "tuple[Any, ...]":
        from ..agent.mcp.connection import MCPConnectionPool

        pool = MCPConnectionPool()
        try:
            values = [
                await pool.get_toolset(mcp_spec(descriptor))
                for descriptor in self.descriptors
            ]
            toolsets = tuple(handle.toolset for handle in values)
        except asyncio.CancelledError:
            try:
                await _close_pool(pool)
            finally:
                await self._reset_connecting_state()
            raise
        except Exception as exc:
            try:
                await _close_pool(pool)
            finally:
                await self._reset_connecting_state()
            raise request_error("mcp_connection_failed") from exc
        async with self.lock:
            stale = self.state is not McpResourceState.CONNECTING
            if not stale:
                self.pool = pool
                self._toolsets = toolsets
                self.connect_task = None
                self.state = McpResourceState.OPEN
        if stale:
            await _close_pool(pool)
            async with self.lock:
                if self.connect_task is asyncio.current_task():
                    self.connect_task = None
            return ()
        return toolsets

    async def close(self) -> None:
        async with self.lock:
            if self.state is McpResourceState.CLOSED:
                return
            if self.close_task is not None:
                task = self.close_task
            else:
                self.state = McpResourceState.CLOSING
                task = asyncio.create_task(self._close_once())
                self.close_task = task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await wait_and_observe(task, timeout=10)
            raise

    async def _close_once(self) -> None:
        async with self.lock:
            connect_task = self.connect_task
            pool = self.pool
        if connect_task is not None and not connect_task.done():
            if not await cancel_and_wait(connect_task, timeout=10):
                async with self.lock:
                    self.state = McpResourceState.OPEN
                    self.close_task = None
                raise TimeoutError("MCP connect task ignored cancellation")
            logger.info("event=acp.mcp.connect_cancelled pending_task_count=0 state=closing")
        if pool is not None:
            try:
                await asyncio.wait_for(pool.close(), timeout=10)
            except Exception:
                async with self.lock:
                    self.state = McpResourceState.OPEN
                    self.close_task = None
                raise
        async with self.lock:
            self.pool = None
            self._toolsets = None
            self.connect_task = None
            self.state = McpResourceState.CLOSED
            self.descriptors = ()
            self.close_task = None

    async def _reset_connecting_state(self) -> None:
        async with self.lock:
            if self.connect_task is asyncio.current_task():
                self.connect_task = None
                if self.state is McpResourceState.CONNECTING:
                    self.state = McpResourceState.NEW


async def _close_pool(pool: Any) -> None:
    await asyncio.wait_for(pool.close(), timeout=10)


__all__ = ["McpResourceState", "SessionMcpResources"]
