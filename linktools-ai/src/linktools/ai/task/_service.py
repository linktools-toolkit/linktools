#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic TaskGraph service contracts."""

from collections.abc import AsyncIterator
from typing import Protocol

from ..core import Principal
from ._graph import (
    CancelGraphRequest,
    TaskGraphHandle,
    TaskGraphLaunch,
    TaskGraphRequest,
    TaskGraphResult,
    TaskGraphSnapshot,
    TaskGraphView,
)


class TaskQueryApi(Protocol):
    async def inspect_graph(
        self,
        graph_id: str,
        *,
        principal: Principal,
    ) -> TaskGraphView: ...

    async def inspect_graph_state(
        self,
        graph_id: str,
        *,
        principal: Principal,
    ) -> TaskGraphSnapshot: ...

    def stream_graph(
        self,
        graph_id: str,
        *,
        principal: Principal,
    ) -> AsyncIterator[TaskGraphSnapshot]: ...

    async def wait_graph(
        self,
        graph_id: str,
        *,
        principal: Principal,
        timeout_seconds: "float | None" = None,
    ) -> TaskGraphResult: ...


class TaskApi(TaskQueryApi, Protocol):
    async def run_graph(self, request: TaskGraphRequest) -> TaskGraphResult: ...

    async def run_graph_and_wait(
        self,
        request: TaskGraphRequest,
        *,
        timeout_seconds: "float | None" = None,
    ) -> TaskGraphResult: ...

    async def cancel_graph(
        self,
        graph_id: str,
        request: CancelGraphRequest,
    ) -> TaskGraphResult: ...

    async def preflight_close(self) -> None: ...


class TaskGraphLauncher(Protocol):
    async def start(self, request: TaskGraphLaunch) -> TaskGraphHandle: ...

    async def cancel(self, request: TaskGraphLaunch) -> None: ...


__all__ = ["TaskApi", "TaskGraphLauncher", "TaskQueryApi"]
