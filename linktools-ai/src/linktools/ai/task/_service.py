#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic TaskGraph service contracts."""

from collections.abc import AsyncIterator
from typing import Protocol

from ..core import Page, Principal
from ._event import TaskEvent
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

    async def list_graph_events(
        self,
        graph_id: str,
        *,
        principal: Principal,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> Page[TaskEvent]: ...

    def stream_graph_events(
        self,
        graph_id: str,
        *,
        principal: Principal,
        after_sequence: int = 0,
    ) -> AsyncIterator[TaskEvent]: ...

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
    ) -> TaskGraphView: ...

    async def preflight_close(self) -> None: ...


class TaskGraphLauncher(Protocol):
    async def start(self, launch: TaskGraphLaunch) -> TaskGraphHandle: ...

    async def cancel(self, launch: TaskGraphLaunch) -> TaskGraphView: ...


__all__ = ["TaskApi", "TaskGraphLauncher", "TaskQueryApi"]
