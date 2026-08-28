#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task API protocol."""

from typing import Protocol

from ..core import Principal
from ._graph import (
    CancelGraphRequest,
    TaskGraphHandle,
    TaskGraphLaunch,
    TaskGraphRequest,
    TaskGraphResult,
    TaskGraphView,
)


class TaskQueryApi(Protocol):
    async def inspect_graph(self, graph_id: str, *, principal: Principal) -> TaskGraphView: ...
    async def wait_graph(self, graph_id: str, *, principal: Principal, timeout_seconds: "float | None" = None) -> TaskGraphResult: ...


class TaskApi(TaskQueryApi, Protocol):
    async def run_graph(self, request: TaskGraphRequest) -> TaskGraphResult: ...
    async def run_graph_and_wait(self, request: TaskGraphRequest, *, timeout_seconds: "float | None" = None) -> TaskGraphResult: ...
    async def cancel_graph(self, graph_id: str, request: CancelGraphRequest) -> TaskGraphView: ...
    async def preflight_close(self) -> None: ...


class TaskGraphLauncher(Protocol):
    async def start(self, launch: TaskGraphLaunch) -> TaskGraphHandle: ...
    async def cancel(self, graph_id: str, request: CancelGraphRequest) -> TaskGraphView: ...


__all__ = ["TaskApi", "TaskGraphLauncher", "TaskQueryApi"]
