#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task API protocol."""

from typing import Protocol

from ..core.value import Principal
from .graph import CancelGraphRequest, TaskGraphRequest, TaskGraphResult, TaskGraphView

class TaskQueryApi(Protocol):
    async def inspect_graph(self, graph_id: str, *, principal: Principal) -> TaskGraphView: ...


class TaskApi(TaskQueryApi, Protocol):
    async def run_graph(self, request: TaskGraphRequest) -> TaskGraphResult: ...
    async def cancel_graph(self, graph_id: str, request: CancelGraphRequest) -> TaskGraphView: ...


__all__ = ["TaskApi", "TaskQueryApi"]
