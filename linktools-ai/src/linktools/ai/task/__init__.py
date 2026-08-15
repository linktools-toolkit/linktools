#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic TaskGraph contracts and local scheduling."""

from ._api import open_local_task_api
from ._graph import (
    CancelGraphRequest,
    TaskCompletionLedger,
    TaskDependencyResult,
    TaskGraph,
    TaskGraphHandle,
    TaskGraphLimits,
    TaskGraphRequest,
    TaskGraphResult,
    TaskGraphView,
    TaskLease,
    TaskNode,
    TaskNodeResult,
    TaskNodeView,
    TaskStatus,
    TaskTerminalRecord,
    ready_nodes,
)
from ._local import (
    LocalTaskGraphLauncher,
    TaskNodeRunner,
    TaskNodeRunResult,
)
from ._service import TaskApi, TaskGraphLauncher, TaskQueryApi
from ._service_impl import DefaultTaskService, TaskPersistence

__all__ = [
    "CancelGraphRequest", "DefaultTaskService", "TaskGraphLimits", "TaskApi",
    "TaskCompletionLedger", "TaskDependencyResult", "TaskGraph", "TaskGraphLauncher",
    "TaskLease", "TaskPersistence", "LocalTaskGraphLauncher", "TaskGraphHandle",
    "TaskGraphRequest", "TaskGraphResult", "TaskGraphView", "TaskNode", "TaskNodeResult",
    "TaskNodeRunResult", "TaskNodeRunner", "TaskNodeView", "TaskQueryApi", "open_local_task_api",
    "TaskStatus", "TaskTerminalRecord", "ready_nodes",
]
