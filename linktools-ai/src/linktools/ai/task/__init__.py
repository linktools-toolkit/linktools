#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic TaskGraph contracts and local scheduling."""

from ._api import open_local_task_api
from ._graph import (
    CancelGraphRequest,
    TaskCompletionLedger,
    TaskDependencyResult,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphHandle,
    TaskGraphLaunch,
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
    TaskNodeRunError,
    TaskNodeRunner,
    TaskNodeRunResult,
)
from ._service import TaskApi, TaskGraphLauncher, TaskQueryApi
from ._service_impl import DefaultTaskService, TaskPersistence

__all__ = [
    "CancelGraphRequest",
    "DefaultTaskService",
    "LocalTaskGraphLauncher",
    "TaskApi",
    "TaskCompletionLedger",
    "TaskDependencyResult",
    "TaskGraph",
    "TaskGraphAdmission",
    "TaskGraphHandle",
    "TaskGraphLaunch",
    "TaskGraphLauncher",
    "TaskGraphLimits",
    "TaskGraphRequest",
    "TaskGraphResult",
    "TaskGraphView",
    "TaskLease",
    "TaskNode",
    "TaskNodeResult",
    "TaskNodeRunError",
    "TaskNodeRunResult",
    "TaskNodeRunner",
    "TaskNodeView",
    "TaskPersistence",
    "TaskQueryApi",
    "TaskStatus",
    "TaskTerminalRecord",
    "open_local_task_api",
    "ready_nodes",
]
