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
    TaskGraphSnapshot,
    TaskGraphView,
    TaskLease,
    TaskNode,
    TaskNodeResult,
    TaskNodeView,
    TaskResultRecord,
    TaskStatus,
    TaskTerminalRecord,
    ready_nodes,
)
from ._handler import TaskDependency, TaskFunction, TaskNodeContext, TaskNodeHandler
from ._local import (
    LocalTaskGraphLauncher,
    TaskNodeRunControl,
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
    "TaskDependency",
    "TaskDependencyResult",
    "TaskFunction",
    "TaskGraph",
    "TaskGraphAdmission",
    "TaskGraphHandle",
    "TaskGraphLaunch",
    "TaskGraphLauncher",
    "TaskGraphLimits",
    "TaskGraphRequest",
    "TaskGraphResult",
    "TaskGraphSnapshot",
    "TaskGraphView",
    "TaskLease",
    "TaskNode",
    "TaskNodeContext",
    "TaskNodeHandler",
    "TaskNodeResult",
    "TaskNodeRunControl",
    "TaskNodeRunError",
    "TaskNodeRunResult",
    "TaskNodeRunner",
    "TaskNodeView",
    "TaskPersistence",
    "TaskQueryApi",
    "TaskResultRecord",
    "TaskStatus",
    "TaskTerminalRecord",
    "open_local_task_api",
    "ready_nodes",
]
