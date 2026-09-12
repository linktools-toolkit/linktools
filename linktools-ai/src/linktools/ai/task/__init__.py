#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic TaskGraph contracts and local scheduling."""

from ._api import open_local_task_graph_service
from ._event import TaskEvent, TaskEventType
from ._graph import (
    CancelGraphRequest,
    RecoverGraphRequest,
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
    TaskExpanderRef,
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
    TaskNodeInvocation,
    TaskNodeRunControl,
    TaskNodeRunError,
    TaskNodeRunner,
    TaskNodeRunResult,
)
from ._service import TaskGraphLauncher, TaskGraphQueryService, TaskGraphService
from ._service_impl import DefaultTaskGraphService, TaskPersistence

__all__ = [
    "CancelGraphRequest",
    "DefaultTaskGraphService",
    "LocalTaskGraphLauncher",
    "RecoverGraphRequest",
    "TaskDependency",
    "TaskDependencyResult",
    "TaskEvent",
    "TaskEventType",
    "TaskFunction",
    "TaskGraph",
    "TaskGraphAdmission",
    "TaskGraphHandle",
    "TaskGraphLaunch",
    "TaskGraphLauncher",
    "TaskGraphLimits",
    "TaskGraphQueryService",
    "TaskGraphRequest",
    "TaskGraphResult",
    "TaskGraphService",
    "TaskGraphSnapshot",
    "TaskGraphView",
    "TaskLease",
    "TaskNode",
    "TaskExpanderRef",
    "TaskNodeContext",
    "TaskNodeHandler",
    "TaskNodeInvocation",
    "TaskNodeResult",
    "TaskNodeRunControl",
    "TaskNodeRunError",
    "TaskNodeRunResult",
    "TaskNodeRunner",
    "TaskNodeView",
    "TaskPersistence",
    "TaskResultRecord",
    "TaskStatus",
    "TaskTerminalRecord",
    "open_local_task_graph_service",
    "ready_nodes",
]
