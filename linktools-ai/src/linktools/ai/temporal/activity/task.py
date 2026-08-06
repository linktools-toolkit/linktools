#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task Activity with one injected graph-operation port."""

from typing import Protocol

from linktools.core import environ

from ..workflow.task import TaskWorkflowInput, TaskWorkflowResult
from .execution import ActivityOptions

try:
    from temporalio import activity as _temporal_activity
except ModuleNotFoundError as error:
    if error.name != "temporalio":
        raise
    _temporal_activity = None

_logger = environ.get_logger("ai.temporal.activity.task")


class TaskOperation(Protocol):
    async def execute(self, request: TaskWorkflowInput) -> TaskWorkflowResult: ...


class TaskActivity:
    options = ActivityOptions(start_to_close_seconds=900, retry_max_attempts=3, heartbeat_timeout_seconds=30)

    def __init__(self, operation: TaskOperation) -> None:
        self._operation = operation

    async def run(self, request: TaskWorkflowInput) -> TaskWorkflowResult:
        _logger.info("executing task activity: graph_id=%s", request.graph_id)
        return await self._operation.execute(request)


if _temporal_activity is not None:
    TaskActivity.run = _temporal_activity.defn(name="task_graph")(TaskActivity.run)


__all__ = ["TaskActivity", "TaskOperation"]
