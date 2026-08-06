#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Task and worker runtime protocols."""

from typing import Protocol

from ..domain.execution import Page
from .model import (
    CancelTaskRequest, ClaimTaskRequest, CompleteTaskRequest, FailTaskRequest,
    ListTasksRequest, RenewTaskRequest, RetryTaskRequest, SubmitTaskRequest,
    TaskClaim, TaskView,
)


class TaskApi(Protocol):
    async def submit(self, request: SubmitTaskRequest) -> TaskView: ...
    async def inspect(self, task_id: str) -> TaskView: ...
    async def list(self, request: ListTasksRequest) -> "Page[TaskView]": ...
    async def retry(self, task_id: str, request: RetryTaskRequest) -> TaskView: ...
    async def cancel(self, task_id: str, request: CancelTaskRequest) -> TaskView: ...


class TaskWorkerApi(Protocol):
    async def claim(self, request: ClaimTaskRequest) -> "TaskClaim | None": ...
    async def renew(self, request: RenewTaskRequest) -> TaskClaim: ...
    async def complete(self, request: CompleteTaskRequest) -> TaskView: ...
    async def fail(self, request: FailTaskRequest) -> TaskView: ...


__all__ = ["TaskApi", "TaskWorkerApi"]
