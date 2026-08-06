#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Public Task, Job and Swarm DTOs."""

from pydantic import BaseModel, ConfigDict, Field

from ..domain.task import TaskStatus


class SubmitTaskRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    task_id: str
    dependencies: "tuple[str, ...]" = ()
    payload: object = None


class ListTasksRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: "str | None" = None
    limit: int = Field(default=100, ge=1, le=200)
    cursor: "str | None" = None


class RetryTaskRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason: "str | None" = None


class CancelTaskRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason: "str | None" = None


class ClaimTaskRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    worker_id: str


class RenewTaskRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    worker_id: str
    fence: int = Field(ge=1)


class CompleteTaskRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    worker_id: str
    fence: int = Field(ge=1)
    result: object = None


class FailTaskRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    worker_id: str
    fence: int = Field(ge=1)
    error: str


class TaskView(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    job_id: str
    status: TaskStatus
    attempt: int
    fence: int
    result: object = None
    error: "str | None" = None


class TaskClaim(BaseModel):
    model_config = ConfigDict(frozen=True)

    task: TaskView
    fence: int
    lease_expires_at: str


__all__ = [
    "CancelTaskRequest", "ClaimTaskRequest", "CompleteTaskRequest", "FailTaskRequest",
    "ListTasksRequest", "RenewTaskRequest", "RetryTaskRequest", "SubmitTaskRequest",
    "TaskClaim", "TaskView",
]
