#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task domain: reliable task execution as an extension of the existing runtime.

Public types. The concrete ``TaskRuntime`` and the orchestration commands land
in later phases; the data models, state machine, handler contract and store
contract are the stable surface exported here.
"""

from .models import (
    ActorChain,
    ActorRef,
    AttemptStatus,
    IllegalTaskTransitionError,
    JobRecord,
    JobStatus,
    ResourceSnapshotRef,
    RetryPolicy,
    SideEffectMode,
    SideEffectPolicy,
    TaskAttemptRecord,
    TaskBudget,
    TaskFailureKind,
    TaskPrincipal,
    TaskRecord,
    TaskSignalRecord,
    TaskStatus,
    TaskTransitionRecord,
)
from .protocols import (
    CancelJob,
    CancelTask,
    CreateTask,
    CompleteJob,
    TaskCommand,
    WaitSignal,
    CancellationToken,
    Clock,
    TaskContext,
    TaskFailure,
    TaskHandler,
    TaskOutcome,
    TaskRequest,
    TaskSuccess,
)
from .runtime import TaskRuntime, TaskRuntimeOptions, TaskStoreRequiredError
from .store import ClaimedTask, TaskClaim, TaskStore

__all__: "list[str]" = [
    "JobStatus",
    "TaskStatus",
    "AttemptStatus",
    "TaskFailureKind",
    "SideEffectMode",
    "RetryPolicy",
    "SideEffectPolicy",
    "TaskPrincipal",
    "ActorRef",
    "ActorChain",
    "TaskBudget",
    "ResourceSnapshotRef",
    "JobRecord",
    "TaskRecord",
    "TaskAttemptRecord",
    "TaskTransitionRecord",
    "TaskSignalRecord",
    "IllegalTaskTransitionError",
    "Clock",
    "CancellationToken",
    "TaskRequest",
    "TaskContext",
    "TaskSuccess",
    "CreateTask",
    "WaitSignal",
    "CompleteJob",
    "CancelTask",
    "CancelJob",
    "TaskCommand",
    "TaskFailure",
    "TaskOutcome",
    "TaskHandler",
    "TaskClaim",
    "ClaimedTask",
    "TaskStore",
    "TaskRuntime",
    "TaskRuntimeOptions",
    "TaskStoreRequiredError",
]
