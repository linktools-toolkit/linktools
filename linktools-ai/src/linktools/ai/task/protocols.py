#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task handler contracts and runtime protocols.

A downstream plugs work into the task runtime by implementing
:class:`TaskHandler`. The runtime hands it a :class:`TaskRequest` plus a
:class:`TaskContext` carrying the full per-execution state (principal, actor
chain, budget, resource snapshots, cancellation); the handler returns a
:class:`TaskOutcome` (success with commands, or a typed failure). Handlers
never read tenant / user / workspace / budget from globals -- only from the
context.

``Clock`` abstracts time and sleep so lease / retry / timeout logic is
deterministic under a fake clock in tests rather than sleeping for real.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .models import (
    RetryPolicy,
    SideEffectPolicy,
    ActorChain,
    ArtifactRef,
    ResourceSnapshotRef,
    TaskBudget,
    TaskFailureKind,
    TaskPrincipal,
)


from ..clock import Clock, SystemClock  # noqa: E402,F401  (re-export)


class CancellationToken:
    """Cooperative cancellation flag the worker raises when a task is
    cancelled or times out; the handler is expected to poll ``is_set``."""

    def __init__(self) -> None:
        self._set = False

    def trigger(self) -> None:
        self._set = True

    @property
    def is_set(self) -> bool:
        return self._set


@dataclass(frozen=True, slots=True)
class TaskRequest:
    input_artifact: "ArtifactRef | None"
    metadata: "Mapping[str, Any]"


@dataclass(frozen=True, slots=True)
class TaskContext:
    job_id: str
    task_id: str
    attempt_id: str
    fencing_token: int
    worker_id: str
    principal: TaskPrincipal
    actor_chain: ActorChain
    delegated_scopes: "tuple[str, ...]"
    budget: TaskBudget
    resource_snapshots: "tuple[ResourceSnapshotRef, ...]"
    cancellation: CancellationToken


# ---- orchestration commands ----


@dataclass(frozen=True, slots=True)
class CreateTask:
    key: str
    handler: str
    input_artifact: "ArtifactRef | None" = None
    dependencies: "tuple[str, ...]" = ()
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    side_effect_policy: SideEffectPolicy = field(default_factory=SideEffectPolicy)
    timeout_seconds: "float | None" = None
    metadata: "Mapping[str, Any]" = field(default_factory=dict)
    delegated_scopes: "tuple[str, ...] | None" = None


@dataclass(frozen=True, slots=True)
class WaitSignal:
    name: str
    correlation_key: str
    timeout_seconds: "float | None" = None


@dataclass(frozen=True, slots=True)
class CompleteJob:
    output_artifact: "ArtifactRef | None" = None


@dataclass(frozen=True, slots=True)
class CancelTask:
    task_id: str


@dataclass(frozen=True, slots=True)
class CancelJob:
    job_id: "str | None" = None  # None = current job


TaskCommand = CreateTask | WaitSignal | CompleteJob | CancelTask | CancelJob


@dataclass(frozen=True, slots=True)
class TaskSuccess:
    output_artifact: "ArtifactRef | None" = None
    commands: "tuple[TaskCommand, ...]" = ()
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskFailure:
    kind: TaskFailureKind
    error_type: str
    message: str
    retryable: "bool | None" = None
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


TaskOutcome = TaskSuccess | TaskFailure


@runtime_checkable
class TaskHandler(Protocol):
    async def execute(
        self,
        request: TaskRequest,
        context: TaskContext,
    ) -> TaskOutcome: ...


__all__: "list[str]" = [
    "Clock",
    "CancellationToken",
    "TaskRequest",
    "TaskContext",
    "TaskSuccess",
    "TaskFailure",
    "TaskOutcome",
    "TaskHandler",
]
