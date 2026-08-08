#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task, Job and Swarm value objects."""

from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..core.errors import ErrorCode, AIError
from ..core.validation import validate_idempotency_key
from ..core.value import Principal, TaskStatus


@dataclass(frozen=True, slots=True)
class SwarmLimits:
    max_concurrency: int = 8
    max_depth: int = 8
    max_nodes: int = 128
    max_budget: int = 1000

    def __post_init__(self) -> None:
        if min(self.max_concurrency, self.max_depth, self.max_nodes, self.max_budget) < 1:
            raise ValueError("swarm limits must be positive")


@dataclass(frozen=True, slots=True)
class TaskNode:
    task_id: str
    dependencies: "tuple[str, ...]" = ()
    binding_digest: "str | None" = None
    budget_cost: int = 1

    def __post_init__(self) -> None:
        if not self.task_id.strip() or len(set(self.dependencies)) != len(self.dependencies) or any(not item.strip() for item in self.dependencies) or self.budget_cost < 1:
            raise ValueError("task node identity is invalid")
        if self.binding_digest is not None and not self.binding_digest.strip():
            raise ValueError("task node binding digest is invalid")
        object.__setattr__(self, "dependencies", tuple(self.dependencies))


@dataclass(frozen=True, slots=True)
class TaskGraph:
    graph_id: str
    nodes: "tuple[TaskNode, ...]"

    def __post_init__(self) -> None:
        if not self.graph_id.strip():
            raise ValueError("task graph id is required")
        object.__setattr__(self, "nodes", tuple(self.nodes))
        ids = {node.task_id for node in self.nodes}
        if len(ids) != len(self.nodes) or any(dependency not in ids for node in self.nodes for dependency in node.dependencies):
            raise TaskGraphValidationError(ErrorCode.TASK_DEPENDENCY_UNKNOWN, "task graph contains an unknown dependency")
        self._topological_order()

    def _topological_order(self) -> 'tuple[str, ...]':
        remaining = {node.task_id: set(node.dependencies) for node in self.nodes}
        order: list[str] = []
        while remaining:
            ready = tuple(sorted(task_id for task_id, dependencies in remaining.items() if not dependencies))
            if not ready:
                raise TaskGraphValidationError(ErrorCode.TASK_GRAPH_CYCLE, "task graph contains a cycle")
            order.extend(ready)
            for task_id in ready:
                remaining.pop(task_id)
            for dependencies in remaining.values():
                dependencies.difference_update(ready)
        return tuple(order)

    def validate_limits(self, limits: SwarmLimits) -> None:
        if len(self.nodes) > limits.max_nodes:
            raise AIError(ErrorCode.TASK_DAG_INVALID, "task graph exceeds node limit")
        depths: dict[str, int] = {}
        for task_id in self._topological_order():
            node = next(node for node in self.nodes if node.task_id == task_id)
            depths[task_id] = 1 + max((depths[item] for item in node.dependencies), default=0)
        if max(depths.values(), default=0) > limits.max_depth:
            raise AIError(ErrorCode.TASK_DAG_INVALID, "task graph exceeds depth limit")
        if sum(node.budget_cost for node in self.nodes) > limits.max_budget:
            raise AIError(ErrorCode.TASK_DAG_INVALID, "task graph exceeds budget limit")


class TaskGraphValidationError(AIError, ValueError):
    def __init__(self, code: ErrorCode, message: str) -> None:
        AIError.__init__(self, code, message)
        ValueError.__init__(self, message)


@dataclass(frozen=True, slots=True)
class TaskTerminalRecord:
    task_id: str
    owner: str
    fence: int
    status: TaskStatus
    result_digest: "str | None"
    error_code: "str | None"
    error_digest: "str | None"
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if self.completed_at.tzinfo is None:
            raise ValueError("task terminal time must be timezone-aware")


class TaskCompletionLedger:
    """Apply terminal task transitions with owner and fence idempotency."""

    def __init__(self) -> None:
        self._records: dict[str, TaskTerminalRecord] = {}

    def complete(self, task_id: str, owner: str, fence: int, result_digest: str) -> TaskTerminalRecord:
        if not result_digest:
            raise ValueError("result digest is required")
        return self._apply(
            TaskTerminalRecord(task_id, owner, fence, TaskStatus.SUCCEEDED, result_digest, None, None)
        )

    def fail(self, task_id: str, owner: str, fence: int, error_code: str, error_digest: str) -> TaskTerminalRecord:
        if not error_code or not error_digest:
            raise ValueError("failure code and digest are required")
        return self._apply(
            TaskTerminalRecord(task_id, owner, fence, TaskStatus.FAILED, None, error_code, error_digest)
        )

    def get(self, task_id: str) -> 'TaskTerminalRecord | None':
        return self._records.get(task_id)

    def _apply(self, candidate: TaskTerminalRecord) -> TaskTerminalRecord:
        if candidate.fence < 1 or not candidate.task_id or not candidate.owner:
            if candidate.fence < 1 and candidate.task_id and candidate.owner and candidate.task_id in self._records:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            raise ValueError("task terminal identity is invalid")
        previous = self._records.get(candidate.task_id)
        if previous is None:
            self._records[candidate.task_id] = candidate
            return candidate
        if candidate.fence < previous.fence:
            raise AIError(ErrorCode.TASK_FENCE_STALE)
        if candidate.fence == previous.fence:
            if candidate.owner != previous.owner:
                raise AIError(ErrorCode.TASK_OWNER_CONFLICT)
            if _same_terminal_result(candidate, previous):
                return previous
            if candidate.status is not previous.status:
                raise AIError(ErrorCode.TASK_TERMINAL_CONFLICT)
            if candidate.status is TaskStatus.SUCCEEDED:
                raise AIError(ErrorCode.TASK_RESULT_CONFLICT)
            raise AIError(ErrorCode.TASK_RESULT_CONFLICT)
        raise AIError(ErrorCode.TASK_TERMINAL_CONFLICT)


def _same_terminal_result(left: TaskTerminalRecord, right: TaskTerminalRecord) -> bool:
    return (
        left.task_id == right.task_id
        and left.owner == right.owner
        and left.fence == right.fence
        and left.status is right.status
        and left.result_digest == right.result_digest
        and left.error_code == right.error_code
        and left.error_digest == right.error_digest
    )


@dataclass(frozen=True, slots=True)
class TaskGraphRequest:
    graph: TaskGraph
    principal: Principal
    idempotency_key: str = ""
    limits: SwarmLimits = field(default_factory=SwarmLimits)

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)
        self.graph.validate_limits(self.limits)


@dataclass(frozen=True, slots=True)
class TaskGraphResult:
    graph_id: str
    status: TaskStatus
    execution_ids: "tuple[str, ...]"


@dataclass(frozen=True, slots=True)
class TaskGraphHandle:
    graph_id: str
    workflow_id: str


@dataclass(frozen=True, slots=True)
class TaskGraphView:
    graph_id: str
    status: TaskStatus
    nodes: "tuple[TaskNode, ...]"


@dataclass(frozen=True, slots=True)
class CancelGraphRequest:
    principal: Principal
    cancel_request_id: str
    force: bool = False

    def __post_init__(self) -> None:
        validate_idempotency_key(self.cancel_request_id)


@dataclass(frozen=True, slots=True)
class Job:
    job_id: str
    graph: TaskGraph
    status: TaskStatus = TaskStatus.PENDING


@dataclass(frozen=True, slots=True)
class Swarm:
    swarm_id: str
    graph: TaskGraph
    parent_execution_id: str


def ready_nodes(graph: TaskGraph, completed: "frozenset[str]") -> "tuple[TaskNode, ...]":
    return tuple(
        node for node in graph.nodes if node.task_id not in completed and all(dependency in completed for dependency in node.dependencies)
    )


__all__ = [
    "CancelGraphRequest", "Job", "Swarm", "SwarmLimits", "TaskCompletionLedger", "TaskGraph",
    "TaskGraphHandle", "TaskGraphRequest", "TaskGraphResult", "TaskGraphView", "TaskNode",
    "TaskGraphValidationError", "TaskStatus", "TaskTerminalRecord",
    "ready_nodes",
]
