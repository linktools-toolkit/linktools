#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic TaskGraph value objects."""

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..core import (
    JsonValue,
    Principal,
    TaskStatus,
    canonical_json_bytes,
    validate_idempotency_key,
    validate_lease_owner,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode


@dataclass(frozen=True, slots=True)
class TaskGraphLimits:
    max_concurrency: int = 8
    max_depth: int = 8
    max_nodes: int = 128
    max_budget: int = 1000

    def __post_init__(self) -> None:
        values = (self.max_concurrency, self.max_depth, self.max_nodes, self.max_budget)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in values):
            raise ValueError("task graph limits must be positive")


def _normalize_json_mapping(value: Mapping[str, JsonValue]) -> "dict[str, JsonValue]":
    normalized: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("task node input keys must be non-empty strings")
        normalized[key] = _normalize_json_value(item)
    return normalized


def _normalize_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("task node input numbers must be finite")
        return value
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return _normalize_json_mapping(value)
    raise TypeError(f"unsupported task node input value: {type(value).__name__}")


@dataclass(frozen=True, slots=True, init=False)
class TaskNode:
    node_id: str
    dependencies: tuple[str, ...]
    budget_cost: int
    _input: bytes = field(repr=False)

    def __init__(
        self,
        node_id: str,
        dependencies: "tuple[str, ...]" = (),
        *,
        input: "Mapping[str, JsonValue] | None" = None,
        budget_cost: int = 1,
    ) -> None:
        if isinstance(dependencies, (str, bytes)):
            raise TypeError("task node dependencies are invalid")
        try:
            normalized_dependencies = tuple(dependencies)
        except TypeError as error:
            raise TypeError("task node dependencies are invalid") from error
        if (
            not isinstance(node_id, str)
            or not node_id.strip()
            or any(not isinstance(item, str) or not item.strip() for item in normalized_dependencies)
            or len(set(normalized_dependencies)) != len(normalized_dependencies)
            or not isinstance(budget_cost, int)
            or isinstance(budget_cost, bool)
            or budget_cost < 1
        ):
            raise ValueError("task node identity is invalid")
        values: Mapping[str, JsonValue] = {} if input is None else input
        if not isinstance(values, Mapping):
            raise TypeError("task node input must be a mapping")
        normalized = _normalize_json_mapping(values)
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "dependencies", normalized_dependencies)
        object.__setattr__(self, "budget_cost", budget_cost)
        object.__setattr__(self, "_input", canonical_json_bytes(normalized))

    @property
    def input(self) -> "dict[str, JsonValue]":
        return json.loads(self._input.decode("utf-8"))


@dataclass(frozen=True, slots=True)
class TaskLease:
    graph_id: str
    node_id: str
    tenant_id: str
    owner: str
    fence: int
    lease_expires_at: datetime

    def __post_init__(self) -> None:
        try:
            validate_tenant_id(self.tenant_id)
            validate_lease_owner(self.owner)
        except AIError as error:
            raise ValueError("task lease identity is invalid") from error
        if not self.graph_id.strip() or not self.node_id.strip() or self.fence < 1 or self.lease_expires_at.tzinfo is None:
            raise ValueError("task lease is invalid")


@dataclass(frozen=True, slots=True)
class TaskNodeView:
    graph_id: str
    node_id: str
    dependencies: "tuple[str, ...]"
    status: TaskStatus
    owner: "str | None"
    fence: int
    lease_expires_at: "datetime | None"
    result_digest: "str | None"
    error_code: "str | None"
    error_digest: "str | None"
    execution_id: "str | None" = None

    def __post_init__(self) -> None:
        if self.owner is not None:
            try:
                validate_lease_owner(self.owner)
            except AIError as error:
                raise ValueError("task node lease owner is invalid") from error


@dataclass(frozen=True, slots=True)
class TaskGraph:
    graph_id: str
    nodes: "tuple[TaskNode, ...]"

    def __post_init__(self) -> None:
        if not self.graph_id.strip():
            raise ValueError("task graph id is required")
        object.__setattr__(self, "nodes", tuple(self.nodes))
        ids = {node.node_id for node in self.nodes}
        if len(ids) != len(self.nodes) or any(dependency not in ids for node in self.nodes for dependency in node.dependencies):
            raise TaskGraphValidationError(ErrorCode.TASK_DEPENDENCY_UNKNOWN, "task graph contains an unknown dependency")
        self._topological_order()

    def _topological_order(self) -> 'tuple[str, ...]':
        remaining = {node.node_id: set(node.dependencies) for node in self.nodes}
        order: list[str] = []
        while remaining:
            ready = tuple(sorted(node_id for node_id, dependencies in remaining.items() if not dependencies))
            if not ready:
                raise TaskGraphValidationError(ErrorCode.TASK_GRAPH_CYCLE, "task graph contains a cycle")
            order.extend(ready)
            for node_id in ready:
                remaining.pop(node_id)
            for dependencies in remaining.values():
                dependencies.difference_update(ready)
        return tuple(order)

    def validate_limits(self, limits: TaskGraphLimits) -> None:
        if len(self.nodes) > limits.max_nodes:
            raise AIError(ErrorCode.TASK_DAG_INVALID, "task graph exceeds node limit")
        depths: dict[str, int] = {}
        for node_id in self._topological_order():
            node = next(node for node in self.nodes if node.node_id == node_id)
            depths[node_id] = 1 + max((depths[item] for item in node.dependencies), default=0)
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
    node_id: str
    owner: str
    fence: int
    status: TaskStatus
    result_digest: "str | None"
    error_code: "str | None"
    error_digest: "str | None"
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_id: "str | None" = None

    def __post_init__(self) -> None:
        try:
            validate_lease_owner(self.owner)
        except AIError as error:
            raise ValueError("task terminal identity is invalid") from error
        if self.completed_at.tzinfo is None:
            raise ValueError("task terminal time must be timezone-aware")


class TaskCompletionLedger:
    """Apply terminal task transitions with owner and fence idempotency."""

    def __init__(self) -> None:
        self._records: dict[str, TaskTerminalRecord] = {}

    def complete(self, node_id: str, owner: str, fence: int, result_digest: str, execution_id: "str | None" = None) -> TaskTerminalRecord:
        if not result_digest:
            raise ValueError("result digest is required")
        return self._apply(
            TaskTerminalRecord(node_id, owner, fence, TaskStatus.SUCCEEDED, result_digest, None, None, execution_id=execution_id)
        )

    def fail(self, node_id: str, owner: str, fence: int, error_code: str, error_digest: str) -> TaskTerminalRecord:
        if not error_code or not error_digest:
            raise ValueError("failure code and digest are required")
        return self._apply(
            TaskTerminalRecord(node_id, owner, fence, TaskStatus.FAILED, None, error_code, error_digest)
        )

    def get(self, node_id: str) -> 'TaskTerminalRecord | None':
        return self._records.get(node_id)

    def _apply(self, candidate: TaskTerminalRecord) -> TaskTerminalRecord:
        if candidate.fence < 1 or not candidate.node_id or not candidate.owner:
            if candidate.fence < 1 and candidate.node_id and candidate.owner and candidate.node_id in self._records:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            raise ValueError("task terminal identity is invalid")
        previous = self._records.get(candidate.node_id)
        if previous is None:
            self._records[candidate.node_id] = candidate
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
        left.node_id == right.node_id
        and left.owner == right.owner
        and left.fence == right.fence
        and left.status is right.status
        and left.result_digest == right.result_digest
        and left.error_code == right.error_code
        and left.error_digest == right.error_digest
        and left.execution_id == right.execution_id
    )


@dataclass(frozen=True, slots=True)
class TaskGraphRequest:
    graph: TaskGraph
    principal: Principal
    idempotency_key: str = ""
    limits: TaskGraphLimits = field(default_factory=TaskGraphLimits)

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)
        self.graph.validate_limits(self.limits)


@dataclass(frozen=True, slots=True)
class TaskGraphResult:
    graph_id: str
    status: TaskStatus
    execution_ids: "tuple[str, ...]"
    node_results: "tuple[TaskNodeResult, ...]" = ()


@dataclass(frozen=True, slots=True)
class TaskNodeResult:
    node_id: str
    status: TaskStatus
    result_digest: "str | None"
    execution_id: "str | None"
    error_code: "str | None"
    error_digest: "str | None"


@dataclass(frozen=True, slots=True)
class TaskDependencyResult:
    result_digest: str
    execution_id: "str | None" = None

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.result_digest) is None:
            raise ValueError("task dependency result digest is invalid")


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
    idempotency_key: str
    force: bool = False

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)


def ready_nodes(graph: TaskGraph, completed: "frozenset[str]") -> "tuple[TaskNode, ...]":
    return tuple(
        node for node in graph.nodes if node.node_id not in completed and all(dependency in completed for dependency in node.dependencies)
    )


__all__ = [
    "CancelGraphRequest",
    "TaskCompletionLedger",
    "TaskDependencyResult",
    "TaskGraph",
    "TaskGraphHandle",
    "TaskGraphLimits",
    "TaskGraphRequest",
    "TaskGraphResult",
    "TaskGraphValidationError",
    "TaskGraphView",
    "TaskNode",
    "TaskStatus",
    "TaskTerminalRecord",
    "ready_nodes",
]
