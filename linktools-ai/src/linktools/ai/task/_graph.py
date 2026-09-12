#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic TaskGraph value objects."""

import heapq
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..core import (
    JsonValue,
    Principal,
    CorrelationData,
    TaskStatus,
    canonical_json_bytes,
    canonical_sha256,
    idempotency_key_digest,
    normalize_correlation,
    principal_identity_payload,
    validate_idempotency_key,
    validate_lease_owner,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode
from ..storage import StoredPayload


@dataclass(frozen=True, slots=True)
class TaskGraphLimits:
    max_concurrency: int = 8
    max_depth: int = 8
    max_nodes: int = 128
    max_budget: int = 1000

    def __post_init__(self) -> None:
        values = (self.max_concurrency, self.max_depth, self.max_nodes, self.max_budget)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in values
        ):
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


_TASK_EXPANDER_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True, slots=True)
class TaskExpanderRef:
    id: str
    version: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.id, str)
            or _TASK_EXPANDER_ID.fullmatch(self.id) is None
            or not isinstance(self.version, int)
            or isinstance(self.version, bool)
            or self.version < 1
        ):
            raise ValueError("task expander reference is invalid")


@dataclass(frozen=True, slots=True, init=False)
class TaskNode:
    node_id: str
    dependencies: tuple[str, ...]
    budget_cost: int
    expander: "TaskExpanderRef | None"
    _input: bytes = field(repr=False)

    def __init__(
        self,
        node_id: str,
        dependencies: "tuple[str, ...]" = (),
        *,
        input: "Mapping[str, JsonValue] | None" = None,
        budget_cost: int = 1,
        expander: "TaskExpanderRef | None" = None,
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
            or any(
                not isinstance(item, str) or not item.strip()
                for item in normalized_dependencies
            )
            or len(set(normalized_dependencies)) != len(normalized_dependencies)
            or not isinstance(budget_cost, int)
            or isinstance(budget_cost, bool)
            or budget_cost < 1
            or (expander is not None and not isinstance(expander, TaskExpanderRef))
        ):
            raise ValueError("task node identity is invalid")
        values: Mapping[str, JsonValue] = {} if input is None else input
        if not isinstance(values, Mapping):
            raise TypeError("task node input must be a mapping")
        normalized = _normalize_json_mapping(values)
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "dependencies", normalized_dependencies)
        object.__setattr__(self, "budget_cost", budget_cost)
        object.__setattr__(self, "expander", expander)
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
        if (
            not self.graph_id.strip()
            or not self.node_id.strip()
            or self.fence < 1
            or self.lease_expires_at.tzinfo is None
        ):
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
        if self.status in {
            TaskStatus.PENDING,
            TaskStatus.READY,
            TaskStatus.RUNNING,
        } and self.execution_id is not None:
            raise ValueError("unbound task node state cannot carry an execution id")
        if self.status is TaskStatus.RECOVERY_REQUIRED and (
            self.owner is not None
            or self.lease_expires_at is not None
            or self.fence < 1
            or self.result_digest is not None
            or self.error_code is None
            or not self.error_code.strip()
            or self.error_digest is None
        ):
            raise ValueError("recovery-required task node state is invalid")
        if self.status is TaskStatus.WAITING and (
            self.execution_id is None
            or not self.execution_id.strip()
            or self.owner is not None
            or self.lease_expires_at is not None
            or self.fence < 1
            or self.result_digest is not None
            or self.error_code is not None
            or self.error_digest is not None
        ):
            raise ValueError("waiting task node state is invalid")


@dataclass(frozen=True, slots=True)
class TaskGraph:
    graph_id: str
    nodes: "tuple[TaskNode, ...]"

    def __post_init__(self) -> None:
        if not self.graph_id.strip():
            raise ValueError("task graph id is required")
        object.__setattr__(self, "nodes", tuple(self.nodes))
        ids = {node.node_id for node in self.nodes}
        if len(ids) != len(self.nodes) or any(
            dependency not in ids
            for node in self.nodes
            for dependency in node.dependencies
        ):
            raise TaskGraphValidationError(
                ErrorCode.TASK_DAG_INVALID,
                "task graph contains an unknown dependency",
                safe_details={"reason": "dependency_unknown"},
            )
        self._topological_order()

    def _topological_order(self) -> "tuple[str, ...]":
        indegree = {node.node_id: len(node.dependencies) for node in self.nodes}
        dependents: dict[str, list[str]] = {node.node_id: [] for node in self.nodes}
        for node in self.nodes:
            for dependency in node.dependencies:
                dependents[dependency].append(node.node_id)
        ready = [node_id for node_id, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)
        order: list[str] = []
        while ready:
            node_id = heapq.heappop(ready)
            order.append(node_id)
            for dependent in dependents[node_id]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    heapq.heappush(ready, dependent)
        if len(order) != len(self.nodes):
            raise TaskGraphValidationError(
                ErrorCode.TASK_DAG_INVALID,
                "task graph contains a cycle",
                safe_details={"reason": "cycle"},
            )
        return tuple(order)

    def validate_limits(self, limits: TaskGraphLimits) -> None:
        if len(self.nodes) > limits.max_nodes:
            raise AIError(ErrorCode.TASK_DAG_INVALID, "task graph exceeds node limit")
        nodes = {node.node_id: node for node in self.nodes}
        depths: dict[str, int] = {}
        for node_id in self._topological_order():
            node = nodes[node_id]
            depths[node_id] = 1 + max(
                (depths[item] for item in node.dependencies),
                default=0,
            )
        if max(depths.values(), default=0) > limits.max_depth:
            raise AIError(ErrorCode.TASK_DAG_INVALID, "task graph exceeds depth limit")
        if sum(node.budget_cost for node in self.nodes) > limits.max_budget:
            raise AIError(ErrorCode.TASK_DAG_INVALID, "task graph exceeds budget limit")


class TaskGraphValidationError(AIError, ValueError):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        safe_details: "Mapping[str, JsonValue] | None" = None,
    ) -> None:
        AIError.__init__(self, code, message, safe_details=safe_details)
        ValueError.__init__(self, message)


@dataclass(frozen=True, slots=True)
class TaskTerminalRecord:
    node_id: str
    owner: "str | None"
    fence: int
    status: TaskStatus
    result_digest: "str | None"
    error_code: "str | None"
    error_digest: "str | None"
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_id: "str | None" = None

    def __post_init__(self) -> None:
        if self.owner is not None:
            try:
                validate_lease_owner(self.owner)
            except AIError as error:
                raise ValueError("task terminal identity is invalid") from error
        if self.fence < 1:
            raise ValueError("task terminal fence is invalid")
        if self.completed_at.tzinfo is None:
            raise ValueError("task terminal time must be timezone-aware")


@dataclass(frozen=True, slots=True)
class TaskGraphRequest:
    graph: TaskGraph
    principal: Principal
    idempotency_key: str = ""
    limits: TaskGraphLimits = field(default_factory=TaskGraphLimits)
    correlation: CorrelationData = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)
        self.graph.validate_limits(self.limits)
        object.__setattr__(self, "correlation", normalize_correlation(self.correlation))


def _task_graph_request_digest(
    graph: TaskGraph,
    principal: Principal,
    limits: TaskGraphLimits,
) -> str:
    return canonical_sha256(
        {
            "principal": principal_identity_payload(principal),
            "graph_id": graph.graph_id,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "dependencies": sorted(node.dependencies),
                    "input": node.input,
                    "budget_cost": node.budget_cost,
                    "expander": (
                        None
                        if node.expander is None
                        else {
                            "id": node.expander.id,
                            "version": node.expander.version,
                        }
                    ),
                }
                for node in sorted(graph.nodes, key=lambda item: item.node_id)
            ],
            "limits": {
                "max_nodes": limits.max_nodes,
                "max_depth": limits.max_depth,
                "max_budget": limits.max_budget,
                "max_concurrency": limits.max_concurrency,
            },
        }
    )


@dataclass(frozen=True, slots=True)
class TaskGraphLaunch:
    graph_id: str
    principal: Principal
    limits: TaskGraphLimits
    correlation: CorrelationData = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.graph_id, str) or not self.graph_id.strip():
            raise ValueError("task graph id is required")
        object.__setattr__(self, "correlation", normalize_correlation(self.correlation))


@dataclass(frozen=True, slots=True)
class TaskGraphAdmission:
    version: int
    graph_id: str
    principal: Principal
    limits: TaskGraphLimits
    operation_id: str
    initial_request_digest: str
    correlation: CorrelationData = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.version, int)
            or isinstance(self.version, bool)
            or self.version < 1
            or not isinstance(self.graph_id, str)
            or not self.graph_id.strip()
            or re.fullmatch(r"[0-9a-f]{64}", self.operation_id) is None
            or re.fullmatch(r"[0-9a-f]{64}", self.initial_request_digest) is None
        ):
            raise ValueError("task graph admission is invalid")
        object.__setattr__(self, "correlation", normalize_correlation(self.correlation))

    @classmethod
    def from_request(cls, request: TaskGraphRequest) -> "TaskGraphAdmission":
        return cls(
            1,
            request.graph.graph_id,
            request.principal,
            request.limits,
            idempotency_key_digest(request.idempotency_key),
            _task_graph_request_digest(
                request.graph, request.principal, request.limits
            ),
            request.correlation,
        )

    def launch(self) -> TaskGraphLaunch:
        if self.version != 1:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        return TaskGraphLaunch(
            self.graph_id,
            self.principal,
            self.limits,
            self.correlation,
        )

    def validate_graph(self, graph: TaskGraph) -> None:
        if graph.graph_id != self.graph_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        graph.validate_limits(self.limits)
        if (
            _task_graph_request_digest(graph, self.principal, self.limits)
            != self.initial_request_digest
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


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
class TaskResultRecord:
    graph_id: str
    node_id: str
    result_digest: str
    payload: StoredPayload

    def __post_init__(self) -> None:
        if not isinstance(self.graph_id, str) or not self.graph_id.strip():
            raise ValueError("task result graph id is required")
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("task result node id is required")
        if re.fullmatch(r"[0-9a-f]{64}", self.result_digest) is None:
            raise ValueError("task result digest is invalid")
        if not isinstance(self.payload, StoredPayload):
            raise TypeError("task result payload is invalid")
        if self.payload.digest != self.result_digest:
            raise ValueError("task result payload digest does not match result")


@dataclass(frozen=True, slots=True)
class TaskDependencyResult:
    result_digest: str
    execution_id: "str | None" = None
    result_payload: "StoredPayload | None" = None

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.result_digest) is None:
            raise ValueError("task dependency result digest is invalid")
        if self.result_payload is not None:
            if not isinstance(self.result_payload, StoredPayload):
                raise TypeError("task dependency result payload is invalid")
            if self.result_payload.digest != self.result_digest:
                raise ValueError(
                    "task dependency result payload digest does not match result"
                )


@dataclass(frozen=True, slots=True)
class TaskGraphHandle:
    graph_id: str
    handle_id: str


@dataclass(frozen=True, slots=True)
class TaskGraphView:
    graph_id: str
    status: TaskStatus
    nodes: "tuple[TaskNode, ...]"


@dataclass(frozen=True, slots=True)
class TaskGraphSnapshot:
    graph_id: str
    status: TaskStatus
    nodes: "tuple[TaskNode, ...]"
    node_states: "tuple[TaskNodeView, ...]"

    def __post_init__(self) -> None:
        if not isinstance(self.graph_id, str) or not self.graph_id.strip():
            raise ValueError("task graph snapshot id is required")
        nodes = tuple(self.nodes)
        states = tuple(self.node_states)
        node_ids = tuple(node.node_id for node in nodes)
        state_ids = tuple(state.node_id for state in states)
        if len(set(node_ids)) != len(node_ids) or node_ids != state_ids:
            raise ValueError("task graph snapshot node set is invalid")
        for node, state in zip(nodes, states, strict=True):
            if state.graph_id != self.graph_id or state.dependencies != node.dependencies:
                raise ValueError("task graph snapshot node identity is invalid")
        aggregate = _aggregate_graph_status(states)
        if aggregate is not self.status:
            terminal = {
                TaskStatus.SUCCEEDED,
                TaskStatus.FAILED,
                TaskStatus.BLOCKED,
                TaskStatus.CANCELLED,
            }
            explicit_cancelled = (
                self.status is TaskStatus.CANCELLED
                and bool(states)
                and all(state.status in terminal for state in states)
                and any(state.status is TaskStatus.CANCELLED for state in states)
            )
            if not explicit_cancelled:
                raise ValueError("task graph snapshot aggregate status is invalid")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "node_states", states)


def _aggregate_graph_status(nodes: "tuple[TaskNodeView, ...]") -> TaskStatus:
    statuses = {node.status for node in nodes}
    if TaskStatus.RECOVERY_REQUIRED in statuses:
        return TaskStatus.RECOVERY_REQUIRED
    if not statuses or statuses <= {TaskStatus.SUCCEEDED}:
        return TaskStatus.SUCCEEDED
    if TaskStatus.RUNNING in statuses or TaskStatus.WAITING in statuses:
        return TaskStatus.RUNNING
    if TaskStatus.PENDING in statuses or TaskStatus.READY in statuses:
        return TaskStatus.PENDING
    if TaskStatus.FAILED in statuses:
        return TaskStatus.FAILED
    if TaskStatus.BLOCKED in statuses:
        return TaskStatus.BLOCKED
    if statuses <= {TaskStatus.CANCELLED, TaskStatus.SUCCEEDED}:
        return TaskStatus.CANCELLED
    raise ValueError("task graph aggregate status is invalid")


@dataclass(frozen=True, slots=True)
class CancelGraphRequest:
    principal: Principal
    idempotency_key: str
    force: bool = False

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class RecoverGraphRequest:
    principal: Principal
    idempotency_key: str

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)


def ready_nodes(
    graph: TaskGraph, completed: "frozenset[str]"
) -> "tuple[TaskNode, ...]":
    return tuple(
        node
        for node in graph.nodes
        if node.node_id not in completed
        and all(dependency in completed for dependency in node.dependencies)
    )


__all__ = [
    "CancelGraphRequest",
    "RecoverGraphRequest",
    "TaskDependencyResult",
    "TaskGraph",
    "TaskGraphAdmission",
    "TaskGraphHandle",
    "TaskGraphLaunch",
    "TaskGraphLimits",
    "TaskGraphRequest",
    "TaskGraphResult",
    "TaskGraphSnapshot",
    "TaskGraphValidationError",
    "TaskGraphView",
    "TaskLease",
    "TaskNode",
    "TaskExpanderRef",
    "TaskNodeResult",
    "TaskNodeView",
    "TaskResultRecord",
    "TaskStatus",
    "TaskTerminalRecord",
    "ready_nodes",
]
