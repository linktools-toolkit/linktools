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
from types import MappingProxyType

from ..core import (
    RunBudget,
    ImmutableJsonMapping,
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
from ..errors import ErrorDiagnostics
from ._capture import TaskInvocationInputRef
from ._definitions import Task, TaskExpander, TaskExpanderRef, TaskRef


def normalize_timeout_seconds(value: object) -> "float | None":
    """Normalize one finite positive task timeout."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("task timeout must be a finite positive number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError("task timeout must be a finite positive number")
    return normalized


def normalize_retry_delay_seconds(value: object) -> float:
    """Normalize one finite non-negative task retry delay."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("task retry delay must be a finite non-negative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError("task retry delay must be a finite non-negative number")
    return 0.0 if normalized == 0 else normalized


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


_RESULT_DIGEST = re.compile(r"[0-9a-f]{64}")
_TASK_DEPENDENCY_POLICIES = frozenset(
    {"all_succeeded", "all_terminal", "any_succeeded"}
)
_TASK_FAILURE_POLICIES = frozenset({"propagate", "isolate"})
_TERMINAL_TASK_STATUSES = frozenset(
    {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.BLOCKED,
        TaskStatus.CANCELLED,
    }
)


@dataclass(frozen=True, slots=True)
class TaskResultRef:
    """Stable reference to a retained successful Task result."""

    namespace: str
    tenant_id: str
    graph_id: str
    node_id: str
    result_digest: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (
                self.namespace,
                self.tenant_id,
                self.graph_id,
                self.node_id,
            )
        ) or _RESULT_DIGEST.fullmatch(self.result_digest) is None:
            raise ValueError("task result reference is invalid")


@dataclass(frozen=True, slots=True)
class TaskNodeResultRef:
    """An explicit result input from a scheduling dependency in this graph."""

    node_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("task node result reference is invalid")


@dataclass(frozen=True, slots=True, init=False)
class TaskNode:
    node_id: str
    dependencies: tuple[str, ...]
    task: "TaskRef | None"
    budget_cost: int
    expander: "TaskExpanderRef | None"
    input_refs: "Mapping[str, TaskResultRef | TaskNodeResultRef]"
    input_capture: "TaskInvocationInputRef | None"
    original_input: "Mapping[str, JsonValue] | None" = field(repr=False)
    timeout_seconds: "float | None"
    max_attempts: int
    retry_delay_seconds: float
    output_type: object | None
    output_contract: "Mapping[str, JsonValue] | None"
    effect_policy: str
    reconcile: bool
    dependency_policy: str
    failure_policy: str
    _input: bytes = field(repr=False)

    def __init__(
        self,
        node_id: str,
        dependencies: "tuple[str, ...]" = (),
        *,
        task: "Task | TaskRef | None" = None,
        input: "Mapping[str, JsonValue] | None" = None,
        budget_cost: int = 1,
        expander: "TaskExpander | TaskExpanderRef | None" = None,
        input_refs: "Mapping[str, TaskResultRef | TaskNodeResultRef] | None" = None,
        input_capture: "TaskInvocationInputRef | None" = None,
        timeout_seconds: "float | None" = None,
        max_attempts: int = 1,
        retry_delay_seconds: float = 0,
        output_type: object | None = None,
        dependency_policy: str = "all_succeeded",
        failure_policy: str = "propagate",
    ) -> None:
        self._initialize(
            node_id,
            dependencies,
            task=task,
            input=input,
            original_input=None,
            budget_cost=budget_cost,
            expander=expander,
            input_refs=input_refs,
            input_capture=input_capture,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
            output_type=output_type,
            output_contract=None,
            effect_policy="none",
            reconcile=False,
            dependency_policy=dependency_policy,
            failure_policy=failure_policy,
        )

    @classmethod
    def from_resolved(
        cls,
        node_id: str,
        dependencies: "tuple[str, ...]" = (),
        *,
        task: "TaskRef | None",
        input: "Mapping[str, JsonValue] | None" = None,
        original_input: "Mapping[str, JsonValue] | None" = None,
        budget_cost: int = 1,
        expander: "TaskExpanderRef | None" = None,
        input_refs: "Mapping[str, TaskResultRef | TaskNodeResultRef] | None" = None,
        input_capture: "TaskInvocationInputRef | None" = None,
        timeout_seconds: "float | None" = None,
        max_attempts: int = 1,
        retry_delay_seconds: float = 0,
        output_contract: "Mapping[str, JsonValue] | None" = None,
        effect_policy: str = "none",
        reconcile: bool = False,
        dependency_policy: str = "all_succeeded",
        failure_policy: str = "propagate",
    ) -> "TaskNode":
        """Build a node whose execution contract has already been resolved."""
        value = cls.__new__(cls)
        value._initialize(
            node_id,
            dependencies,
            task=task,
            input=input,
            original_input=original_input,
            budget_cost=budget_cost,
            expander=expander,
            input_refs=input_refs,
            input_capture=input_capture,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
            output_type=None,
            output_contract=output_contract,
            effect_policy=effect_policy,
            reconcile=reconcile,
            dependency_policy=dependency_policy,
            failure_policy=failure_policy,
        )
        return value

    def _initialize(
        self,
        node_id: str,
        dependencies: "tuple[str, ...]",
        *,
        task: "Task | TaskRef | None",
        input: "Mapping[str, JsonValue] | None",
        original_input: "Mapping[str, JsonValue] | None",
        budget_cost: int,
        expander: "TaskExpander | TaskExpanderRef | None",
        input_refs: "Mapping[str, TaskResultRef | TaskNodeResultRef] | None",
        input_capture: "TaskInvocationInputRef | None",
        timeout_seconds: "float | None",
        max_attempts: int,
        retry_delay_seconds: float,
        output_type: object | None,
        output_contract: "Mapping[str, JsonValue] | None",
        effect_policy: str,
        reconcile: bool,
        dependency_policy: str,
        failure_policy: str,
    ) -> None:
        if isinstance(dependencies, (str, bytes)):
            raise TypeError("task node dependencies are invalid")
        try:
            normalized_dependencies = tuple(dependencies)
        except TypeError as error:
            raise TypeError("task node dependencies are invalid") from error
        normalized_timeout = normalize_timeout_seconds(timeout_seconds)
        normalized_retry_delay = normalize_retry_delay_seconds(retry_delay_seconds)
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
            or (
                expander is not None
                and not isinstance(expander, (TaskExpander, TaskExpanderRef))
            )
            or isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or max_attempts < 1
            or effect_policy not in {"none", "replay_safe", "non_replay_safe"}
            or not isinstance(reconcile, bool)
            or not isinstance(dependency_policy, str)
            or dependency_policy not in _TASK_DEPENDENCY_POLICIES
            or not isinstance(failure_policy, str)
            or failure_policy not in _TASK_FAILURE_POLICIES
        ):
            raise ValueError("task node identity is invalid")
        values: Mapping[str, JsonValue] = {} if input is None else input
        if not isinstance(values, Mapping):
            raise TypeError("task node input must be a mapping")
        normalized = _normalize_json_mapping(values)
        if task is None:
            task_ref = None
        elif isinstance(task, Task):
            task_ref = task.ref
        elif isinstance(task, TaskRef):
            task_ref = task
        else:
            raise TypeError("task must be Task or TaskRef")
        if isinstance(expander, TaskExpander):
            expander_ref = expander.ref
        else:
            expander_ref = expander
        references = {} if input_refs is None else dict(input_refs)
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(reference, (TaskResultRef, TaskNodeResultRef))
            for name, reference in references.items()
        ):
            raise ValueError("task node input references are invalid")
        for name, reference in references.items():
            if isinstance(reference, TaskNodeResultRef):
                if reference.node_id not in normalized_dependencies:
                    raise ValueError("symbolic result must name a scheduling dependency")
                if name in normalized_dependencies and name != reference.node_id:
                    raise ValueError("task input alias conflicts with a dependency")
            elif name in normalized_dependencies:
                raise ValueError("task node input reference names conflict with dependencies")
        if input_capture is not None:
            if not isinstance(input_capture, TaskInvocationInputRef):
                raise TypeError("task input capture is invalid")
            if normalized:
                raise ValueError("task input and input capture are mutually exclusive")
        contract = None
        if output_contract is not None:
            normalized_contract = _normalize_json_value(dict(output_contract))
            if not isinstance(normalized_contract, dict):
                raise ValueError("task node output contract is invalid")
            contract = ImmutableJsonMapping(normalized_contract)
        if output_type is not None and contract is not None:
            raise ValueError("task node cannot contain both output type and contract")
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "dependencies", normalized_dependencies)
        object.__setattr__(self, "task", task_ref)
        object.__setattr__(self, "budget_cost", budget_cost)
        object.__setattr__(self, "expander", expander_ref)
        object.__setattr__(self, "input_refs", MappingProxyType(references))
        object.__setattr__(self, "input_capture", input_capture)
        object.__setattr__(self, "original_input", None if original_input is None else ImmutableJsonMapping(original_input))
        object.__setattr__(self, "timeout_seconds", normalized_timeout)
        object.__setattr__(self, "max_attempts", max_attempts)
        object.__setattr__(self, "retry_delay_seconds", normalized_retry_delay)
        object.__setattr__(self, "output_type", output_type)
        object.__setattr__(self, "output_contract", contract)
        object.__setattr__(self, "effect_policy", effect_policy)
        object.__setattr__(self, "reconcile", reconcile)
        object.__setattr__(self, "dependency_policy", dependency_policy)
        object.__setattr__(self, "failure_policy", failure_policy)
        object.__setattr__(self, "_input", canonical_json_bytes(normalized))

    def to_mapping(self) -> dict[str, JsonValue]:
        """Project the resolved node semantics used by TaskGraph identity."""
        node_input = dict(self.input)
        prompt = node_input.get("prompt")
        if (
            self.task is not None
            and node_input.get("kind") == "agent-task-input"
            and isinstance(prompt, Mapping)
        ):
            if prompt.get("kind") == "stored-user-content-v1":
                intent_digest = prompt.get("source_intent_digest")
            else:
                intent_digest = canonical_sha256(prompt)
            if isinstance(intent_digest, str):
                node_input["prompt"] = {
                    "kind": "task-prompt-intent-v1",
                    "digest": intent_digest,
                }
        value: dict[str, JsonValue] = {
            "node_id": self.node_id,
            "dependencies": sorted(self.dependencies),
            "input": node_input,
            "budget_cost": self.budget_cost,
            "dependency_policy": self.dependency_policy,
            "failure_policy": self.failure_policy,
            "task": (
                None
                if self.task is None
                else {"id": self.task.id, "revision": self.task.revision}
            ),
            "expander": (
                None
                if self.expander is None
                else {
                    "id": self.expander.id,
                    "revision": self.expander.revision,
                }
            ),
        }
        if self.input_refs:
            value["input_refs"] = {
                name: {"node_id": reference.node_id} if isinstance(reference, TaskNodeResultRef) else {
                    "namespace": reference.namespace,
                    "tenant_id": reference.tenant_id,
                    "graph_id": reference.graph_id,
                    "node_id": reference.node_id,
                    "result_digest": reference.result_digest,
                }
                for name, reference in sorted(self.input_refs.items())
            }
        if self.original_input is not None:
            value["original_input"] = dict(self.original_input)
        if self.input_capture is not None:
            reference = self.input_capture
            value["input_capture"] = {
                "namespace": reference.namespace,
                "tenant_id": reference.tenant_id,
                "capture_id": reference.capture_id,
                "digest": reference.digest,
                "source_execution_id": reference.source_execution_id,
            }
        if self.timeout_seconds is not None:
            value["timeout_seconds"] = self.timeout_seconds
        if self.max_attempts != 1:
            value["max_attempts"] = self.max_attempts
        if self.retry_delay_seconds != 0:
            value["retry_delay_seconds"] = self.retry_delay_seconds
        if self.output_contract is not None:
            value["output_contract"] = dict(self.output_contract)
        if self.effect_policy != "none":
            value["effect_policy"] = self.effect_policy
        if self.reconcile:
            value["reconcile"] = True
        return value

    def dependency_status(self, dependency_states: Mapping[str, TaskStatus]) -> TaskStatus:
        """Return the dependency-derived state without changing persisted state."""
        statuses = tuple(dependency_states[dependency] for dependency in self.dependencies)
        if any(not isinstance(status, TaskStatus) for status in statuses):
            raise ValueError("task dependency status is invalid")
        if self.dependency_policy == "all_succeeded":
            if any(
                status in _TERMINAL_TASK_STATUSES
                and status is not TaskStatus.SUCCEEDED
                for status in statuses
            ):
                return TaskStatus.BLOCKED
            if all(status is TaskStatus.SUCCEEDED for status in statuses):
                return TaskStatus.READY
            return TaskStatus.PENDING
        if self.dependency_policy == "all_terminal":
            return (
                TaskStatus.READY
                if all(status in _TERMINAL_TASK_STATUSES for status in statuses)
                else TaskStatus.PENDING
            )
        if self.dependency_policy == "any_succeeded":
            if not all(status in _TERMINAL_TASK_STATUSES for status in statuses):
                return TaskStatus.PENDING
            return (
                TaskStatus.READY
                if any(status is TaskStatus.SUCCEEDED for status in statuses)
                else TaskStatus.BLOCKED
            )
        raise ValueError("task node dependency policy is invalid")

    @property
    def input(self) -> "dict[str, JsonValue]":
        return json.loads(self._input.decode("utf-8"))

    @classmethod
    def wait(
        cls,
        node_id: str,
        *,
        dependencies: "tuple[str, ...]" = (),
        input: "Mapping[str, JsonValue] | None" = None,
        output_type: object | None = None,
    ) -> "TaskNode":
        """Declare a node whose value is supplied through the Runtime API."""
        values = {} if input is None else dict(input)
        return cls(
            node_id,
            dependencies,
            task=TaskRef.deferred_input(),
            input=values,
            output_type=output_type,
        )


@dataclass(frozen=True, slots=True)
class TaskLease:
    graph_id: str
    node_id: str
    tenant_id: str
    owner: str
    fence: int
    lease_expires_at: datetime
    execution_id: "str | None" = None

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
            or (
                self.execution_id is not None
                and (
                    not isinstance(self.execution_id, str)
                    or not self.execution_id.strip()
                )
            )
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
    next_attempt_at: "datetime | None" = None
    occupies_concurrency: bool = False
    error_origin: "str | None" = None
    safe_error_details: "Mapping[str, JsonValue]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.owner is not None:
            try:
                validate_lease_owner(self.owner)
            except AIError as error:
                raise ValueError("task node lease owner is invalid") from error
        if self.status is TaskStatus.PENDING and self.execution_id is not None:
            raise ValueError("pending task node cannot carry an execution id")
        if not isinstance(self.occupies_concurrency, bool):
            raise TypeError("task concurrency projection must be bool")
        if self.error_origin not in {None, "node", "execution"}:
            raise ValueError("task node failure origin is invalid")
        try:
            details = ImmutableJsonMapping(self.safe_error_details)
        except (TypeError, ValueError) as error:
            raise ValueError("task node safe error details are invalid") from error
        object.__setattr__(self, "safe_error_details", details)
        if self.occupies_concurrency and self.status is not TaskStatus.WAITING:
            raise ValueError("only waiting task can retain concurrency capacity")
        if self.next_attempt_at is not None and (
            self.status is not TaskStatus.READY
            or self.execution_id is None
            or self.next_attempt_at.tzinfo is None
        ):
            raise ValueError("task retry schedule is invalid")
        if self.status is not TaskStatus.READY and self.next_attempt_at is not None:
            raise ValueError("only ready task can carry a retry schedule")
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
        self.topological_order()

    def topological_order(self) -> "tuple[str, ...]":
        """Return the deterministic node order used to validate this DAG."""
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
        for node_id in self.topological_order():
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
    budget: RunBudget | None = None

    def __post_init__(self) -> None:
        if self.budget is not None and not isinstance(self.budget, RunBudget):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        validate_idempotency_key(self.idempotency_key)
        self.graph.validate_limits(self.limits)
        object.__setattr__(self, "correlation", normalize_correlation(self.correlation))


def _task_graph_request_digest(
    graph: TaskGraph,
    principal: Principal,
    limits: TaskGraphLimits,
    budget: RunBudget | None = None,
) -> str:
    return canonical_sha256(
        {
            "principal": principal_identity_payload(principal),
            "graph_id": graph.graph_id,
            **({"budget": budget.digest_payload()} if budget is not None else {}),
            "nodes": [
                node.to_mapping()
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
    budget: RunBudget | None = None

    def __post_init__(self) -> None:
        if self.budget is not None and not isinstance(self.budget, RunBudget):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
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
    budget: RunBudget | None = None
    budget_scope_id: str | None = None

    def __post_init__(self) -> None:
        if self.budget is not None and not isinstance(self.budget, RunBudget):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
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
        expected_scope = None if self.budget is None else "graph:" + self.graph_id
        if self.budget_scope_id != expected_scope:
            raise ValueError("task graph budget scope is invalid")
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
                request.graph, request.principal, request.limits, request.budget
            ),
            request.correlation,
            request.budget,
            None if request.budget is None else "graph:" + request.graph.graph_id,
        )

    def launch(self) -> TaskGraphLaunch:
        if self.version != 1:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        return TaskGraphLaunch(
            self.graph_id,
            self.principal,
            self.limits,
            self.correlation,
            self.budget,
        )

    def validate_graph(self, graph: TaskGraph) -> None:
        if graph.graph_id != self.graph_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        graph.validate_limits(self.limits)
        if (
            _task_graph_request_digest(graph, self.principal, self.limits, self.budget)
            != self.initial_request_digest
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


@dataclass(frozen=True, slots=True)
class TaskGraphResult:
    graph_id: str
    status: TaskStatus
    node_results: "tuple[TaskNodeResult, ...]" = ()

    @property
    def wait_status(self) -> TaskStatus:
        return _wait_status(self.status, self.node_results)


@dataclass(frozen=True, slots=True)
class TaskNodeResult:
    node_id: str
    status: TaskStatus
    result_digest: "str | None"
    execution_id: "str | None"
    error_code: "str | None"
    error_digest: "str | None"
    result_ref: "TaskResultRef | None" = None
    safe_error_details: "Mapping[str, JsonValue]" = field(default_factory=dict)
    error_diagnostics: "ErrorDiagnostics | None" = None
    output: "JsonValue | None" = None
    content_included: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("task node result id is invalid")
        try:
            details = ImmutableJsonMapping(self.safe_error_details)
        except (TypeError, ValueError) as error:
            raise ValueError("task node result error details are invalid") from error
        if not isinstance(self.content_included, bool):
            raise TypeError("task node content flag must be bool")
        if self.error_diagnostics is not None and not isinstance(
            self.error_diagnostics,
            ErrorDiagnostics,
        ):
            raise ValueError("task node error diagnostics are invalid")
        if self.result_ref is not None and (
            self.status is not TaskStatus.SUCCEEDED
            or self.result_digest != self.result_ref.result_digest
        ):
            raise ValueError("task node result reference is inconsistent")
        if self.content_included:
            if self.status is not TaskStatus.SUCCEEDED:
                raise ValueError("failed task node cannot include output")
            object.__setattr__(self, "output", _normalize_json_value(self.output))
        elif self.output is not None:
            raise ValueError("omitted task output must be None")
        object.__setattr__(self, "safe_error_details", details)


@dataclass(frozen=True, slots=True)
class TaskResultRecord:
    graph_id: str
    node_id: str
    result_digest: str
    execution_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.graph_id, str) or not self.graph_id.strip():
            raise ValueError("task result graph id is required")
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("task result node id is required")
        if re.fullmatch(r"[0-9a-f]{64}", self.result_digest) is None:
            raise ValueError("task result digest is invalid")
        if not isinstance(self.execution_id, str) or not self.execution_id.strip():
            raise ValueError("task result execution id is invalid")


@dataclass(frozen=True, slots=True)
class TaskDependencyResult:
    result_digest: str
    execution_id: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.result_digest) is None:
            raise ValueError("task dependency result digest is invalid")
        if not isinstance(self.execution_id, str) or not self.execution_id.strip():
            raise ValueError("task dependency execution id is required")


@dataclass(frozen=True, slots=True)
class TaskGraphHandle:
    graph_id: str


@dataclass(frozen=True, slots=True)
class TaskGraphView:
    graph_id: str
    status: TaskStatus
    nodes: "tuple[TaskNode, ...]"


@dataclass(frozen=True, slots=True)
class TaskNodeInfo:
    """Safe public task-node metadata without raw input content."""

    node_id: str
    task: "TaskRef | None"
    dependencies: tuple[str, ...]
    budget_cost: int
    expander: "TaskExpanderRef | None"
    input_refs: "Mapping[str, TaskResultRef | TaskNodeResultRef]"
    input_capture: "TaskInvocationInputRef | None"
    timeout_seconds: "float | None"
    max_attempts: int
    retry_delay_seconds: float
    output_contract: "Mapping[str, JsonValue] | None"
    effect_policy: str
    dependency_policy: str = "all_succeeded"
    reconcile: bool = False
    failure_policy: str = "propagate"

    @classmethod
    def from_node(cls, node: TaskNode) -> "TaskNodeInfo":
        return cls(
            node.node_id,
            node.task,
            node.dependencies,
            node.budget_cost,
            node.expander,
            node.input_refs,
            node.input_capture,
            node.timeout_seconds,
            node.max_attempts,
            node.retry_delay_seconds,
            node.output_contract,
            node.effect_policy,
            node.dependency_policy,
            reconcile=node.reconcile,
            failure_policy=node.failure_policy,
        )


@dataclass(frozen=True, slots=True)
class TaskGraphInfo:
    """Safe public task graph view without raw node inputs."""

    graph_id: str
    status: TaskStatus
    nodes: tuple[TaskNodeInfo, ...]
    node_states: tuple[TaskNodeView, ...]
    event_seq: int = 0

    @property
    def wait_status(self) -> TaskStatus:
        return _wait_status(self.status, self.node_states)


    @classmethod
    def from_state(cls, state: "TaskGraphState") -> "TaskGraphInfo":
        return cls(
            state.graph_id,
            state.status,
            tuple(TaskNodeInfo.from_node(node) for node in state.nodes),
            state.node_states,
            state.event_seq,
        )


@dataclass(frozen=True, slots=True)
class TaskGraphState:
    graph_id: str
    status: TaskStatus
    nodes: "tuple[TaskNode, ...]"
    node_states: "tuple[TaskNodeView, ...]"
    event_seq: int = 0

    @property
    def wait_status(self) -> TaskStatus:
        return _wait_status(self.status, self.node_states)


    def __post_init__(self) -> None:
        if not isinstance(self.graph_id, str) or not self.graph_id.strip():
            raise ValueError("task graph state id is required")
        if not isinstance(self.status, TaskStatus):
            raise ValueError("task graph state status is invalid")
        if (
            isinstance(self.event_seq, bool)
            or not isinstance(self.event_seq, int)
            or self.event_seq < 0
        ):
            raise ValueError("task graph state event sequence is invalid")
        nodes = tuple(self.nodes)
        states = tuple(self.node_states)
        node_ids = tuple(node.node_id for node in nodes)
        state_ids = tuple(state.node_id for state in states)
        if len(set(node_ids)) != len(node_ids) or node_ids != state_ids:
            raise ValueError("task graph state node set is invalid")
        for node, state in zip(nodes, states, strict=True):
            if (
                state.graph_id != self.graph_id
                or state.dependencies != node.dependencies
            ):
                raise ValueError("task graph state node identity is invalid")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "node_states", states)


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


@dataclass(frozen=True, slots=True)
class TaskInputSupplyRequest:
    principal: Principal
    wait_id: str
    value: JsonValue
    idempotency_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.wait_id, str) or not self.wait_id.strip():
            raise ValueError("task wait id is required")
        object.__setattr__(self, "value", _normalize_json_value(self.value))
        validate_idempotency_key(self.idempotency_key)


__all__ = [
    "CancelGraphRequest",
    "RecoverGraphRequest",
    "TaskDependencyResult",
    "TaskGraph",
    "TaskGraphAdmission",
    "TaskGraphHandle",
    "TaskGraphInfo",
    "TaskGraphLaunch",
    "TaskGraphLimits",
    "TaskGraphRequest",
    "TaskGraphResult",
    "TaskGraphState",
    "TaskGraphValidationError",
    "TaskGraphView",
    "TaskLease",
    "TaskNode",
    "TaskNodeInfo",
    "TaskExpanderRef",
    "TaskNodeResult",
    "TaskNodeView",
    "TaskResultRecord",
    "TaskResultRef",
    "TaskNodeResultRef",
    "TaskInputSupplyRequest",
    "TaskStatus",
    "TaskTerminalRecord",
]


def _wait_status(
    status: TaskStatus,
    states: tuple[TaskNodeView | TaskNodeResult, ...] = (),
) -> TaskStatus:
    if status is not TaskStatus.RUNNING:
        return status
    unfinished = tuple(
        state
        for state in states
        if state.status
        not in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
            TaskStatus.CANCELLED,
        }
    )
    if unfinished and any(
        state.status is TaskStatus.WAITING
        for state in unfinished
    ) and all(
        state.status
        not in {TaskStatus.READY, TaskStatus.RUNNING}
        for state in unfinished
    ):
        return TaskStatus.WAITING
    return status
