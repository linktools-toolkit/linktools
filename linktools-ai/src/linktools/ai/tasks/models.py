#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Immutable task-plan and task-execution values for the task_graph swarm."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from ..errors import InvalidSpecError
from ..json import normalize_json
from ..storage.coordination.lease import Lease

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..execution.domain import RunErrorInfo
    from ..json import JsonValue


class DependencyFailurePolicy(StrEnum):
    SKIP = "skip"
    PROCEED_DEGRADED = "proceed_degraded"


class TaskStatus(StrEnum):
    READY = "ready"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


TERMINAL_TASK_STATUSES: "frozenset[TaskStatus]" = frozenset(
    {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.SKIPPED,
        TaskStatus.CANCELLED,
    }
)

ALLOWED_TASK_TRANSITIONS: "Mapping[TaskStatus, frozenset[TaskStatus]]" = {
    TaskStatus.READY: frozenset(
        {TaskStatus.CLAIMED, TaskStatus.SKIPPED, TaskStatus.CANCELLED}
    ),
    TaskStatus.CLAIMED: frozenset(
        {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
    ),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.SKIPPED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class TaskDependency:
    node_id: str
    on_failure: DependencyFailurePolicy = DependencyFailurePolicy.SKIP


@dataclass(frozen=True, slots=True)
class TaskGraphNodePayload:
    agent_id: str
    prompt: str
    metadata: "Mapping[str, JsonValue]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskNode:
    id: str
    payload: TaskGraphNodePayload
    dependencies: "tuple[TaskDependency, ...]" = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise InvalidSpecError("task node id must not be empty")
        if not self.payload.agent_id:
            raise InvalidSpecError(f"node {self.id!r}: agent_id must not be empty")
        seen: "set[str]" = set()
        for dep in self.dependencies:
            if not dep.node_id:
                raise InvalidSpecError(f"node {self.id!r}: dependency node_id empty")
            if dep.node_id == self.id:
                raise InvalidSpecError(f"node {self.id!r}: self dependency")
            if dep.node_id in seen:
                raise InvalidSpecError(
                    f"node {self.id!r}: duplicate dependency {dep.node_id!r}"
                )
            seen.add(dep.node_id)
        object.__setattr__(self, "payload", _freeze_payload(self.payload))


@dataclass(frozen=True, slots=True)
class TaskPlan:
    id: str
    nodes: "tuple[TaskNode, ...]"

    def __post_init__(self) -> None:
        if not self.id:
            raise InvalidSpecError("task plan id must not be empty")
        node_ids: "dict[str, TaskNode]" = {}
        agent_ids: "set[str]" = set()
        for node in self.nodes:
            if node.id in node_ids:
                raise InvalidSpecError(f"plan {self.id!r}: duplicate node {node.id!r}")
            node_ids[node.id] = node
            if node.payload.agent_id in agent_ids:
                raise InvalidSpecError(
                    f"plan {self.id!r}: agent {node.payload.agent_id!r} appears twice"
                )
            agent_ids.add(node.payload.agent_id)
        for node in self.nodes:
            for dep in node.dependencies:
                if dep.node_id not in node_ids:
                    raise InvalidSpecError(
                        f"plan {self.id!r}: node {node.id!r} depends on missing "
                        f"node {dep.node_id!r}"
                    )
        _assert_no_cycle(self.id, self.nodes)
        for node in self.nodes:
            normalize_json(node.payload, path=f"plan.{self.id}.{node.id}.payload")


def _assert_no_cycle(plan_id: str, nodes: "tuple[TaskNode, ...]") -> None:
    adjacency: "dict[str, tuple[str, ...]]" = {
        node.id: tuple(d.node_id for d in node.dependencies) for node in nodes
    }
    state: "dict[str, int]" = {}  # 0=visiting, 1=done

    def visit(node_id: str, path: "tuple[str, ...]") -> None:
        marker = state.get(node_id)
        if marker == 1:
            return
        if marker == 0:
            chain = " -> ".join(path + (node_id,))
            raise InvalidSpecError(f"plan {plan_id!r}: cycle detected: {chain}")
        state[node_id] = 0
        for dep_id in adjacency[node_id]:
            visit(dep_id, path + (node_id,))
        state[node_id] = 1

    for node in nodes:
        visit(node.id, ())


def _freeze_payload(payload: TaskGraphNodePayload) -> TaskGraphNodePayload:
    return TaskGraphNodePayload(
        agent_id=payload.agent_id,
        prompt=payload.prompt,
        metadata=MappingProxyType(dict(payload.metadata)),
    )


@dataclass(frozen=True, slots=True)
class TaskUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: "Decimal | None" = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.input_tokens, int)
            or isinstance(self.input_tokens, bool)
            or self.input_tokens < 0
        ):
            raise ValueError("input_tokens must be a non-negative int")
        if (
            not isinstance(self.output_tokens, int)
            or isinstance(self.output_tokens, bool)
            or self.output_tokens < 0
        ):
            raise ValueError("output_tokens must be a non-negative int")
        if self.total_cost is not None:
            if not isinstance(self.total_cost, Decimal):
                raise ValueError("total_cost must be a Decimal or None")
            if not self.total_cost.is_finite() or self.total_cost < 0:
                raise ValueError("total_cost must be finite and non-negative")

    def add(self, other: "TaskUsage") -> "TaskUsage":
        # Cost is sticky-unknown: if either contribution cannot report cost, the
        # sum is unknown (None). This matches the task_graph rule that a cost cap
        # is only enforceable when every agent reports cost.
        if self.total_cost is None or other.total_cost is None:
            cost: "Decimal | None" = None
        else:
            cost = self.total_cost + other.total_cost
        return TaskUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_cost=cost,
        )

    @property
    def cost_known(self) -> bool:
        return self.total_cost is not None


@dataclass(frozen=True, slots=True)
class TaskExecution:
    id: str
    plan_id: str
    node_id: str
    status: TaskStatus
    lease: Lease = field(default_factory=Lease)
    attempt: int = 0
    active_run_id: "str | None" = None
    result: "JsonValue | None" = None
    error: "RunErrorInfo | None" = None
    blocked_by: "tuple[str, ...]" = ()
    terminal_reason: "str | None" = None
    usage: TaskUsage = field(default_factory=TaskUsage)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            object.__setattr__(self, "status", TaskStatus(self.status))
        status = self.status
        if status is TaskStatus.READY:
            if self.attempt != 0:
                raise ValueError("READY execution must have attempt == 0")
            if self.active_run_id is not None or self.result is not None:
                raise ValueError("READY execution must not carry a result or run")
            if self.error is not None or self.blocked_by:
                raise ValueError("READY execution must not carry error or blocked_by")
        elif status is TaskStatus.CLAIMED:
            if self.attempt != 1:
                raise ValueError("CLAIMED execution must have attempt == 1")
            if self.lease.owner is None or self.lease.expires_at is None:
                raise ValueError("CLAIMED execution must hold an active lease")
        elif status is TaskStatus.COMPLETED:
            if self.active_run_id is None:
                raise ValueError("COMPLETED execution must reference its child run")
            if self.error is not None or self.blocked_by:
                raise ValueError("COMPLETED execution must not carry error/blocked_by")
        elif status is TaskStatus.FAILED:
            if self.error is None:
                raise ValueError("FAILED execution must carry a structured error")
        elif status is TaskStatus.SKIPPED:
            if not self.blocked_by:
                raise ValueError("SKIPPED execution must carry non-empty blocked_by")
            if self.active_run_id is not None:
                raise ValueError("SKIPPED execution must not reference a child run")
        elif status is TaskStatus.CANCELLED:
            if not self.terminal_reason:
                raise ValueError("CANCELLED execution must carry a terminal_reason")
        else:  # pragma: no cover - exhaustiveness guard
            raise ValueError(f"unknown task status: {status!r}")

    @property
    def owner(self) -> "str | None":
        return self.lease.owner

    @property
    def fence(self) -> int:
        return self.lease.fence

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_TASK_STATUSES


__all__ = [
    "ALLOWED_TASK_TRANSITIONS",
    "DependencyFailurePolicy",
    "TERMINAL_TASK_STATUSES",
    "TaskDependency",
    "TaskExecution",
    "TaskGraphNodePayload",
    "TaskNode",
    "TaskPlan",
    "TaskStatus",
    "TaskUsage",
]
