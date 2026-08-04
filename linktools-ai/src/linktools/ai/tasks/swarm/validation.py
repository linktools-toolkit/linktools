#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure DAG checks for task_graph plans.

Construction-time validation (TaskPlan.__post_init__) already rejects malformed
graphs; these helpers give the engine a typed projection it can schedule
against without re-deriving the topology each loop iteration."""

from dataclasses import dataclass
from typing import Mapping

from ..models import TaskExecution, TaskNode, TaskPlan, TaskStatus
from ...errors import InvalidSpecError

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


_TERMINAL_STATUSES = frozenset(
    {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.SKIPPED,
        TaskStatus.CANCELLED,
    }
)


def validate_plan_against_swarm(
    plan: TaskPlan, *, allowed_agent_ids: "set[str]"
) -> None:
    """Every node's payload agent must be in the SwarmSpec.agents allow-set."""
    for node in plan.nodes:
        if node.payload.agent_id not in allowed_agent_ids:
            raise InvalidSpecError(
                f"plan {plan.id!r}: node {node.id!r} references agent "
                f"{node.payload.agent_id!r} not in the swarm"
            )


def node_lookup(plan: TaskPlan) -> "dict[str, TaskNode]":
    return {node.id: node for node in plan.nodes}


def all_terminal(executions: "Mapping[str, TaskExecution]") -> bool:
    return all(exec.status in _TERMINAL_STATUSES for exec in executions.values())


def terminal_statuses() -> "frozenset[TaskStatus]":
    return _TERMINAL_STATUSES


@dataclass(frozen=True, slots=True)
class Readiness:
    """Outcome of classifying one node against the current executions.

    READY: every dependency COMPLETED (or all unsucceeded edges
    proceed_degraded). SKIP: at least one unsucceeded edge is ``skip`` — the
    node should move straight to SKIPPED, recording the failed deps. Otherwise
    the node stays blocked (waiting on an in-flight dependency)."""

    ready: bool
    skip: bool = False
    blocked_by: "tuple[str, ...]" = ()


def classify_readiness(
    node: TaskNode, executions: "Mapping[str, TaskExecution]"
) -> Readiness:
    """Decide whether ``node`` should run, be skipped, or stay blocked.

    A dependency counts as resolved only once it is terminal. A non-terminal
    dependency (READY/CLAIMED, still in flight) blocks the node — it does NOT
    trigger skip/proceed_degraded. Once a dependency reaches a terminal
    non-COMPLETED status, its edge policy decides: any ``skip`` edge skips the
    node; only when every unsucceeded edge is ``proceed_degraded`` does the node
    run degraded. ``skip`` wins over ``proceed_degraded`` in a mixed set."""
    unsucceeded = []
    blocked = False
    for dep in node.dependencies:
        dep_exec = executions.get(dep.node_id)
        if dep_exec is None:
            continue
        if dep_exec.status is TaskStatus.COMPLETED:
            continue
        if dep_exec.status not in _TERMINAL_STATUSES:
            blocked = True
            continue
        unsucceeded.append((dep, dep_exec))
    if blocked:
        return Readiness(ready=False)
    if not unsucceeded:
        return Readiness(ready=True)
    if any(dep.on_failure.value == "skip" for dep, _ in unsucceeded):
        blocked_by = tuple(dep.node_id for dep, _ in unsucceeded)
        return Readiness(ready=False, skip=True, blocked_by=blocked_by)
    return Readiness(ready=True)


__all__ = [
    "Readiness",
    "all_terminal",
    "classify_readiness",
    "node_lookup",
    "terminal_statuses",
    "validate_plan_against_swarm",
]
