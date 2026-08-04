#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Swarm domain value types shared across strategies.

The task_graph strategy holds NO authoritative swarm-level state: the parent
RunRecord is the sole authority for the run and TaskExecution is the sole
authority for a node. Only the AgentRef member type and the strategy outcome
shapes live here."""

from dataclasses import dataclass, field
from typing import Mapping, TypeAlias

from ...json import JsonValue

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import TaskUsage
    from ...execution.domain import RunErrorInfo


@dataclass(frozen=True, slots=True)
class AgentRef:
    agent_id: str
    role: "str | None" = None


@dataclass(frozen=True, slots=True)
class SwarmRunView:
    """Read-only projection of a swarm run for inspect_swarm: the parent run's
    status/error plus the per-node TaskExecution snapshot, already authorized
    by the caller."""

    plan_id: str
    parent_run_id: str
    status: str
    error: "RunErrorInfo | None"
    nodes: "tuple[JsonValue, ...]"
    status_counts: "Mapping[str, int]"


@dataclass(frozen=True, slots=True)
class SwarmCompleted:
    """task_graph drove every node to a terminal status with no run-level
    error. ``collect`` is the structured projection of all nodes; ``usage`` is
    the aggregate across worker runs. Node-level FAILED/SKIPPED does NOT make
    the swarm FAILED."""

    collect: JsonValue
    usage: "TaskUsage"


@dataclass(frozen=True, slots=True)
class SwarmFailed:
    """task_graph hit a run-level error (plan/DAG/store/limit/usage/protocol or
    programming error). The redacted error is carried here."""

    error: "RunErrorInfo"


SwarmExecutionOutcome: TypeAlias = "SwarmCompleted | SwarmFailed"
"""Discriminated union returned by the task_graph executor so the runtime can
converge the parent RunRecord lifecycle from one object."""


__all__ = [
    "AgentRef",
    "SwarmCompleted",
    "SwarmExecutionOutcome",
    "SwarmFailed",
    "SwarmRunView",
]
