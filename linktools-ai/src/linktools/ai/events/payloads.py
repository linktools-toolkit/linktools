#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strongly-typed event payloads, per docs/linktools-ai.md section 23.2. Each
payload carries the minimum data meaningful for that event type -- the spec
mandates which payload TYPES must exist, not their exact fields."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Union


@dataclass(frozen=True, slots=True)
class RunStarted:
    run_id: str
    runnable_id: str


@dataclass(frozen=True, slots=True)
class RunCompleted:
    run_id: str
    result_summary: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunFailed:
    run_id: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class RunPaused:
    run_id: str
    reason: "str | None" = None


@dataclass(frozen=True, slots=True)
class RunResumed:
    run_id: str


@dataclass(frozen=True, slots=True)
class RunCancelled:
    run_id: str
    reason: "str | None" = None


@dataclass(frozen=True, slots=True)
class ModelStarted:
    model_type: str


@dataclass(frozen=True, slots=True)
class ModelCompleted:
    model_type: str
    token_usage: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelFailed:
    model_type: str
    error_message: str


@dataclass(frozen=True, slots=True)
class ToolStarted:
    tool_name: str
    tool_call_id: str


@dataclass(frozen=True, slots=True)
class ToolCompleted:
    tool_name: str
    tool_call_id: str
    success: bool


@dataclass(frozen=True, slots=True)
class ToolFailed:
    tool_name: str
    tool_call_id: str
    error_message: str


@dataclass(frozen=True, slots=True)
class ApprovalRequested:
    approval_id: str
    tool_name: str
    reason: str


@dataclass(frozen=True, slots=True)
class ApprovalApproved:
    approval_id: str
    resolved_by: "str | None" = None


@dataclass(frozen=True, slots=True)
class ApprovalRejected:
    approval_id: str
    resolved_by: "str | None" = None
    reason: "str | None" = None


@dataclass(frozen=True, slots=True)
class SwarmStarted:
    swarm_run_id: str
    swarm_id: str


@dataclass(frozen=True, slots=True)
class SwarmRoundStarted:
    swarm_run_id: str
    round: int


@dataclass(frozen=True, slots=True)
class SwarmRoundCompleted:
    swarm_run_id: str
    round: int


@dataclass(frozen=True, slots=True)
class SwarmTaskCreated:
    swarm_run_id: str
    task_id: str
    description: str


@dataclass(frozen=True, slots=True)
class SwarmTaskClaimed:
    swarm_run_id: str
    task_id: str
    assigned_agent_id: str


@dataclass(frozen=True, slots=True)
class SwarmTaskCompleted:
    swarm_run_id: str
    task_id: str


@dataclass(frozen=True, slots=True)
class SwarmTaskFailed:
    swarm_run_id: str
    task_id: str
    error_message: str


@dataclass(frozen=True, slots=True)
class SwarmCompleted:
    swarm_run_id: str


@dataclass(frozen=True, slots=True)
class ResourceChanged:
    path: str
    revision: int


# Union of every event payload type. This is the type of the ``payload`` field
# EventStore.append accepts (review doc §8.3) -- callers pass a concrete
# payload instance and the store wraps it in an EventEnvelope.
EventPayload = Union[
    RunStarted,
    RunCompleted,
    RunFailed,
    RunPaused,
    RunResumed,
    RunCancelled,
    ModelStarted,
    ModelCompleted,
    ModelFailed,
    ToolStarted,
    ToolCompleted,
    ToolFailed,
    ApprovalRequested,
    ApprovalApproved,
    ApprovalRejected,
    SwarmStarted,
    SwarmRoundStarted,
    SwarmRoundCompleted,
    SwarmTaskCreated,
    SwarmTaskClaimed,
    SwarmTaskCompleted,
    SwarmTaskFailed,
    SwarmCompleted,
    ResourceChanged,
]
