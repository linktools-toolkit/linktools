"""Validated commands accepted by the execution lifecycle port."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from ..json import JsonValue
from .domain import ApprovalDecision, RunApproval, RunDefinition, RunError, RunKind
from .snapshots import AgentSnapshotData


@dataclass(frozen=True, slots=True)
class CreateSession:
    session_id: str
    user_id: str | None
    tenant_id: str | None


@dataclass(frozen=True, slots=True)
class StartExecution:
    run_id: str
    session_id: str
    kind: RunKind
    definition: RunDefinition
    input: JsonValue
    root_execution_id: str | None = None
    parent_execution_id: str | None = None


@dataclass(frozen=True, slots=True)
class ClaimExecution:
    run_id: str
    owner: str
    now: datetime
    duration: timedelta


@dataclass(frozen=True, slots=True)
class HeartbeatExecution:
    run_id: str
    owner: str
    fence: int
    now: datetime
    duration: timedelta


@dataclass(frozen=True, slots=True)
class PauseExecution:
    run_id: str
    owner: str
    fence: int
    snapshot: AgentSnapshotData
    pending_approval: RunApproval


@dataclass(frozen=True, slots=True)
class DecideApproval:
    run_id: str
    approval_id: str
    decision: ApprovalDecision
    decided_by: str


@dataclass(frozen=True, slots=True)
class ResumeExecution:
    run_id: str


@dataclass(frozen=True, slots=True)
class RequestCancellation:
    run_id: str
    owner: str
    fence: int
    requested_at: datetime


@dataclass(frozen=True, slots=True)
class CompleteExecution:
    run_id: str
    owner: str
    fence: int
    snapshot: AgentSnapshotData


@dataclass(frozen=True, slots=True)
class FailExecution:
    run_id: str
    owner: str
    fence: int
    snapshot: AgentSnapshotData
    error: RunError | None = None


@dataclass(frozen=True, slots=True)
class AcknowledgeCancellation:
    run_id: str
    owner: str
    fence: int
    snapshot: AgentSnapshotData


@dataclass(frozen=True, slots=True)
class AbortExecution:
    run_id: str
    owner: str
    fence: int
    error: RunError
    trace_end_sequence: int


__all__ = [
    "AbortExecution",
    "AcknowledgeCancellation",
    "ClaimExecution",
    "CompleteExecution",
    "CreateSession",
    "DecideApproval",
    "FailExecution",
    "HeartbeatExecution",
    "PauseExecution",
    "RequestCancellation",
    "ResumeExecution",
    "StartExecution",
]
