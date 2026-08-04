#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Validated commands accepted by the execution lifecycle port."""


from dataclasses import dataclass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime, timedelta
    from ..json import JsonValue
    from .domain import ApprovalDecision, RunApproval, RunDefinition, RunError, RunKind
    from .snapshots import AgentSnapshotData


@dataclass(frozen=True, slots=True)
class ParentLeaseGuard:
    run_id: str
    owner: str
    fence: int


@dataclass(frozen=True, slots=True)
class CreateSession:
    session_id: str
    user_id: "str | None"
    tenant_id: "str | None"


@dataclass(frozen=True, slots=True)
class StartExecution:
    run_id: str
    session_id: str
    kind: "RunKind"
    definition: "RunDefinition"
    input: "JsonValue"
    root_execution_id: "str | None" = None
    parent_execution_id: "str | None" = None
    parent_guard: "ParentLeaseGuard | None" = None


@dataclass(frozen=True, slots=True)
class ClaimExecution:
    run_id: str
    owner: str
    now: "datetime"
    duration: "timedelta"


@dataclass(frozen=True, slots=True)
class HeartbeatExecution:
    run_id: str
    owner: str
    fence: int
    now: "datetime"
    duration: "timedelta"


@dataclass(frozen=True, slots=True)
class PauseExecution:
    run_id: str
    owner: str
    fence: int
    snapshot: "AgentSnapshotData"
    pending_approval: "RunApproval"


@dataclass(frozen=True, slots=True)
class DecideApproval:
    run_id: str
    approval_id: str
    decision: "ApprovalDecision"
    decided_by: str


@dataclass(frozen=True, slots=True)
class ResumeExecution:
    run_id: str


@dataclass(frozen=True, slots=True)
class RequestCancellation:
    run_id: str
    owner: str
    fence: int
    requested_at: "datetime"


@dataclass(frozen=True, slots=True)
class CompleteExecution:
    run_id: str
    owner: str
    fence: int
    snapshot: "AgentSnapshotData"


@dataclass(frozen=True, slots=True)
class FailExecution:
    run_id: str
    owner: str
    fence: int
    snapshot: "AgentSnapshotData"
    error: "RunError | None" = None


@dataclass(frozen=True, slots=True)
class AcknowledgeCancellation:
    run_id: str
    owner: str
    fence: int
    snapshot: "AgentSnapshotData"


@dataclass(frozen=True, slots=True)
class AbortExecution:
    run_id: str
    owner: str
    fence: int
    snapshot: "AgentSnapshotData"
    error: "RunError"

    def __post_init__(self) -> None:
        from .domain import RunError
        from .snapshots import AgentSnapshotData

        if not isinstance(self.snapshot, AgentSnapshotData):
            raise TypeError("AbortExecution.snapshot must be AgentSnapshotData")
        if not isinstance(self.error, RunError):
            raise TypeError("AbortExecution.error must be RunError")

    @property
    def trace_end_sequence(self) -> int:
        return self.snapshot.trace_end_sequence


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
    "ParentLeaseGuard",
    "RequestCancellation",
    "ResumeExecution",
    "StartExecution",
]
