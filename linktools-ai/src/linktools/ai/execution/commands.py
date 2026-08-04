#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Validated commands accepted by the execution lifecycle port."""


from dataclasses import dataclass
from hashlib import sha256

from ..json import canonical_json_bytes
from .domain import compute_run_definition_hash
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime, timedelta
    from ..json import JsonValue
    from .domain import (
        ApprovalDecision,
        RunApproval,
        RunDefinition,
        RunError,
        RunKind,
        RunRecord,
        RunUsage,
        RunnableType,
    )
from .snapshots import AgentSnapshotData


@dataclass(frozen=True, slots=True)
class ParentLeaseGuard:
    run_id: str
    owner: str
    fence: int


@dataclass(frozen=True, slots=True)
class StartExecutionIdentity:
    run_id: str
    session_id: str
    kind: "RunKind"
    runnable_id: str
    runnable_type: "RunnableType"
    definition_hash: str
    input_hash: str
    parent_execution_id: "str | None"
    root_execution_id: str
    tenant_id: "str | None"
    user_id: "str | None"


@dataclass(frozen=True, slots=True)
class StartRunResult:
    record: "RunRecord"
    created: bool


def start_execution_identity(
    command: "StartExecution", *, tenant_id: "str | None", user_id: "str | None"
) -> StartExecutionIdentity:
    return StartExecutionIdentity(
        run_id=command.run_id,
        session_id=command.session_id,
        kind=command.kind,
        runnable_id=command.definition.runnable_id,
        runnable_type=command.definition.runnable_type,
        definition_hash=compute_run_definition_hash(
            schema=command.definition.schema, spec=command.definition.spec
        ),
        input_hash=sha256(canonical_json_bytes(command.input)).hexdigest(),
        parent_execution_id=command.parent_execution_id,
        root_execution_id=command.root_execution_id or command.run_id,
        tenant_id=tenant_id,
        user_id=user_id,
    )


def run_record_identity(record: "RunRecord") -> StartExecutionIdentity:
    return StartExecutionIdentity(
        run_id=record.id,
        session_id=record.session_id,
        kind=record.kind,
        runnable_id=record.runnable_id,
        runnable_type=record.runnable_type,
        definition_hash=compute_run_definition_hash(
            schema=record.definition.schema, spec=record.definition.spec
        ),
        input_hash=sha256(canonical_json_bytes(record.input)).hexdigest(),
        parent_execution_id=record.parent_execution_id,
        root_execution_id=record.root_execution_id,
        tenant_id=record.tenant_id,
        user_id=record.user_id,
    )


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
    parent_guard: "ParentLeaseGuard | None" = None


@dataclass(frozen=True, slots=True)
class StartClaimedChildExecution:
    start: "StartExecution"
    child_owner: str
    now: "datetime"
    lease_duration: "timedelta"


@dataclass(frozen=True, slots=True)
class StartClaimedChildResult:
    record: "RunRecord"
    created: bool
    terminal: bool


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
    expected_snapshot_revision: int


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
    expected_snapshot_revision: int


@dataclass(frozen=True, slots=True)
class FailExecution:
    run_id: str
    owner: str
    fence: int
    snapshot: "AgentSnapshotData"
    error: "RunError | None"
    expected_snapshot_revision: int


@dataclass(frozen=True, slots=True)
class AcknowledgeCancellation:
    run_id: str
    owner: str
    fence: int
    snapshot: "AgentSnapshotData"
    expected_snapshot_revision: int


@dataclass(frozen=True, slots=True)
class AbortExecution:
    run_id: str
    owner: str
    fence: int
    snapshot: "AgentSnapshotData"
    error: "RunError"
    expected_snapshot_revision: int

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


@dataclass(frozen=True, slots=True)
class CheckpointExecutionUsage:
    run_id: str
    owner: str
    fence: int
    expected_snapshot_revision: int
    usage: "RunUsage"
    trace_end_sequence: int


__all__ = [
    "AbortExecution",
    "AcknowledgeCancellation",
    "ClaimExecution",
    "CheckpointExecutionUsage",
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
    "StartClaimedChildExecution",
    "StartClaimedChildResult",
    "StartExecutionIdentity",
    "StartRunResult",
    "run_record_identity",
    "start_execution_identity",
]
