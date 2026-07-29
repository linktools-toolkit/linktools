"""Pure Run domain values and the single Run state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Mapping

from ..storage.coordination.lease import Lease
from ..storage.json import JsonValue


class RunKind(StrEnum):
    USER_TURN = "user_turn"
    SUBAGENT = "subagent"
    BACKGROUND = "background"
    TASK = "task"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunnableType(StrEnum):
    AGENT = "agent"
    TASK = "task"


@dataclass(frozen=True, slots=True)
class RunDefinition:
    runnable_id: str
    runnable_type: RunnableType
    spec_schema: str
    spec: JsonValue
    spec_hash: str


@dataclass(frozen=True, slots=True)
class RunApproval:
    approval_id: str
    tool_call_id: str
    tool_name: str
    arguments: JsonValue
    decision: str | None = None
    decided_by: str | None = None


@dataclass(frozen=True, slots=True)
class RunUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class RunError:
    error_type: str
    message: str
    detail: JsonValue | None = None


RunErrorInfo = RunError


@dataclass(frozen=True, slots=True)
class RunRecord:
    id: str
    session_id: str
    kind: RunKind
    runnable_id: str
    runnable_type: RunnableType
    definition: RunDefinition
    status: RunStatus
    session_turn_sequence: int | None
    parent_run_id: str | None
    root_run_id: str
    pending_approval: RunApproval | None
    lease: Lease
    cancel_requested_at: datetime | None
    snapshot_revision: int
    trace_sequence: int
    event_sequence: int
    tenant_id: str | None
    user_id: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class RunContextValue:
    run_id: str
    root_run_id: str
    parent_run_id: str | None
    session_id: str
    runnable_id: str
    runnable_type: RunnableType
    user_id: str | None
    tenant_id: str | None
    workspace: object | None


# These result values remain model/agent boundary values; they are not stored
# Run records and do not introduce a second lifecycle model.
@dataclass(frozen=True, slots=True)
class RunInput:
    prompt: str
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunResult:
    output: object
    token_usage: Mapping[str, JsonValue] = field(default_factory=dict)
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)


ALLOWED_RUN_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PENDING: frozenset({RunStatus.RUNNING}),
    RunStatus.RUNNING: frozenset({RunStatus.PAUSED, RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLING}),
    RunStatus.PAUSED: frozenset({RunStatus.PENDING, RunStatus.CANCELLING}),
    RunStatus.CANCELLING: frozenset({RunStatus.CANCELLED, RunStatus.FAILED}),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class RunCheckpoint:
    id: str
    run_id: str
    sequence: int
    format: str
    schema_version: int
    payload: bytes
    created_at: datetime
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NewRunCheckpoint:
    run_id: str
    format: str
    schema_version: int
    payload: bytes
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)


__all__ = [
    "ALLOWED_RUN_TRANSITIONS",
    "NewRunCheckpoint",
    "RunApproval",
    "RunCheckpoint",
    "RunDefinition",
    "RunError",
    "RunErrorInfo",
    "RunInput",
    "RunKind",
    "RunRecord",
    "RunResult",
    "RunStatus",
    "RunUsage",
    "RunnableType",
]
