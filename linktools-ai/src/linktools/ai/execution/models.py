"""Domain values for execution history and semantic run traces."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Mapping

JsonValue = Any


class RunKind(str, Enum):
    USER_TURN = "user_turn"
    SUBAGENT = "subagent"
    BACKGROUND = "background"
    TASK = "task"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class RunDefinitionSnapshot:
    agent_id: str
    model: str | None = None
    settings: Mapping[str, JsonValue] = field(default_factory=dict)


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
class RunErrorInfo:
    error_type: str
    message: str
    detail: JsonValue | None = None


@dataclass(frozen=True, slots=True)
class AgentOutcome:
    status: RunStatus
    snapshot: "RunSnapshot"
    error: RunErrorInfo | None = None
    pause: Any | None = None


@dataclass(frozen=True, slots=True)
class SessionRecord:
    id: str
    user_id: str | None
    tenant_id: str | None
    next_turn_sequence: int
    latest_completed_run_id: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class SessionTurn:
    session_id: str
    sequence: int
    run_id: str
    user_prompt: JsonValue | str
    assistant_summary: JsonValue | str | None
    status: RunStatus
    created_at: datetime
    completed_at: datetime | None


@dataclass(frozen=True, slots=True)
class RunRecord:
    id: str
    session_id: str
    kind: RunKind
    session_turn_sequence: int | None
    parent_run_id: str | None
    root_run_id: str
    status: RunStatus
    definition: RunDefinitionSnapshot
    pending_approval: RunApproval | None
    execution_owner: str | None
    execution_fence: int
    lease_expires_at: datetime | None
    cancel_requested_at: datetime | None
    snapshot_revision: int
    trace_sequence: int
    created_at: datetime
    updated_at: datetime
    tenant_id: str | None = None
    user_id: str | None = None


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    schema: Literal["run-snapshot.v1"]
    run_id: str
    revision: int
    resume_messages: tuple[JsonValue, ...]
    final_output: JsonValue | str | None
    status: RunStatus
    usage: RunUsage
    trace_end_sequence: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class NewRunTraceStep:
    kind: Literal["model_interaction", "tool_result"]
    payload: JsonValue
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RunTraceStep:
    run_id: str
    sequence: int
    kind: Literal["model_interaction", "tool_result"]
    payload: JsonValue
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RunEvent:
    run_id: str
    sequence: int
    type: str
    payload: JsonValue
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RunEvaluation:
    run_id: str
    evaluator: str
    score: float | None
    result: JsonValue
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Page:
    items: tuple[Any, ...]
    has_more: bool
    next_cursor: int | None = None
