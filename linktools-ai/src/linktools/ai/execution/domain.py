#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure Run domain values and the single Run state machine."""


from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, Literal, TypeVar

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from ..storage.coordination.lease import Lease
    from ..json import JsonValue

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Page(Generic[T]):
    items: "tuple[T, ...]"
    has_more: bool
    next_cursor: "int | None" = None


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


class MessageCaptureState(StrEnum):
    # The turn's TURN_DELTA is a complete record of what the run produced.
    COMPLETE = "complete"
    # The run was interrupted mid-flight (streaming/tool/cancel-before-commit);
    # the delta holds whatever complete ModelMessages could be salvaged.
    PARTIAL = "partial"
    # No trustworthy delta exists (e.g. engine pre-stage failure, legacy turn).
    UNAVAILABLE = "unavailable"


class RunnableType(StrEnum):
    AGENT = "agent"
    TASK = "task"


class ApprovalDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class RunDefinition:
    runnable_id: str
    runnable_type: RunnableType
    schema: "Literal['agent-spec.v1']"
    spec: "JsonValue"
    spec_hash: str


@dataclass(frozen=True, slots=True)
class RunApproval:
    approval_id: str
    tool_call_id: str
    tool_name: str
    binding_fingerprint: str
    decision: "ApprovalDecision | None" = None
    decided_by: "str | None" = None
    decided_at: "datetime | None" = None


@dataclass(frozen=True, slots=True)
class RunUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # Prompt-cache token counts (Anthropic-style; 0 when the provider has no
    # cache concept). Kept separate from total_tokens so existing
    # total-based accounting is unchanged; callers that want the true
    # consumption add these in: real_total = total + cache_write + cache_read.
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass(frozen=True, slots=True)
class RunError:
    error_type: str
    message: str
    detail: "JsonValue | None" = None


RunErrorInfo = RunError


@dataclass(frozen=True, slots=True)
class RunRecord:
    id: str
    session_id: str
    kind: RunKind
    runnable_id: str
    runnable_type: RunnableType
    input: "JsonValue"
    definition: RunDefinition
    status: RunStatus
    session_turn_sequence: "int | None"
    parent_execution_id: "str | None"
    root_execution_id: str
    approval: "RunApproval | None"
    lease: "Lease"
    cancel_requested_at: "datetime | None"
    snapshot_revision: int
    trace_sequence: int
    event_sequence: int
    tenant_id: "str | None"
    user_id: "str | None"
    error: "RunError | None"
    created_at: "datetime"
    updated_at: "datetime"

    @property
    def pending_approval(self) -> "RunApproval | None":
        if self.approval is not None and self.approval.decision is None:
            return self.approval
        return None


ALLOWED_RUN_TRANSITIONS: "dict[RunStatus, frozenset[RunStatus]]" = {
    RunStatus.PENDING: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED}),
    RunStatus.RUNNING: frozenset({RunStatus.PAUSED, RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLING}),
    RunStatus.PAUSED: frozenset({RunStatus.PENDING, RunStatus.CANCELLED, RunStatus.CANCELLING}),
    RunStatus.CANCELLING: frozenset({RunStatus.CANCELLED, RunStatus.FAILED}),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}


__all__ = [
    "ALLOWED_RUN_TRANSITIONS",
    "ApprovalDecision",
    "MessageCaptureState",
    "Page",
    "RunApproval",
    "RunDefinition",
    "RunError",
    "RunErrorInfo",
    "RunKind",
    "RunRecord",
    "RunStatus",
    "RunUsage",
    "RunnableType",
]
