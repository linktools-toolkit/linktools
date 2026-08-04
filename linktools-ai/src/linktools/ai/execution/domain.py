#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure Run domain values and the single Run state machine."""


from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import Generic, Literal, TypeVar

from ..errors import RunDefinitionIntegrityError
from ..json import canonical_json_bytes

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


def compute_run_definition_hash(*, schema: str, spec: "JsonValue") -> str:
    payload = {"schema": schema, "spec": spec}
    return sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class RunDefinition:
    runnable_id: str
    runnable_type: RunnableType
    schema: "Literal['agent-spec.v1', 'swarm-spec.v1', 'swarm-task-graph.v1']"
    spec: "JsonValue"
    spec_hash: str

    def __post_init__(self) -> None:
        expected = compute_run_definition_hash(schema=self.schema, spec=self.spec)
        if self.spec_hash != expected:
            raise RunDefinitionIntegrityError("definition hash mismatch")


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
    total_cost: "Decimal | None" = None

    def __post_init__(self) -> None:
        for name, value in (
            ("cache_write_tokens", self.cache_write_tokens),
            ("cache_read_tokens", self.cache_read_tokens),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        if self.total_cost is not None and not isinstance(self.total_cost, Decimal):
            object.__setattr__(self, "total_cost", Decimal(str(self.total_cost)))
        if (
            self.total_cost is not None
            and (not self.total_cost.is_finite() or self.total_cost < 0)
        ):
            raise ValueError("total_cost must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class RunError:
    error_type: str
    message: str
    detail: "JsonValue | None" = None


def sanitize_run_error(exc: BaseException) -> RunError:
    error_type = type(exc).__name__
    kind = getattr(exc, "kind", None)
    message = str(kind) if isinstance(kind, str) else "execution failed"
    return RunError(error_type=error_type, message=message)


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
    RunStatus.PENDING: frozenset({RunStatus.RUNNING, RunStatus.FAILED, RunStatus.CANCELLED}),
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
    "compute_run_definition_hash",
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
    "sanitize_run_error",
]
