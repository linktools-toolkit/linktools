#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Execution DTOs and the product state vocabulary."""

from datetime import datetime
from enum import StrEnum
from typing import Any, AsyncIterator, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..foundation.json import canonical_json_bytes

T = TypeVar("T")


class ExecutionProfile(StrEnum):
    """Supported security and durability profiles."""

    PRODUCTION_SERVICE = "production-service"
    PRODUCTION_SANDBOXED = "production-sandboxed"
    LOCAL_CODING = "local-coding"


class PayloadRef(BaseModel):
    """Reference for input/result data too large for an inline payload."""

    model_config = ConfigDict(frozen=True)

    blob_id: str
    digest: str
    size: int = Field(ge=0)

    def validate_size_digest(self, size: int, digest: str) -> bool:
        """Validate metadata without downloading the referenced bytes."""
        return self.size == size and self.digest == digest


class ExecutionStatus(StrEnum):
    """Product execution projection states."""

    ACCEPTED = "ACCEPTED"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    WAITING_EXTERNAL = "WAITING_EXTERNAL"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


ALLOWED_EXECUTION_TRANSITIONS: "dict[ExecutionStatus, frozenset[ExecutionStatus]]" = {
    ExecutionStatus.ACCEPTED: frozenset({ExecutionStatus.RUNNING, ExecutionStatus.CANCELLED, ExecutionStatus.EXPIRED, ExecutionStatus.FAILED}),
    ExecutionStatus.RUNNING: frozenset({ExecutionStatus.WAITING_APPROVAL, ExecutionStatus.WAITING_EXTERNAL, ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED, ExecutionStatus.EXPIRED}),
    ExecutionStatus.WAITING_APPROVAL: frozenset({ExecutionStatus.RUNNING, ExecutionStatus.CANCELLED, ExecutionStatus.EXPIRED, ExecutionStatus.FAILED}),
    ExecutionStatus.WAITING_EXTERNAL: frozenset({ExecutionStatus.RUNNING, ExecutionStatus.CANCELLED, ExecutionStatus.EXPIRED, ExecutionStatus.FAILED}),
    ExecutionStatus.SUCCEEDED: frozenset(),
    ExecutionStatus.FAILED: frozenset(),
    ExecutionStatus.CANCELLED: frozenset(),
    ExecutionStatus.EXPIRED: frozenset(),
}


def can_transition(current: ExecutionStatus, target: ExecutionStatus) -> bool:
    """Return whether an execution transition is allowed."""
    return target in ALLOWED_EXECUTION_TRANSITIONS[current]


class Execution(BaseModel):
    """Workflow projection value with explicit state transitions."""

    model_config = ConfigDict(frozen=True)

    execution_id: str
    status: ExecutionStatus = ExecutionStatus.ACCEPTED

    def transition_to(self, target: ExecutionStatus) -> "Execution":
        """Return a new projection after an allowed transition."""
        if not can_transition(self.status, target):
            raise ValueError(f"invalid execution transition: {self.status} -> {target}")
        return self.model_copy(update={"status": target})


class ExecutionRequest(BaseModel):
    """Validated request accepted by the Runtime Protocol."""

    model_config = ConfigDict(frozen=True)

    idempotency_key: str = Field(min_length=1, max_length=256)
    agent_id: str = Field(min_length=1)
    agent_revision: "int | None" = Field(default=None, ge=1)
    requested_profile: "ExecutionProfile | None" = None
    input: "Any | PayloadRef"
    conversation_id: "str | None" = None
    deadline: "datetime | None" = None
    metadata: "dict[str, Any]" = Field(default_factory=dict)


class ExecutionHandle(BaseModel):
    """Stable handle returned after accepting an execution."""

    model_config = ConfigDict(frozen=True)

    execution_id: str
    workflow_id: "str | None" = None
    status: ExecutionStatus


class ExecutionView(BaseModel):
    """Tenant-scoped execution projection."""

    model_config = ConfigDict(frozen=True)

    execution_id: str
    tenant_id: str
    agent_id: str
    agent_revision: int
    profile: ExecutionProfile
    conversation_id: "str | None"
    status: ExecutionStatus
    run_id: "str | None" = None
    result_digest: "str | None" = None
    workflow_id: "str | None" = None
    client_request_fingerprint: "str | None" = None
    resolved_execution_fingerprint: "str | None" = None


class ExecutionResult(BaseModel):
    """Validated business result, never a live token delta."""

    model_config = ConfigDict(frozen=True)

    execution_id: str
    contract_id: str
    contract_version: int
    digest: str
    inline_value: Any = None
    payload_ref: "str | None" = None

    @model_validator(mode="after")
    def validate_inline_size(self) -> "ExecutionResult":
        """Keep large results outside durable workflow payloads."""
        if self.inline_value is not None and len(canonical_json_bytes(self.inline_value)) > 64 * 1024:
            if self.payload_ref is None:
                raise ValueError("large results require payload_ref")
        return self

    def verify_digest(self, value: object) -> bool:
        """Verify an inline value against the stored result digest."""
        from ..foundation.digest import sha256_digest

        return sha256_digest(canonical_json_bytes(value)) == self.digest


class ExecutionEvent(BaseModel):
    """Durable semantic event."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    execution_id: str
    sequence: int = Field(ge=1)
    event_type: str
    source_id: "str | None" = None
    source_phase: "str | None" = None
    correlation_id: "str | None" = None
    causation_id: "str | None" = None
    source_attempt: int = Field(default=1, ge=1)
    occurred_at: datetime
    recorded_at: datetime
    payload: Any = Field(default_factory=dict)


class LiveDeltaEnvelope(BaseModel):
    """Best-effort provisional model output."""

    model_config = ConfigDict(frozen=True)

    execution_id: str
    model_request_id: str
    attempt: int = Field(ge=1)
    offset: int = Field(ge=0)
    kind: 'Literal["attempt_started", "delta", "attempt_aborted", "attempt_completed"]'
    content: "str | None" = Field(default=None, max_length=32 * 1024)


class Page(BaseModel, Generic[T]):
    """Opaque-cursor page."""

    model_config = ConfigDict(frozen=True)

    items: "tuple[T, ...]"
    next_cursor: "str | None" = None


class ApprovalDecisionRequest(BaseModel):
    """Decision submitted by an authenticated caller."""

    model_config = ConfigDict(frozen=True)

    approval_id: str
    decision_id: str
    approved: bool
    reason: "str | None" = None


class ApprovalDecisionResult(BaseModel):
    """Explicit delivery status for an approval decision."""

    model_config = ConfigDict(frozen=True)

    decision_id: str
    status: str


class ApprovalView(BaseModel):
    """Public pending-approval view without mutable tool parameters."""

    model_config = ConfigDict(frozen=True)

    approval_id: str
    execution_id: str
    status: str
    tool_name: str
    parameter_digest: str


class CancelExecutionRequest(BaseModel):
    """Cancellation command."""

    model_config = ConfigDict(frozen=True)

    reason: "str | None" = None


class CancelExecutionResult(BaseModel):
    """Cancellation acknowledgement."""

    model_config = ConfigDict(frozen=True)

    execution_id: str
    status: ExecutionStatus


class RetryRequest(BaseModel):
    """Retry policy selected by the caller."""

    model_config = ConfigDict(frozen=True)

    from_checkpoint: bool = False


class ForkRequest(BaseModel):
    """Allowed transcript boundary for a new execution."""

    model_config = ConfigDict(frozen=True)

    through_sequence: "int | None" = Field(default=None, ge=0)


class ArtifactView(BaseModel):
    """Artifact metadata exposed by Runtime."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str
    execution_id: str
    digest: str
    content_type: str
    retention: str


class ArtifactDownload(BaseModel):
    """Opaque, short-lived artifact delivery grant."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str
    uri: str
    expires_at: datetime


class CreateConversationRequest(BaseModel):
    """Conversation creation command."""

    model_config = ConfigDict(frozen=True)

    title: "str | None" = None


class ConversationView(BaseModel):
    """Conversation ownership view."""

    model_config = ConfigDict(frozen=True)

    conversation_id: str
    tenant_id: str
    subject_id: str
    title: "str | None" = None


class RunEvaluationRequest(BaseModel):
    """Evaluation start command."""

    model_config = ConfigDict(frozen=True)

    agent_id: str
    agent_revision: int
    dataset_digest: str


class EvaluationHandle(BaseModel):
    """Stable evaluation handle."""

    model_config = ConfigDict(frozen=True)

    evaluation_id: str


class EvaluationView(BaseModel):
    """Evaluation report view."""

    model_config = ConfigDict(frozen=True)

    evaluation_id: str
    status: str
    dataset_digest: str


class DeleteDataRequest(BaseModel):
    """Auditable deletion command."""

    model_config = ConfigDict(frozen=True)

    execution_id: "str | None" = None
    conversation_id: "str | None" = None


class DeletionHandle(BaseModel):
    """Deletion job handle."""

    model_config = ConfigDict(frozen=True)

    deletion_id: str


class DeletionView(BaseModel):
    """Deletion evidence view."""

    model_config = ConfigDict(frozen=True)

    deletion_id: str
    status: str


ExecutionStreamItem = ExecutionEvent | LiveDeltaEnvelope
ExecutionStream = AsyncIterator[ExecutionStreamItem]
