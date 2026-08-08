#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure values shared by every AI subsystem."""

from dataclasses import dataclass
from enum import StrEnum

from ._paging import Page


class ResourceKind(StrEnum):
    SESSION = "SESSION"
    EXECUTION = "EXECUTION"
    TASK_GRAPH = "TASK_GRAPH"
    EVALUATION = "EVALUATION"
    APPROVAL = "APPROVAL"
    EXTERNAL_CALL = "EXTERNAL_CALL"
    ARTIFACT = "ARTIFACT"
    MEMORY = "MEMORY"
    TOOL_OPERATION = "TOOL_OPERATION"
    DOWNLOAD_GRANT = "DOWNLOAD_GRANT"


class ExecutionEventType(StrEnum):
    EXECUTION_CREATED = "EXECUTION_CREATED"
    EXECUTION_STARTED = "EXECUTION_STARTED"
    EXECUTION_START_UNKNOWN = "EXECUTION_START_UNKNOWN"
    APPROVAL_REQUESTED = "APPROVAL_REQUESTED"
    APPROVAL_DECIDED = "APPROVAL_DECIDED"
    EXTERNAL_REQUESTED = "EXTERNAL_REQUESTED"
    EXTERNAL_SUPPLIED = "EXTERNAL_SUPPLIED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    EXECUTION_SUCCEEDED = "EXECUTION_SUCCEEDED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    EXECUTION_CANCELLED = "EXECUTION_CANCELLED"


class ExecutionLineageKind(StrEnum):
    RUN = "RUN"
    SESSION_RESUME = "SESSION_RESUME"
    RETRY = "RETRY"
    FORK = "FORK"
    SUBAGENT = "SUBAGENT"


class ExecutionStatus(StrEnum):
    PENDING_START = "PENDING_START"
    STARTED = "STARTED"
    START_UNKNOWN = "START_UNKNOWN"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    WAITING_EXTERNAL = "WAITING_EXTERNAL"
    CANCELLING = "CANCELLING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class SessionStatus(StrEnum):
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    CLEANUP_REQUIRED = "CLEANUP_REQUIRED"


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    BLOCKED = "BLOCKED"


class EvaluationStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ApprovalStatus(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class ApprovalDecision(StrEnum):
    APPROVE = "APPROVE"
    DENY = "DENY"


class IdempotencyStatus(StrEnum):
    RESERVED = "RESERVED"
    STARTED = "STARTED"
    START_UNKNOWN = "START_UNKNOWN"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ToolOperationStatus(StrEnum):
    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EFFECT_UNKNOWN = "EFFECT_UNKNOWN"


class ExternalCallStatus(StrEnum):
    PENDING = "PENDING"
    SUPPLIED = "SUPPLIED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class BlobStatus(StrEnum):
    UPLOADING = "UPLOADING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class OperationKind(StrEnum):
    EXECUTION_START = "EXECUTION_START"
    MODEL = "MODEL"
    TOOL = "TOOL"
    APPROVAL = "APPROVAL"
    EXTERNAL = "EXTERNAL"
    BUDGET = "BUDGET"
    RESULT = "RESULT"
    EVENT = "EVENT"
    EXECUTION_CANCEL = "EXECUTION_CANCEL"
    TASK_CANCEL = "TASK_CANCEL"
    SESSION_CREATE = "SESSION_CREATE"
    SESSION_FORK = "SESSION_FORK"
    SESSION_UPDATE = "SESSION_UPDATE"
    SESSION_CLOSE = "SESSION_CLOSE"
    TASK_NODE = "TASK_NODE"
    DOWNLOAD_GRANT = "DOWNLOAD_GRANT"


class OperationStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EFFECT_UNKNOWN = "EFFECT_UNKNOWN"
    COMPACTED = "COMPACTED"


class StopReason(StrEnum):
    END_TURN = "END_TURN"
    REFUSAL = "REFUSAL"
    TURN_LIMIT = "TURN_LIMIT"
    OUTPUT_VALIDATION_FAILED = "OUTPUT_VALIDATION_FAILED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class Principal:
    principal_id: str
    tenant_id: str
    kind: str = "user"

    def __post_init__(self) -> None:
        if not self.principal_id.strip() or not self.tenant_id.strip() or not self.kind.strip():
            raise ValueError("principal is incomplete")


class PrincipalKind(StrEnum):
    USER = "user"
    SERVICE = "service"
    LOCAL_TRUSTED = "LOCAL_TRUSTED"


__all__ = [
    "ApprovalDecision", "ApprovalStatus", "BlobStatus", "EvaluationStatus",
    "ExecutionEventType", "ExecutionLineageKind", "ExecutionStatus", "ExternalCallStatus",
    "IdempotencyStatus", "OperationKind", "OperationStatus", "Page", "Principal",
    "PrincipalKind", "ResourceKind", "SessionStatus", "StopReason", "TaskStatus",
    "ToolOperationStatus",
]
