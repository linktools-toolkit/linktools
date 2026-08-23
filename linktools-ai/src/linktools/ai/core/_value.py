"""Pure values shared by every AI subsystem."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from ..errors import AIError, ErrorCode
from ._paging import Page
from ._validation import (
    validate_principal_id,
    validate_principal_kind,
    validate_tenant_id,
)


def canonical_string_tuple(value: Sequence[str], *, field: str) -> "tuple[str, ...]":
    """Validate, deduplicate, and sort one string selector sequence."""
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, f"{field} must be an array of strings")
    selectors: list[str] = []
    for selector in value:
        if not isinstance(selector, str) or not selector or selector != selector.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, f"{field} contains an invalid selector")
        selectors.append(selector)
    normalized = tuple(sorted(set(selectors)))
    if "*" in normalized and normalized != ("*",):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, f"{field} cannot mix '*' with other selectors")
    return normalized


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
    ASSISTANT_PART_COMPLETED = "ASSISTANT_PART_COMPLETED"
    TOOL_CALL_STARTED = "TOOL_CALL_STARTED"
    TOOL_CALL_FINISHED = "TOOL_CALL_FINISHED"


class ExecutionDeltaType(StrEnum):
    """Process-local presentation updates that never enter durable state."""

    ASSISTANT_TEXT_DELTA = "ASSISTANT_TEXT_DELTA"
    ASSISTANT_THINKING_DELTA = "ASSISTANT_THINKING_DELTA"


class ExecutionLineageKind(StrEnum):
    RUN = "RUN"
    SESSION_RESUME = "SESSION_RESUME"
    RETRY = "RETRY"
    FORK = "FORK"
    SUBAGENT = "SUBAGENT"


class ExecutionStatus(StrEnum):
    PENDING_START = "PENDING_START"
    STARTED = "STARTED"
    FINALIZING = "FINALIZING"
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
    MEMORY_WRITE = "MEMORY_WRITE"
    MEMORY_DELETE = "MEMORY_DELETE"
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


class PrincipalKind(StrEnum):
    USER = "user"
    SERVICE = "service"
    LOCAL_TRUSTED = "local_trusted"


@dataclass(frozen=True, slots=True)
class Principal:
    principal_id: str
    tenant_id: str
    kind: str = PrincipalKind.USER.value

    def __post_init__(self) -> None:
        try:
            validate_principal_id(self.principal_id)
            validate_tenant_id(self.tenant_id)
            validate_principal_kind(self.kind)
        except AIError as error:
            raise ValueError("principal identity is invalid") from error


__all__ = [
    "ApprovalDecision", "ApprovalStatus", "EvaluationStatus",
    "ExecutionDeltaType", "ExecutionEventType", "ExecutionLineageKind",
    "ExecutionStatus", "ExternalCallStatus",
    "IdempotencyStatus", "OperationKind", "OperationStatus", "Page", "Principal",
    "PrincipalKind", "ResourceKind", "SessionStatus", "StopReason", "TaskStatus",
    "ToolOperationStatus",
]
