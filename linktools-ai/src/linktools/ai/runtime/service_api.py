#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime service protocols and transport-neutral request values."""

from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, cast

from ..agent import AgentBindingContract
from ..core import (
    ApprovalDecision,
    ApprovalStatus,
    CorrelationData,
    ExecutionLineageKind,
    ExecutionMode,
    ExecutionStatus,
    JsonValue,
    Page,
    Principal,
    SessionStatus,
    ThinkingValue,
    UsageMetrics,
    normalize_correlation,
    normalize_execution_mode,
    normalize_thinking,
    validate_idempotency_key,
    validate_memory_scope,
    validate_page_limit,
    validate_resource_id,
)
from ..errors import AIError, ErrorCode, ErrorDiagnostics
from ..task import TaskBindingContract, TaskEffectResolution, TaskEvent
from ._execution_context import ExecutionInputContext
from ._input_contract import (
    UserPromptInput,
    normalize_input_files,
    validate_user_input,
)
from .recovery import (
    ExecutionRecoveryEffect,
    ResolveToolEffectRequest,
    ToolEffectResolutionResult,
)


class _ExecutionStreamFailure(Exception):
    """Marks an optional broker or validated presentation-only failure.

    Authoritative reads, durable decoding, identity/cursor validation and
    cancellation retain their original exceptions. Broker boundaries must
    preserve AIError rather than classifying it as an optional failure.
    """

    def __init__(self, cause: Exception) -> None:
        super().__init__(type(cause).__name__)
        self.cause = cause


def _request_correlation(value: Mapping[str, object] | None) -> CorrelationData:
    try:
        return normalize_correlation(value)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    user_prompt: "UserPromptInput"
    principal: Principal
    idempotency_key: str
    memory_scope: "str | None"
    mode: ExecutionMode
    planning: bool
    thinking: ThinkingValue
    correlation: CorrelationData = field(default_factory=dict)
    files: tuple[str, ...] = ()
    input_context: ExecutionInputContext | None = None

    def __post_init__(self) -> None:
        if self.input_context is not None and not isinstance(self.input_context, ExecutionInputContext):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self.input_context is not None and self.input_context.unavailable_reason is not None:
            raise AIError(ErrorCode.INPUT_CONTEXT_UNAVAILABLE)
        if self.input_context is not None and self.memory_scope is not None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "imported_context_owns_memory_scope"})
        object.__setattr__(self, "user_prompt", validate_user_input(self.user_prompt))
        files = normalize_input_files(self.files)
        validate_idempotency_key(self.idempotency_key)
        if self.memory_scope is not None:
            validate_memory_scope(self.memory_scope)
        mode = normalize_execution_mode(self.mode)
        thinking = normalize_thinking(self.thinking)
        if not isinstance(self.planning, bool) or mode == "plan" and not self.planning:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "thinking", thinking)
        object.__setattr__(self, "correlation", _request_correlation(self.correlation))
        object.__setattr__(self, "files", files)


@dataclass(frozen=True, slots=True)
class RetryExecutionRequest:
    user_prompt: "UserPromptInput"
    principal: Principal
    idempotency_key: str
    correlation: CorrelationData = field(default_factory=dict)
    files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_prompt", validate_user_input(self.user_prompt))
        files = normalize_input_files(self.files)
        validate_idempotency_key(self.idempotency_key)
        object.__setattr__(self, "correlation", _request_correlation(self.correlation))
        object.__setattr__(self, "files", files)


@dataclass(frozen=True, slots=True)
class ForkExecutionRequest:
    user_prompt: "UserPromptInput"
    principal: Principal
    idempotency_key: str
    correlation: CorrelationData = field(default_factory=dict)
    files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_prompt", validate_user_input(self.user_prompt))
        files = normalize_input_files(self.files)
        validate_idempotency_key(self.idempotency_key)
        object.__setattr__(self, "correlation", _request_correlation(self.correlation))
        object.__setattr__(self, "files", files)


@dataclass(frozen=True, slots=True)
class CancelExecutionRequest:
    principal: Principal
    idempotency_key: str
    force: bool = False

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class CancelExecutionResult:
    execution_id: str
    cancelled: bool


@dataclass(frozen=True, slots=True)
class ExecutionHandle:
    execution_id: str


class _ExecutionViewSource(Protocol):
    execution_id: str
    agent_id: str | None
    status: ExecutionStatus
    lineage_kind: ExecutionLineageKind
    parent_execution_id: str | None
    root_execution_id: str
    parent_invocation_id: str | None
    session_id: str | None
    binding_kind: str
    task_id: str | None
    task_attempt: int
    task_deadline_at: datetime | None
    task_next_attempt_at: datetime | None
    event_seq: int


@dataclass(frozen=True, slots=True)
class ExecutionView:
    execution_id: str
    agent_id: str | None
    status: ExecutionStatus
    lineage_kind: ExecutionLineageKind
    parent_execution_id: str | None
    root_execution_id: str
    parent_invocation_id: str | None
    session_id: str | None = None
    binding_kind: str = "agent"
    task_id: str | None = None
    task_attempt: int = 0
    task_deadline_at: datetime | None = None
    task_next_attempt_at: datetime | None = None
    event_seq: int = 0


def project_execution_view(source: object) -> ExecutionView:
    """Project an internal execution source into the stable public view."""
    value = cast(_ExecutionViewSource, source)
    return ExecutionView(
        execution_id=value.execution_id,
        agent_id=value.agent_id,
        status=value.status,
        lineage_kind=value.lineage_kind,
        parent_execution_id=value.parent_execution_id,
        root_execution_id=value.root_execution_id,
        parent_invocation_id=value.parent_invocation_id,
        session_id=value.session_id,
        binding_kind=value.binding_kind,
        task_id=value.task_id,
        task_attempt=value.task_attempt,
        task_deadline_at=value.task_deadline_at,
        task_next_attempt_at=value.task_next_attempt_at,
        event_seq=value.event_seq,
    )


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    execution_id: str
    status: ExecutionStatus
    output: JsonValue | None
    usage: UsageMetrics
    error_code: "str | None" = None
    safe_error_details: "Mapping[str, JsonValue]" = field(default_factory=dict)
    error_diagnostics: "ErrorDiagnostics | None" = None

    def __post_init__(self) -> None:
        details = dict(self.safe_error_details)
        object.__setattr__(self, "safe_error_details", details)
        if self.error_diagnostics is not None and not isinstance(
            self.error_diagnostics, ErrorDiagnostics
        ):
            raise ValueError("execution result error diagnostics are invalid")
        if self.status is ExecutionStatus.SUCCEEDED:
            if (
                self.error_code is not None
                or details
                or self.error_diagnostics is not None
            ):
                raise ValueError("successful execution result cannot carry an error")
            return
        if self.status is ExecutionStatus.CANCELLED:
            if self.error_code != ErrorCode.EXECUTION_CANCELLED.value:
                raise ValueError(
                    "cancelled execution result requires EXECUTION_CANCELLED"
                )
            if self.output is not None or self.error_diagnostics is not None:
                raise ValueError(
                    "cancelled execution result cannot carry output or diagnostics"
                )
            return
        if self.status is ExecutionStatus.FAILED:
            if self.error_code is None:
                raise ValueError("failed execution result requires an error code")
            if self.error_code == ErrorCode.EXECUTION_CANCELLED.value:
                raise ValueError(
                    "failed execution result cannot carry EXECUTION_CANCELLED"
                )
            if self.output is not None:
                raise ValueError("failed execution result cannot carry output")
            return
        raise ValueError("execution result requires a terminal status")


@dataclass(frozen=True, slots=True)
class ExecutionTraceItem:
    execution_id: str
    step_event_seq: int
    payload: JsonValue

    def __post_init__(self) -> None:
        if self.step_event_seq < 0:
            raise ValueError("execution trace step_event_seq must be non-negative")


@dataclass(frozen=True, slots=True)
class TranscriptItem:
    execution_id: str
    message_seq: int
    text: "str | None"
    content_included: bool = True

    def __post_init__(self) -> None:
        if self.message_seq < 0:
            raise ValueError("transcript message_seq must be non-negative")
        if not isinstance(self.content_included, bool):
            raise TypeError("transcript content flag must be bool")
        if self.content_included:
            if not isinstance(self.text, str):
                raise ValueError("included transcript content must be text")
        elif self.text is not None:
            raise ValueError("omitted transcript content must be None")


@dataclass(frozen=True, slots=True)
class ExecutionHistoryItem:
    """One raw transcript part; request/step identify its originating model request."""

    execution_id: str
    message_seq: int
    item_kind: str
    content: JsonValue
    tool_name: "str | None" = None
    tool_call_id: "str | None" = None
    content_included: bool = True
    agent_run_seq: "int | None" = None
    model_request_seq: "int | None" = None
    tool_operation_id: "str | None" = None
    started_at: "datetime | None" = None
    finished_at: "datetime | None" = None
    duration_ns: "int | None" = None
    status: "str | None" = None

    part_index: int | None = None
    step_index: int | None = None

    def __post_init__(self) -> None:
        if self.message_seq < 0 or not isinstance(self.item_kind, str) or not self.item_kind:
            raise ValueError("execution history item is invalid")
        if not isinstance(self.content_included, bool):
            raise TypeError("history content flag must be bool")
        if not self.content_included and self.content is not None:
            raise ValueError("omitted history content must be None")
        if self.agent_run_seq is not None and self.agent_run_seq < 1:
            raise ValueError("agent_run_seq is invalid")
        if self.model_request_seq is not None and self.model_request_seq < 1:
            raise ValueError("model_request_seq is invalid")
        if self.part_index is not None and (
            isinstance(self.part_index, bool)
            or not isinstance(self.part_index, int)
            or self.part_index < 0
        ):
            raise ValueError("history part index is invalid")
        if self.step_index is not None and (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or self.step_index < 0
        ):
            raise ValueError("history step index is invalid")
        if self.duration_ns is not None and self.duration_ns < 0:
            raise ValueError("history duration is invalid")


@dataclass(frozen=True, slots=True)
class ModelInteractionItem:
    execution_id: str
    agent_run_seq: int
    depth: int
    model_request_seq: int
    purpose: str
    step_index: int
    output_retry_index: int | None
    model: Mapping[str, JsonValue]
    request: Mapping[str, JsonValue]
    response: JsonValue | None
    status: str
    error_code: str | None
    duration_ns: "int | None"
    usage: UsageMetrics | None
    content_included: bool = True
    started_at: "datetime | None" = None
    finished_at: "datetime | None" = None

    def __post_init__(self) -> None:
        if (
            not self.execution_id
            or self.agent_run_seq < 1
            or self.depth < 0
            or self.model_request_seq < 1
            or self.step_index < 0
            or self.status not in {"RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"}
            or self.duration_ns is not None and self.duration_ns < 0
        ):
            raise ValueError("model interaction item is invalid")
        if not isinstance(self.content_included, bool):
            raise TypeError("model interaction content flag must be bool")
        if not self.content_included and (self.request or self.response is not None):
            raise ValueError("omitted model interaction content must be empty")
        if self.status == "RUNNING" and (
            self.started_at is None
            or self.response is not None
            or self.finished_at is not None
            or self.duration_ns is not None
            or self.usage is not None
            or self.error_code is not None
        ):
            raise ValueError("running model interaction has terminal data")
        object.__setattr__(self, "model", deepcopy(dict(self.model)))
        object.__setattr__(self, "request", deepcopy(dict(self.request)))
        object.__setattr__(self, "response", deepcopy(self.response))


@dataclass(frozen=True, slots=True)
class AttachmentFact:
    execution_id: str
    attachment_id: str
    fact: str
    source: str
    media_type: str | None
    size: int | None
    digest: str | None
    position: int
    processing_status: str = "unknown"
    agent_run_seq: int | None = None
    model_request_seq: int | None = None
    step_index: int | None = None
    call_id: str | None = None
    input_identifier: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.execution_id, str)
            or not self.execution_id
            or not _is_digest(self.attachment_id)
            or self.fact not in {"accepted", "included_in_request"}
            or not isinstance(self.source, str)
            or not self.source
            or self.media_type is not None
            and (not isinstance(self.media_type, str) or not self.media_type)
            or self.size is not None
            and (
                isinstance(self.size, bool)
                or not isinstance(self.size, int)
                or self.size < 0
            )
            or self.digest is not None
            and not _is_digest(self.digest)
            or isinstance(self.position, bool)
            or not isinstance(self.position, int)
            or self.position < 0
            or self.processing_status != "unknown"
        ):
            raise ValueError("attachment fact is invalid")
        for value in (self.agent_run_seq, self.model_request_seq):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError("attachment request association is invalid")
        if self.step_index is not None and (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or self.step_index < 0
        ):
            raise ValueError("attachment step association is invalid")
        if self.call_id is not None and (
            not isinstance(self.call_id, str) or not self.call_id
        ):
            raise ValueError("attachment call association is invalid")
        if self.input_identifier is not None and not isinstance(
            self.input_identifier,
            str,
        ):
            raise ValueError("attachment input identifier is invalid")


@dataclass(frozen=True, slots=True)
class UsageReadCutoff:
    execution_id: str
    agent_run_seq: int
    model_request_seq: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.execution_id, str)
            or not self.execution_id
            or isinstance(self.agent_run_seq, bool)
            or not isinstance(self.agent_run_seq, int)
            or self.agent_run_seq < 1
            or isinstance(self.model_request_seq, bool)
            or not isinstance(self.model_request_seq, int)
            or self.model_request_seq < 0
        ):
            raise ValueError("usage read cutoff is invalid")


@dataclass(frozen=True, slots=True)
class UsageSummary:
    logical_requests: int = 0
    succeeded_requests: int = 0
    failed_requests: int = 0
    cancelled_requests: int = 0
    output_correction_retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model_duration_ns: int = 0
    unknown_usage_requests: int = 0
    transport_retries: "int | None" = None
    unrecorded_executions: int = 0
    cutoffs: "tuple[UsageReadCutoff, ...]" = ()
    unknown_duration_requests: int = 0

    def __post_init__(self) -> None:
        counts = (
            self.logical_requests,
            self.succeeded_requests,
            self.failed_requests,
            self.cancelled_requests,
            self.output_correction_retries,
            self.input_tokens,
            self.output_tokens,
            self.cache_read_tokens,
            self.cache_write_tokens,
            self.model_duration_ns,
            self.unknown_usage_requests,
            self.unknown_duration_requests,
            self.unrecorded_executions,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in counts
        ):
            raise ValueError("usage summary counts must be non-negative integers")
        if (
            self.succeeded_requests
            + self.failed_requests
            + self.cancelled_requests
            != self.logical_requests
            or self.output_correction_retries > self.logical_requests
            or self.unknown_usage_requests > self.logical_requests
            or self.unknown_duration_requests > self.logical_requests
        ):
            raise ValueError("usage summary request counts are inconsistent")
        if self.transport_retries is not None and (
            isinstance(self.transport_retries, bool)
            or not isinstance(self.transport_retries, int)
            or self.transport_retries < 0
        ):
            raise ValueError("usage transport retries must be non-negative")
        cutoffs = tuple(sorted(
            self.cutoffs,
            key=lambda value: (
                value.execution_id,
                value.agent_run_seq,
            ),
        ))
        if (
            any(not isinstance(value, UsageReadCutoff) for value in cutoffs)
            or len({
                (value.execution_id, value.agent_run_seq)
                for value in cutoffs
            }) != len(cutoffs)
        ):
            raise ValueError("usage read cutoffs are invalid")
        object.__setattr__(self, "cutoffs", cutoffs)


@dataclass(frozen=True, slots=True)
class SessionHistoryItem:
    message_seq: int
    item_kind: str
    content: JsonValue
    tool_name: "str | None" = None
    tool_call_id: "str | None" = None

    def __post_init__(self) -> None:
        if self.message_seq < 1 or not isinstance(self.item_kind, str) or not self.item_kind:
            raise ValueError("session history item is invalid")


@dataclass(frozen=True, slots=True)
class SessionTurnItem:
    ordinal: int
    item_kind: str
    content: JsonValue
    tool_name: "str | None" = None
    tool_call_id: "str | None" = None

    def __post_init__(self) -> None:
        if self.ordinal < 1 or not self.item_kind:
            raise ValueError("session timeline item is invalid")


@dataclass(frozen=True, slots=True)
class SessionTurn:
    execution_id: str
    status: ExecutionStatus
    created_at: datetime
    updated_at: datetime
    user_input: JsonValue
    conversation_committed: bool
    items: tuple[SessionTurnItem, ...]
    error_code: "str | None" = None
    safe_error_details: "Mapping[str, JsonValue]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "safe_error_details", dict(self.safe_error_details))


class ExecutionHistoryReader(Protocol):
    async def history(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: "str | None",
        limit: int,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        step_index: int | None = None,
        tool_call_id: str | None = None,
        message_seq: int | None = None,
        part_index: int | None = None,
    ) -> Page[ExecutionHistoryItem]: ...

    async def trace(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: str | None,
        limit: int,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        step_index: int | None = None,
        tool_call_id: str | None = None,
    ) -> "Page[ExecutionTraceItem]": ...

    async def transcript(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: str | None,
        limit: int,
    ) -> Page[TranscriptItem]: ...

    async def model_interactions(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: "str | None",
        limit: int,
        cutoffs: "tuple[UsageReadCutoff, ...] | None" = None,
        include_content: bool = True,
    ) -> Page[ModelInteractionItem]: ...

    async def attachment_facts(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: "str | None",
        limit: int,
    ) -> Page[AttachmentFact]: ...

    async def usage(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cutoffs: "tuple[UsageReadCutoff, ...] | None" = None,
    ) -> UsageSummary: ...


class SessionHistoryReader(Protocol):
    async def history(
        self,
        session_id: str,
        *,
        tenant_id: str,
        continuation_agent_run_id: "str | None",
        continuation_history_id: "str | None" = None,
        cursor: "str | None",
        limit: int,
    ) -> "Page[SessionHistoryItem]": ...


@dataclass(frozen=True, slots=True)
class CreateSessionRequest:
    principal: Principal
    session_id: str
    idempotency_key: str
    cwd: "str | None" = None
    metadata: "Mapping[str, JsonValue]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class ListSessionRequest:
    principal: Principal
    cursor: "str | None" = None
    limit: int = 100


@dataclass(frozen=True, slots=True)
class ListExecutionRequest:
    principal: Principal
    session_id: str | None = None
    agent_id: str | None = None
    parent_execution_id: str | None = None
    cursor: str | None = None
    limit: int = 100

    def __post_init__(self) -> None:
        validate_page_limit(self.limit)


@dataclass(frozen=True, slots=True)
class ResumeSessionRequest:
    principal: Principal
    user_prompt: "UserPromptInput"
    idempotency_key: str
    memory_scope: "str | None"
    mode: ExecutionMode
    planning: bool
    thinking: ThinkingValue
    correlation: CorrelationData = field(default_factory=dict)
    files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_prompt", validate_user_input(self.user_prompt))
        files = normalize_input_files(self.files)
        validate_idempotency_key(self.idempotency_key)
        if self.memory_scope is not None:
            validate_memory_scope(self.memory_scope)
        mode = normalize_execution_mode(self.mode)
        thinking = normalize_thinking(self.thinking)
        if not isinstance(self.planning, bool) or mode == "plan" and not self.planning:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "thinking", thinking)
        object.__setattr__(self, "correlation", _request_correlation(self.correlation))
        object.__setattr__(self, "files", files)


@dataclass(frozen=True, slots=True)
class ForkSessionRequest:
    principal: Principal
    new_session_id: str
    idempotency_key: str = ""
    cwd: "str | None" = None

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class UpdateSessionRequest:
    principal: Principal
    expected_revision: int
    idempotency_key: str
    metadata: "Mapping[str, JsonValue]"
    cwd: "str | None" = None

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class CloseSessionRequest:
    principal: Principal
    idempotency_key: str
    force: bool = False
    wait_timeout_seconds: int = 30

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)
        if not 1 <= self.wait_timeout_seconds <= 300:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


@dataclass(frozen=True, slots=True)
class SessionView:
    session_id: str
    agent_id: str
    status: SessionStatus
    revision: int = 0
    cwd: "str | None" = None
    active_execution_id: "str | None" = None
    metadata: "Mapping[str, JsonValue]" = field(default_factory=dict)
    history_quality: str = "complete"


@dataclass(frozen=True, slots=True)
class ApprovalView:
    approval_id: str
    status: ApprovalStatus
    tool_name: str | None = None
    arguments: JsonValue = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ApprovalDecisionRequest:
    principal: Principal
    approval_id: str
    idempotency_key: str
    decision: ApprovalDecision
    message: "str | None" = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_resource_id(self.approval_id)
        validate_idempotency_key(self.idempotency_key)
        if self.message is not None and not isinstance(self.message, str):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        try:
            object.__setattr__(self, "metadata", dict(self.metadata))
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error


@dataclass(frozen=True, slots=True)
class ApprovalDecisionResult:
    approval_id: str
    idempotency_key: str
    decision: ApprovalDecision


@dataclass(frozen=True, slots=True)
class ExternalCallSucceeded:
    value: JsonValue


@dataclass(frozen=True, slots=True)
class ExternalCallRetry:
    message: str


@dataclass(frozen=True, slots=True)
class ExternalCallFailed:
    message: str


ExternalResolution = ExternalCallSucceeded | ExternalCallRetry | ExternalCallFailed


@dataclass(frozen=True, slots=True)
class ExternalCallView:
    call_id: str
    status: str
    tool_name: str
    arguments: JsonValue
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExternalSupplyRequest:
    principal: Principal
    call_id: str
    idempotency_key: str
    resolution: ExternalResolution
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_resource_id(self.call_id)
        validate_idempotency_key(self.idempotency_key)
        if not isinstance(
            self.resolution,
            (ExternalCallSucceeded, ExternalCallRetry, ExternalCallFailed),
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if isinstance(
            self.resolution,
            (ExternalCallRetry, ExternalCallFailed),
        ) and not self.resolution.message:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        try:
            object.__setattr__(self, "metadata", dict(self.metadata))
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error


@dataclass(frozen=True, slots=True)
class ExternalSupplyResult:
    call_id: str
    idempotency_key: str
    accepted: bool


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    execution_id: str
    event_seq: int
    event_type: str
    payload: JsonValue

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, str) or not self.event_type:
            raise ValueError("execution event type is required")


@dataclass(frozen=True, slots=True)
class ExecutionStreamEvent:
    execution_id: str
    durable_seq: int | None
    event_type: str
    payload: JsonValue

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, str) or not self.event_type:
            raise ValueError("execution stream event type is required")


@dataclass(frozen=True, slots=True)
class ExecutionTreeEvent:
    """Preserve real lineage with depth relative to the selected watch root."""

    execution_id: str
    agent_id: str | None
    lineage_kind: ExecutionLineageKind
    parent_execution_id: str | None
    root_execution_id: str
    parent_invocation_id: str | None
    depth: int
    event: ExecutionStreamEvent
    cursor: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.execution_id, str)
            or not self.execution_id
            or not isinstance(self.root_execution_id, str)
            or not self.root_execution_id
            or (
                self.agent_id is not None
                and (not isinstance(self.agent_id, str) or not self.agent_id)
            )
        ):
            raise ValueError("execution tree event identity is invalid")
        if not isinstance(self.lineage_kind, ExecutionLineageKind):
            raise TypeError("execution tree event lineage kind is invalid")
        if (
            isinstance(self.depth, bool)
            or not isinstance(self.depth, int)
            or self.depth < 0
        ):
            raise ValueError("execution tree event depth must be nonnegative")
        if not isinstance(self.event, ExecutionStreamEvent):
            raise TypeError("execution tree event requires an execution event")
        if self.cursor is not None and (
            not isinstance(self.cursor, str) or not self.cursor
        ):
            raise ValueError("execution tree event cursor is invalid")
        if self.execution_id != self.event.execution_id:
            raise ValueError("execution tree event identity does not match execution")
        if self.lineage_kind is ExecutionLineageKind.SUBAGENT:
            if (
                not isinstance(self.parent_execution_id, str)
                or not self.parent_execution_id
                or self.parent_execution_id == self.execution_id
                or not isinstance(self.parent_invocation_id, str)
                or not self.parent_invocation_id
                or self.root_execution_id == self.execution_id
            ):
                raise ValueError("subagent execution tree event lineage is invalid")
        elif (
            self.depth != 0
            or self.parent_execution_id is not None
            or self.parent_invocation_id is not None
        ):
            raise ValueError("root execution tree event lineage is invalid")


@dataclass(frozen=True, slots=True)
class TaskGraphRunEvent:
    graph_id: str
    node_id: "str | None"
    event: "TaskEvent | ExecutionTreeEvent"
    cursor: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.graph_id, str) or not self.graph_id.strip():
            raise ValueError("task graph run event graph id is required")
        if self.node_id is not None and (
            not isinstance(self.node_id, str) or not self.node_id.strip()
        ):
            raise ValueError("task graph run event node id is invalid")
        if self.cursor is not None and (
            not isinstance(self.cursor, str) or not self.cursor
        ):
            raise ValueError("task graph run event cursor is invalid")
        if isinstance(self.event, TaskEvent):
            if (
                self.event.graph_id != self.graph_id
                or self.event.node_id != self.node_id
            ):
                raise ValueError("task graph run event task identity is invalid")
            return
        if isinstance(self.event, ExecutionTreeEvent):
            if self.node_id is None:
                raise ValueError("task run execution event requires a node id")
            return
        raise TypeError("task graph run event payload is invalid")


@dataclass(frozen=True, slots=True)
class ArtifactView:
    artifact_id: str
    execution_id: str
    size: int


@dataclass(frozen=True, slots=True)
class ArtifactDownload:
    artifact_id: str
    url: str
    expires_at: str


class ExecutionHistoryService(Protocol):
    async def inspect(
        self, execution_id: str, *, principal: Principal
    ) -> ExecutionView: ...

    async def list(
        self, request: ListExecutionRequest
    ) -> "Page[ExecutionView]": ...

    async def trace(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        step_index: int | None = None,
        tool_call_id: str | None = None,
    ) -> "Page[ExecutionTraceItem]": ...

    async def transcript(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
    ) -> Page[TranscriptItem]: ...

    async def history(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        step_index: int | None = None,
        tool_call_id: str | None = None,
        message_seq: int | None = None,
        part_index: int | None = None,
    ) -> "Page[ExecutionHistoryItem]": ...

    async def model_interactions(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
        cutoffs: "tuple[UsageReadCutoff, ...] | None" = None,
    ) -> "Page[ModelInteractionItem]": ...

    async def attachment_facts(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> "Page[AttachmentFact]": ...

    async def usage(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cutoffs: "tuple[UsageReadCutoff, ...] | None" = None,
    ) -> UsageSummary: ...


class ExecutionService(Protocol):
    async def acquire_dependency_hold(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        hold_id: str,
    ) -> bool: ...

    async def release_dependency_hold(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        hold_id: str,
    ) -> None: ...

    async def request_terminal_handoff(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> None: ...

    async def start(
        self,
        binding_digest: str,
        request: ExecutionRequest,
        *,
        dependency_hold_id: "str | None" = None,
        binding_contract: "AgentBindingContract | None" = None,
        requires_task_invocation_capture: bool = False,
    ) -> ExecutionHandle: ...
    async def start_task(
        self,
        binding: TaskBindingContract,
        *,
        principal: Principal,
        input: Mapping[str, JsonValue],
        idempotency_key: str,
        correlation: Mapping[str, str | int],
        requires_task_invocation_capture: bool = False,
    ) -> ExecutionHandle: ...

    async def claim_task_attempt(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionView: ...

    async def schedule_task_retry(
        self,
        execution_id: str,
        *,
        principal: Principal,
        error_code: str,
        attempt: "ExecutionView | None" = None,
    ) -> ExecutionView: ...

    async def defer_task_input(
        self,
        execution_id: str,
        *,
        principal: Principal,
        wait_id: str,
    ) -> ExecutionView: ...

    async def supply_task_input(
        self,
        execution_id: str,
        *,
        principal: Principal,
        value: JsonValue,
    ) -> ExecutionView: ...

    async def resume_task_not_applied(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionView: ...

    async def resolve_task_effect(
        self,
        execution_id: str,
        *,
        principal: Principal,
        resolution: TaskEffectResolution,
    ) -> ExecutionView: ...

    async def complete_task(
        self,
        execution_id: str,
        *,
        principal: Principal,
        output: JsonValue,
        attempt: "ExecutionView | None" = None,
    ) -> ExecutionResult: ...

    async def fail_task(
        self,
        execution_id: str,
        *,
        principal: Principal,
        error: AIError,
        attempt: "ExecutionView | None" = None,
    ) -> ExecutionResult: ...

    async def require_task_recovery(
        self,
        execution_id: str,
        *,
        principal: Principal,
        error_code: str,
        attempt: "ExecutionView | None" = None,
    ) -> ExecutionView: ...
    async def cancel_task(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> CancelExecutionResult: ...


    async def resolve_existing(
        self,
        binding_digest: str,
        request: ExecutionRequest,
        *,
        binding_contract: "AgentBindingContract | None" = None,
        requires_task_invocation_capture: bool = False,
    ) -> "ExecutionHandle | None": ...
    async def inspect(
        self, execution_id: str, *, principal: Principal
    ) -> ExecutionView: ...
    async def list(
        self, request: ListExecutionRequest
    ) -> "Page[ExecutionView]": ...
    async def list_children(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> "tuple[ExecutionView, ...]": ...
    async def result(
        self, execution_id: str, *, principal: Principal
    ) -> ExecutionResult: ...
    async def result_payload_size(
        self, execution_id: str, *, principal: Principal
    ) -> int: ...
    async def wait(
        self,
        execution_id: str,
        *,
        principal: Principal,
        timeout_seconds: "float | None" = None,
    ) -> ExecutionResult: ...
    async def run(
        self,
        binding_digest: str,
        request: ExecutionRequest,
        *,
        timeout_seconds: "float | None" = None,
        binding_contract: "AgentBindingContract | None" = None,
    ) -> ExecutionResult: ...
    async def retry(
        self, execution_id: str, request: RetryExecutionRequest
    ) -> ExecutionHandle: ...
    async def fork(
        self, execution_id: str, request: ForkExecutionRequest
    ) -> ExecutionHandle: ...
    async def cancel(
        self, execution_id: str, request: CancelExecutionRequest
    ) -> CancelExecutionResult: ...
    async def recovery_effects(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> tuple[ExecutionRecoveryEffect, ...]: ...
    async def resolve_tool_effect(
        self,
        execution_id: str,
        request: ResolveToolEffectRequest,
    ) -> ToolEffectResolutionResult: ...
    async def recover(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionHandle: ...
    async def trace(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        step_index: int | None = None,
        tool_call_id: str | None = None,
    ) -> "Page[ExecutionTraceItem]": ...
    async def transcript(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
    ) -> "Page[TranscriptItem]": ...
    async def history(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        step_index: int | None = None,
        tool_call_id: str | None = None,
        message_seq: int | None = None,
        part_index: int | None = None,
    ) -> "Page[ExecutionHistoryItem]": ...

    async def model_interactions(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
        cutoffs: "tuple[UsageReadCutoff, ...] | None" = None,
    ) -> "Page[ModelInteractionItem]": ...


class SessionService(Protocol):
    async def create(
        self, agent_id: str, request: CreateSessionRequest
    ) -> SessionView: ...
    async def get(self, session_id: str, *, principal: Principal) -> SessionView: ...
    async def reconcile(
        self,
        session_id: str,
        *,
        principal: Principal,
    ) -> SessionView: ...
    async def list(self, request: ListSessionRequest) -> "Page[SessionView]": ...
    async def history(
        self,
        session_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> "Page[SessionHistoryItem]": ...
    async def timeline(
        self,
        session_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> "Page[SessionTurn]": ...
    async def resume(
        self,
        agent_id: str,
        binding_digest: str,
        session_id: str,
        request: ResumeSessionRequest,
        *,
        binding_contract: "AgentBindingContract | None" = None,
        dependency_hold_id: "str | None" = None,
        requires_task_invocation_capture: bool = False,
    ) -> ExecutionHandle: ...
    async def fork(
        self, agent_id: str, session_id: str, request: ForkSessionRequest
    ) -> SessionView: ...
    async def update(
        self, agent_id: str, session_id: str, request: UpdateSessionRequest
    ) -> SessionView: ...
    async def close(
        self, session_id: str, request: CloseSessionRequest
    ) -> SessionView: ...


class ApprovalService(Protocol):
    async def list(
        self, execution_id: str, *, principal: Principal
    ) -> "tuple[ApprovalView, ...]": ...
    async def decide(
        self, execution_id: str, request: ApprovalDecisionRequest
    ) -> ApprovalDecisionResult: ...


class ExternalService(Protocol):
    async def list(
        self, execution_id: str, *, principal: Principal
    ) -> "tuple[ExternalCallView, ...]": ...

    async def supply(
        self, execution_id: str, request: ExternalSupplyRequest
    ) -> ExternalSupplyResult: ...


class EventService(Protocol):
    async def list(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_event_seq: int = 0,
        limit: int = 100,
    ) -> "Page[ExecutionEvent]": ...

    def stream(
        self, execution_id: str, *, principal: Principal, after_event_seq: int = 0
    ) -> "AsyncIterator[ExecutionStreamEvent]": ...


class ArtifactService(Protocol):
    async def list(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> "Page[ArtifactView]": ...
    async def get(
        self, artifact_id: str, *, principal: Principal
    ) -> ArtifactDownload: ...


__all__ = [
    "ApprovalDecisionRequest",
    "AttachmentFact",
    "ApprovalDecisionResult",
    "ApprovalService",
    "ApprovalView",
    "ArtifactDownload",
    "ArtifactService",
    "ArtifactView",
    "CancelExecutionRequest",
    "CancelExecutionResult",
    "CloseSessionRequest",
    "CreateSessionRequest",
    "EventService",
    "ExternalCallFailed",
    "ExternalCallRetry",
    "ExternalCallSucceeded",
    "ExternalCallView",
    "ExternalResolution",
    "ExecutionEvent",
    "ExecutionHandle",
    "ExecutionHistoryItem",
    "ExecutionHistoryReader",
    "ExecutionHistoryService",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutionService",
    "ExecutionStreamEvent",
    "ExecutionTraceItem",
    "ExecutionTreeEvent",
    "ExecutionView",
    "ExternalService",
    "ExternalSupplyRequest",
    "ExternalSupplyResult",
    "ForkExecutionRequest",
    "ForkSessionRequest",
    "ListExecutionRequest",
    "ListSessionRequest",
    "ModelInteractionItem",
    "Page",
    "ResumeSessionRequest",
    "RetryExecutionRequest",
    "SessionHistoryItem",
    "SessionHistoryReader",
    "SessionService",
    "SessionTurn",
    "SessionTurnItem",
    "SessionView",
    "TaskGraphRunEvent",
    "TranscriptItem",
    "UpdateSessionRequest",
    "UsageReadCutoff",
    "UsageSummary",
    "project_execution_view",
]
