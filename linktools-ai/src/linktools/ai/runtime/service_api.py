#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime service protocols and transport-neutral request values."""

from ..execution import (CancelExecutionRequest, CancelExecutionResult, ExecutionHandle, ExecutionView, project_execution_view, ExecutionResult)

from ..session import (SessionHistoryItem, SessionTurnItem, SessionTurn, SessionHistoryReader, CreateSessionRequest, ListSessionRequest, ResumeSessionRequest, ForkSessionRequest, UpdateSessionRequest, CloseSessionRequest, SessionView, SessionDetail)

from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Protocol, cast

from ..agent import AgentBinding, AgentBindingContract
from ..asset import AssetVersionRef
from ..core import (
    BudgetUsage,
    RunBudget,
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
from ..model import ModelClientRequest
from ..task import TaskBindingContract, TaskEffectResolution, TaskEvent, TaskGraphInfo
from ._execution_context import ExecutionInputContext
from ..model import (
    UserPromptInput,
    normalize_input_files,
    validate_user_input,
)
from .recovery import (
    ExecutionRecoveryEffect,
    ResolveToolEffectRequest,
    ToolEffectResolutionResult,
)


if TYPE_CHECKING:
    from .state._contracts import ResultRecord


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
    budget: RunBudget | None = None
    tool_response_ref: AssetVersionRef | None = None
    # Ephemeral admission state; never included in durable request projections.
    resolved_binding: AgentBinding | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.tool_response_ref is not None and not isinstance(self.tool_response_ref, AssetVersionRef):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self.budget is not None and not isinstance(self.budget, RunBudget):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
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
    budget: RunBudget | None = None

    def __post_init__(self) -> None:
        if self.budget is not None and not isinstance(self.budget, RunBudget):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        object.__setattr__(self, "user_prompt", validate_user_input(self.user_prompt))
        files = normalize_input_files(self.files)
        validate_idempotency_key(self.idempotency_key)
        object.__setattr__(self, "correlation", _request_correlation(self.correlation))
        object.__setattr__(self, "files", files)


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
class ModelStreamAttemptItem:
    """One-based provider stream within a logical model request.

    SUCCEEDED means this transport segment completed, not that it alone owns
    the logical response. Native continuation can complete several segments.
    Client request bounds select a zero-based, half-open range of the original
    client request array, not stream attempt identities. Only failed/cancelled
    partial content appears here; the canonical response stays on the interaction.
    """

    stream_attempt_seq: int
    status: str
    started_at: datetime
    finished_at: datetime
    duration_ns: int
    client_request_start: int
    client_request_end: int
    response: JsonValue | None
    usage: UsageMetrics | None
    error_code: str | None
    error_type: str | None
    retry_scheduled: bool

    def __post_init__(self) -> None:
        if (self.stream_attempt_seq < 1 or self.duration_ns < 0
            or self.client_request_start < 0 or self.client_request_end < self.client_request_start
            or self.status not in {"SUCCEEDED", "FAILED", "CANCELLED"}
            or self.status == "SUCCEEDED" and self.response is not None
            or self.retry_scheduled and self.status != "FAILED"):
            raise ValueError("model stream attempt item is invalid")
        object.__setattr__(self, "response", deepcopy(self.response))


@dataclass(frozen=True, slots=True)
class ModelInteractionItem:
    """SDK projections and independently selected original client attempts.

    With client_requests_included set, None means no capture was recorded;
    an empty tuple means the scope observed no supported client dispatch.
    stream_attempts similarly distinguishes unobserved history (None) from an
    observed scope without stream attempts (()). include_content controls all
    SDK response bodies, including failed stream partials, but not metadata.
    """

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
    client_requests: tuple[ModelClientRequest, ...] | None = None
    client_requests_included: bool = False
    stream_attempts: tuple[ModelStreamAttemptItem, ...] | None = None

    def __post_init__(self) -> None:
        if (
            not self.execution_id
            or self.agent_run_seq < 1
            or self.depth < 0
            or self.model_request_seq < 1
            or self.step_index < 0
            or self.status not in {"RUNNING", "SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED"}
            or self.duration_ns is not None and self.duration_ns < 0
        ):
            raise ValueError("model interaction item is invalid")
        if not isinstance(self.content_included, bool):
            raise TypeError("model interaction content flag must be bool")
        if not self.content_included and (self.request or self.response is not None):
            raise ValueError("omitted model interaction content must be empty")
        if not isinstance(self.client_requests_included, bool):
            raise TypeError("client request content flag must be bool")
        if not self.client_requests_included and self.client_requests is not None:
            raise ValueError("omitted client request content must be empty")
        if self.client_requests is not None and (
            not isinstance(self.client_requests, tuple)
            or any(not isinstance(value, ModelClientRequest) for value in self.client_requests)
        ):
            raise TypeError("client request content is invalid")
        if self.status in {"RUNNING", "INTERRUPTED"} and (
            self.started_at is None
            or self.response is not None
            or self.finished_at is not None
            or self.duration_ns is not None
            or self.usage is not None
            or self.error_code is not None
        ):
            raise ValueError("unfinished model interaction has terminal data")
        if self.stream_attempts is not None and (
            not isinstance(self.stream_attempts, tuple)
            or any(not isinstance(value, ModelStreamAttemptItem) for value in self.stream_attempts)
            or not self.content_included and any(value.response is not None for value in self.stream_attempts)
        ):
            raise ValueError("model stream attempt content is invalid")
        object.__setattr__(self, "model", deepcopy(dict(self.model)))
        object.__setattr__(self, "request", deepcopy(dict(self.request)))
        object.__setattr__(self, "response", deepcopy(self.response))


@dataclass(frozen=True, slots=True)
class AttachmentFact:
    execution_id: str
    attachment_id: str
    fact: str
    source: str
    mime_type: str | None
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
            or self.mime_type is not None
            and (not isinstance(self.mime_type, str) or not self.mime_type)
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
class ModelInteractionReadBoundary:
    """Captured admitted identities; durable availability covers active runs too."""

    cutoffs: tuple[UsageReadCutoff, ...]
    durable_cutoffs: tuple[UsageReadCutoff, ...]
    local_staging_available: bool
    durable_history_available: bool


class ModelInteractionSubscription(Protocol):
    @property
    def generation(self) -> int: ...

    async def wait(self, after_generation: int) -> int: ...

    async def close(self) -> None: ...


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
    running_requests: int = 0
    interrupted_requests: int = 0

    def __post_init__(self) -> None:
        counts = (
            self.logical_requests,
            self.succeeded_requests,
            self.failed_requests,
            self.cancelled_requests,
            self.running_requests,
            self.interrupted_requests,
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
            + self.running_requests
            + self.interrupted_requests
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


class ExecutionHistoryReader(Protocol):
    async def capture_model_interaction_cutoffs(
        self, execution_id: str, *, tenant_id: str,
    ) -> ModelInteractionReadBoundary: ...

    async def read_model_interaction_metadata(
        self, execution_id: str, *, tenant_id: str, agent_run_seq: int,
        after_model_request_seq: int, through_model_request_seq: int, limit: int = 200,
    ) -> tuple[ModelInteractionItem, ...]: ...

    async def subscribe_model_interactions(
        self, execution_id: str, *, tenant_id: str,
    ) -> ModelInteractionSubscription | None: ...

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
        include_client_requests: bool = False,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
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
class TaskGraphProjectionCoverage:
    """Finite read coverage; successful callback ACK accepts this checkpoint.

    Positions describe source reads, never a durable replay cursor. Local active
    visibility applies only to the producing Runtime, not other processes.
    """

    execution_event_seqs: Mapping[str, Mapping[str, int]]
    model_cutoffs: tuple["UsageReadCutoff", ...]
    durable_model_cutoffs: tuple["UsageReadCutoff", ...]
    local_active_execution_ids: tuple[str, ...]
    unavailable_active_execution_ids: tuple[str, ...]
    unavailable_execution_ids: tuple[str, ...]
    durable_events_complete: bool
    state_complete: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution_event_seqs", {
            node: dict(values) for node, values in self.execution_event_seqs.items()
        })


@dataclass(frozen=True, slots=True)
class TaskGraphProjection:
    """A metadata-only replacement view at graph.event_seq, not an event ACK."""

    graph: TaskGraphInfo
    phase: Literal["initial", "update", "final"]
    observed_at: datetime
    coverage: TaskGraphProjectionCoverage | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.graph, TaskGraphInfo):
            raise TypeError("graph projection requires a safe graph view")
        if self.phase not in {"initial", "update", "final"}:
            raise ValueError("graph projection phase is invalid")
        if self.observed_at.tzinfo is None:
            raise ValueError("graph projection time requires a timezone")


@dataclass(frozen=True, slots=True)
class TaskModelProjection:
    """Upsert by (execution_id, agent_run_seq, model_request_seq).

    Terminal facts dominate RUNNING; duplicates assign own usage rather than
    adding it. Neither a projection nor local staging advances a replay cursor.
    """

    item: ModelInteractionItem
    root_execution_id: str
    parent_execution_id: str | None
    parent_invocation_id: str | None
    visibility: Literal["local_staging", "durable_history"]
    observed_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.item, ModelInteractionItem) or self.item.content_included:
            raise ValueError("model projection requires metadata-only history")
        if self.visibility not in {"local_staging", "durable_history"}:
            raise ValueError("model projection visibility is invalid")
        if self.observed_at.tzinfo is None:
            raise ValueError("model projection time requires a timezone")


@dataclass(frozen=True, slots=True)
class ExecutionObservationEvent:
    """Execution-tree or metadata delivery with an independent observation cursor.

    The nested tree cursor is a tree-only checkpoint. Model metadata
    reuses the outer cursor and is compensated again on every reconnection.
    """

    event: ExecutionTreeEvent | TaskModelProjection
    cursor: str | None


@dataclass(frozen=True, slots=True)
class TaskGraphRunEvent:
    graph_id: str
    node_id: "str | None"
    event: "TaskEvent | ExecutionTreeEvent | TaskGraphProjection | TaskModelProjection"
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
        if isinstance(self.event, TaskGraphProjection):
            if self.event.graph.graph_id != self.graph_id or self.node_id is not None:
                raise ValueError("task graph projection identity is invalid")
            return
        if isinstance(self.event, TaskModelProjection):
            if self.node_id is None:
                raise ValueError("task model projection requires a node id")
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
    async def capture_model_interaction_cutoffs(
        self, execution_id: str, *, principal: Principal,
    ) -> ModelInteractionReadBoundary: ...

    async def read_model_interaction_metadata(
        self, execution_id: str, *, principal: Principal, agent_run_seq: int,
        after_model_request_seq: int, through_model_request_seq: int, limit: int = 200,
    ) -> tuple[ModelInteractionItem, ...]: ...

    async def subscribe_model_interactions(
        self, execution_id: str, *, principal: Principal,
    ) -> ModelInteractionSubscription | None: ...

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
        include_client_requests: bool = False,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
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
        resolved_binding: "AgentBinding | None" = None,
        requires_task_invocation_capture: bool = False,
        budget_scope_id: str | None = None,
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
        budget_scope_id: str | None = None,
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
        budget_scope_id: str | None = None,
    ) -> "ExecutionHandle | None": ...
    async def budget_usage(
        self, execution_id: str, *, principal: Principal,
    ) -> BudgetUsage | None: ...

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
        idempotency_key: str | None = None,
    ) -> ExecutionHandle: ...
    async def trace(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
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
        include_client_requests: bool = False,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
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
    async def refresh(
        self,
        session_id: str,
        *,
        principal: Principal,
        cursor: str | None = None,
        limit: int = 100,
        max_age: float = 0,
    ) -> SessionDetail:
        """Read a complete display; positive max_age permits bounded stale display values."""
        ...
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
        resolved_binding: "AgentBinding | None" = None,
        dependency_hold_id: "str | None" = None,
        requires_task_invocation_capture: bool = False,
        budget_scope_id: str | None = None,
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
    "ExecutionObservationEvent",
    "ExecutionView",
    "ExternalService",
    "ExternalSupplyRequest",
    "ExternalSupplyResult",
    "ForkExecutionRequest",
    "ForkSessionRequest",
    "ListExecutionRequest",
    "ListSessionRequest",
    "ModelInteractionItem",
    "ModelStreamAttemptItem",
    "ModelInteractionReadBoundary",
    "ModelInteractionSubscription",
    "Page",
    "ResumeSessionRequest",
    "RetryExecutionRequest",
    "SessionDetail",
    "SessionHistoryItem",
    "SessionHistoryReader",
    "SessionService",
    "SessionTurn",
    "SessionTurnItem",
    "SessionView",
    "TaskGraphRunEvent",
    "TaskGraphProjection",
    "TaskGraphProjectionCoverage",
    "TaskModelProjection",
    "TranscriptItem",
    "UpdateSessionRequest",
    "UsageReadCutoff",
    "UsageSummary",
    "project_execution_view",
]
