#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution-owned durable records and state transition contracts."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING
from ..agent import AgentBindingContract
from ..asset import AssetVersionRef
from ..core import (PersistenceDomain, RunBudget, ExecutionEventType, ExecutionLineageKind,
    ExecutionMode, ExecutionStatus, IdempotencyStatus, JsonValue, OperationStatus,
    ResourceKind, StopReason, ThinkingValue, UsageMetrics, canonical_sha256, validate_idempotency_key, validate_resource_id,
    normalize_correlation, normalize_execution_mode, normalize_thinking, normalize_tool_effects, ToolEffects)
from ..session import SessionContext, normalize_session_context
from ..errors import AIError, ErrorDiagnostics
from ..storage import StoredPayload
from ..steps import NativeStepSource
from ..spec import CapabilityRef
from ..model import StoredUserInput
from ..task import TaskBindingContract
from ..session import ConversationCursor


if TYPE_CHECKING:
    from ..session import SessionRecord


@dataclass(frozen=True, slots=True)
class RuntimePayloadRef:
    payload: StoredPayload
    source_domain: PersistenceDomain | None


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ExecutionRecord:
    execution_id: str
    session_id: str | None
    parent_execution_id: str | None
    root_execution_id: str
    previous_execution_id: str | None
    fork_base_execution_id: str | None
    lineage_kind: ExecutionLineageKind
    status: ExecutionStatus
    revision: int
    event_seq: int
    agent_run_seq: int
    error_code: str | None
    safe_error_details: Mapping[str, JsonValue]
    created_at: datetime
    updated_at: datetime
    mode: ExecutionMode | None
    planning: bool | None
    thinking: ThinkingValue | None
    binding: AgentBindingContract | TaskBindingContract
    principal_id: str
    principal_kind: str
    stored_user_input: StoredUserInput
    producer_generation: int = 0
    producer_claim_id: str | None = None
    parent_invocation_id: str | None = None
    memory_scope: str | None = None
    conversation_agent_run_id: str | None = None
    result: ResultRecord | None = None
    repository_instructions: RuntimePayloadRef | None = None
    input_context: RuntimePayloadRef | None = None
    context_imported: bool = False
    error_diagnostics: ErrorDiagnostics | None = None
    correlation: Mapping[str, str | int] = field(default_factory=dict)
    task_attempt: int = 0
    task_deadline_at: datetime | None = None
    task_next_attempt_at: datetime | None = None
    dependency_hold_ids: tuple[str, ...] = ()
    # Admission provenance persists even if invocation storage fails or holds are released.
    requires_task_invocation_capture: bool = False
    # Monotonic publication proof: retained dependencies are confirmed, including
    # the Conversation turn receipt when admitted with retention enabled.
    # Never-admitted executions, disabled retention and sessionless runs are exempt.
    retention_closed: bool = False
    started_at: datetime | None = None
    budget_scope_id: str | None = None
    tool_response_ref: AssetVersionRef | None = None
    tool_effects: ToolEffects = "live"
    session_context: SessionContext = "all"
    native_step_source: NativeStepSource | None = None
    restart_checkpoints: bool = False
    durable_trace: bool = False
    # Whether this execution retains conversation and fork context.
    conversation: bool = True
    recovery: ExecutionRecoveryDetail | None = None
    history_seal_digest: str | None = None
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.completed_at is not None and (not isinstance(self.completed_at, datetime)
                or self.completed_at.tzinfo is None or self.completed_at.utcoffset() is None):
            raise ValueError("execution completed_at must be timezone-aware")
        if self.completed_at is not None:
            _validate_terminal(self)
        if self.native_step_source is not None and (
                not isinstance(self.native_step_source, NativeStepSource) or not self.restart_checkpoints):
            raise ValueError("native checkpoint source requires checkpoint retention")
        normalize_tool_effects(self.tool_effects)
        normalize_session_context(self.session_context)
        if (isinstance(self.producer_generation, bool)
                or not isinstance(self.producer_generation, int)
                or self.producer_generation < 0):
            raise ValueError("execution producer generation is invalid")
        if self.producer_claim_id is not None and (
                not isinstance(self.producer_claim_id, str) or not self.producer_claim_id
                or self.producer_generation < 1):
            raise ValueError("execution producer claim is invalid")
        if any(not isinstance(value, bool) for value in (
            self.restart_checkpoints, self.durable_trace, self.conversation,
        )):
            raise TypeError("execution persistence guarantees must be bool")
        if self.tool_response_ref is not None and not isinstance(self.tool_response_ref, AssetVersionRef):
            raise TypeError("execution tool response reference is invalid")
        if self.budget_scope_id is not None and (
            not isinstance(self.budget_scope_id, str) or not self.budget_scope_id.strip()
        ):
            raise ValueError("execution budget scope is invalid")
        if not isinstance(self.context_imported, bool) or (self.input_context is not None and not isinstance(self.input_context, RuntimePayloadRef)):
            raise TypeError("execution input context is invalid")
        if self.context_imported and (self.input_context is None or self.session_id is not None):
            raise ValueError("imported context must belong to an independent execution")
        if self.recovery is not None and not isinstance(self.recovery, ExecutionRecoveryDetail):
            raise TypeError("execution recovery detail is invalid")
        if self.history_seal_digest is not None and (not self.durable_trace or not isinstance(self.history_seal_digest, str) or len(self.history_seal_digest) != 64):
            raise ValueError("execution history seal reference is invalid")
        agent_binding = isinstance(self.binding, AgentBindingContract)
        task_binding = isinstance(self.binding, TaskBindingContract)
        if agent_binding == task_binding:
            raise TypeError("execution binding contract is invalid")
        if agent_binding:
            if self.status is ExecutionStatus.WAITING_DEFERRED and (self.recovery is None or self.recovery.pending_tools is None):
                raise ValueError("waiting agent execution requires deferred continuation")
            mode = normalize_execution_mode(self.mode)
            thinking = normalize_thinking(self.thinking)
            if not isinstance(self.planning, bool):
                raise TypeError("agent execution planning must be bool")
            if mode == "plan" and not self.planning:
                raise ValueError("plan mode requires planning")
            if (
                self.task_attempt != 0
                or self.task_deadline_at is not None
                or self.task_next_attempt_at is not None
            ):
                raise ValueError("agent execution cannot carry task attempt state")
            object.__setattr__(self, "mode", mode)
            object.__setattr__(self, "thinking", thinking)
        else:
            if self.native_step_source is not None:
                raise ValueError("task execution cannot carry a native checkpoint source")
            if self.mode is not None or self.planning is not None or self.thinking is not None:
                raise ValueError("task execution cannot carry agent execution policy")
            if (
                self.session_id is not None
                or self.memory_scope is not None
                or self.input_context is not None
                or self.context_imported
                or self.tool_response_ref is not None
                or self.tool_effects != "live"
                or self.session_context != "all"
                or self.conversation_agent_run_id is not None
                or self.parent_execution_id is not None
                or self.parent_invocation_id is not None
                or self.lineage_kind is not ExecutionLineageKind.RUN
                or self.agent_run_seq != 0
            ):
                raise ValueError("task execution carries agent-only state")
            if (
                isinstance(self.task_attempt, bool)
                or not isinstance(self.task_attempt, int)
                or self.task_attempt < 0
                or self.task_attempt > self.binding.max_attempts
            ):
                raise ValueError("task execution attempt is invalid")
            for value in (self.task_deadline_at, self.task_next_attempt_at):
                if value is not None and value.tzinfo is None:
                    raise ValueError("task execution timestamps must be timezone-aware")
        if not isinstance(self.requires_task_invocation_capture, bool):
            raise TypeError("execution task invocation capture requirement must be bool")
        holds = tuple(self.dependency_hold_ids)
        if (
            any(not isinstance(value, str) or not value.strip() for value in holds)
            or len(set(holds)) != len(holds)
            or tuple(sorted(holds)) != holds
        ):
            raise ValueError("execution dependency holds must be sorted and unique")
        if not isinstance(self.retention_closed, bool):
            raise TypeError("execution retention_closed must be bool")
        if self.started_at is not None and self.started_at.tzinfo is None:
            raise ValueError("execution started_at must be timezone-aware")
        object.__setattr__(self, "dependency_hold_ids", holds)
        object.__setattr__(self, "correlation", normalize_correlation(self.correlation))
        if self.lineage_kind is ExecutionLineageKind.SUBAGENT:
            if (
                not isinstance(self.parent_execution_id, str)
                or not self.parent_execution_id
                or not isinstance(self.parent_invocation_id, str)
                or not self.parent_invocation_id
                or self.previous_execution_id is not None
                or self.fork_base_execution_id is not None
            ):
                raise ValueError("subagent execution lineage is invalid")
        elif self.parent_execution_id is not None or self.parent_invocation_id is not None:
            raise ValueError("non-subagent execution cannot carry parent lineage")
        if not isinstance(self.principal_id, str) or not self.principal_id:
            raise TypeError("execution principal id is invalid")
        if not isinstance(self.principal_kind, str) or not self.principal_kind:
            raise TypeError("execution principal kind is invalid")
        if not isinstance(self.stored_user_input, StoredUserInput):
            raise TypeError("execution stored user input is invalid")
        if self.error_diagnostics is not None and not isinstance(
            self.error_diagnostics, ErrorDiagnostics
        ):
            raise TypeError("execution error diagnostics are invalid")
        if (
            self.status is not ExecutionStatus.FAILED
            and self.error_diagnostics is not None
        ):
            raise ValueError("only failed execution can carry error diagnostics")

    def finalize_terminal(self, result: ResultRecord) -> ExecutionRecord:
        """Record completion once, independently of later control updates."""
        if self.completed_at is not None:
            if self.result != result:
                raise ValueError("execution terminal result is immutable")
            return self
        return replace(self, result=result, completed_at=self.updated_at)

    @property
    def recovery_state(self) -> ExecutionRecoveryState | None:
        detail = self.recovery
        if detail is None:
            return None
        if detail.handoff_phase is RecoveryHandoffPhase.PREPARED:
            return ExecutionRecoveryState.HANDOFF
        if detail.handoff_phase is RecoveryHandoffPhase.COMPLETED or self.status in {
            ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
        }:
            return ExecutionRecoveryState.COMPLETED
        if self.status is ExecutionStatus.WAITING_DEFERRED:
            return ExecutionRecoveryState.WAITING
        return ExecutionRecoveryState.ADMITTED if detail.agent_run_id is None else ExecutionRecoveryState.ACTIVE

    @property
    def invalidates_session_history(self) -> bool:
        """A completed unretained turn supersedes any older session context."""
        return (
            self.session_id is not None
            and not self.conversation
            and self.status is ExecutionStatus.SUCCEEDED
        )

    @property
    def binding_digest(self) -> str:
        return self.binding.binding_digest

    @property
    def binding_kind(self) -> str:
        return "agent" if isinstance(self.binding, AgentBindingContract) else "task"

    @property
    def agent_id(self) -> str | None:
        return (
            self.binding.agent_spec.id
            if isinstance(self.binding, AgentBindingContract)
            else None
        )

    @property
    def task_id(self) -> str | None:
        return (
            self.binding.id
            if isinstance(self.binding, TaskBindingContract)
            else None
        )


@dataclass(frozen=True, slots=True)
class ExecutionTimelineRecord:
    """Read-only projection of the Execution owner, without its executable binding."""
    execution_id: str
    session_id: str | None
    parent_execution_id: str | None
    principal_id: str
    principal_kind: str
    status: ExecutionStatus
    stored_user_input: StoredUserInput
    result: ResultRecord | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    error_code: str | None
    safe_error_details: Mapping[str, JsonValue]
    error_diagnostics: ErrorDiagnostics | None
    conversation: bool

    def __post_init__(self) -> None:
        if self.completed_at is not None:
            _validate_terminal(self)

    @property
    def invalidates_session_history(self) -> bool:
        return self.session_id is not None and not self.conversation and self.status is ExecutionStatus.SUCCEEDED

    def terminal_matches(self, value: ExecutionRecord) -> bool:
        """Terminal facts are owned once, while control metadata may still advance."""
        return (
            self.execution_id == value.execution_id
            and self.session_id == value.session_id
            and self.parent_execution_id == value.parent_execution_id
            and self.principal_id == value.principal_id and self.principal_kind == value.principal_kind
            and self.status == value.status and self.stored_user_input == value.stored_user_input
            and self.result == value.result and self.created_at == value.created_at
            and self.completed_at == value.completed_at and self.error_code == value.error_code
            and self.safe_error_details == value.safe_error_details
            and self.error_diagnostics == value.error_diagnostics
            and self.conversation == value.conversation
        )


def _validate_terminal(value: ExecutionRecord | ExecutionTimelineRecord) -> None:
    if (value.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}
            or value.result is None or value.completed_at is None
            or value.completed_at.tzinfo is None):
        raise ValueError("execution terminal result and completion timestamp are required")
    if (value.result.output is not None) != (value.status is ExecutionStatus.SUCCEEDED):
        raise ValueError("execution terminal output does not match status")


@dataclass(frozen=True, slots=True)
class ExecutionStartClaim:
    execution_id: str
    expected_revision: int
    expected_event_seq: int
    scope: str
    idempotency_key: str = field(repr=False)
    request_digest: str
    started_at: datetime
    request_identity_version: int = 1

    def __post_init__(self) -> None:
        try:
            validate_resource_id(self.scope)
            validate_idempotency_key(self.idempotency_key)
        except AIError as error:
            raise ValueError("idempotency identity is invalid") from error


@dataclass(frozen=True, slots=True)
class ExecutionStartUnknownCommit:
    execution_id: str
    expected_revision: int
    expected_event_seq: int
    scope: str
    idempotency_key: str = field(repr=False)
    request_digest: str
    occurred_at: datetime

    def __post_init__(self) -> None:
        try:
            validate_resource_id(self.scope)
            validate_idempotency_key(self.idempotency_key)
        except AIError as error:
            raise ValueError("idempotency identity is invalid") from error


@dataclass(frozen=True, slots=True)
class ExecutionCancelRequestCommit:
    execution_id: str
    expected_revision: int
    expected_event_seq: int
    operation_id: str = field(repr=False)
    requested_at: datetime


@dataclass(frozen=True, slots=True)
class ExecutionStartReservation:
    execution: ExecutionRecord
    idempotency: IdempotencyRecord
    budget: RunBudget | None = None
    # Admission proof is transient; the parent execution owns the generation.
    parent_producer_generation: int | None = None

    def __post_init__(self) -> None:
        generation = self.parent_producer_generation
        if generation is not None and (
            isinstance(generation, bool) or not isinstance(generation, int) or generation < 1
            or self.execution.parent_execution_id is None
        ):
            raise ValueError("parent producer generation requires a subagent and a positive integer")


@dataclass(frozen=True, slots=True)
class ExecutionStartReservationResult:
    execution: ExecutionRecord
    idempotency: IdempotencyRecord
    created: bool
    session: SessionRecord | None = None


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    scope: str
    idempotency_key: str = field(repr=False)
    request_digest: str
    resource_kind: ResourceKind
    resource_id: str
    status: IdempotencyStatus
    result_digest: str | None
    error_code: str | None
    created_at: datetime
    updated_at: datetime
    request_identity_version: int = 1

    def __post_init__(self) -> None:
        try:
            validate_resource_id(self.scope)
            validate_idempotency_key(self.idempotency_key)
        except AIError as error:
            raise ValueError("idempotency identity is invalid") from error
        if type(self.request_identity_version) is not int or self.request_identity_version not in (1, 2):
            raise ValueError("request identity version is unsupported")
        if self.created_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise ValueError("idempotency timestamps require timezone awareness")


@dataclass(frozen=True, slots=True)
class ResultRecord:
    output: StoredPayload | None
    stop_reason: StopReason
    usage: UsageMetrics
    created_at: datetime

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None:
            raise ValueError("result requires an aware timestamp")


@dataclass(frozen=True, slots=True)
class ExecutionTerminalCommit:
    expected_revision: int
    expected_event_seq: int
    execution: ExecutionRecord
    result: ResultRecord
    terminal_event_type: ExecutionEventType
    terminal_event_payload: Mapping[str, JsonValue]
    idempotency: IdempotencyTerminalUpdate | None = None
    operation: OperationTerminalUpdate | None = None
    producer_generation: int | None = None

    def __post_init__(self) -> None:
        if self.producer_generation is not None and (
            isinstance(self.producer_generation, bool)
            or not isinstance(self.producer_generation, int)
            or self.producer_generation < 1
        ):
            raise ValueError("terminal producer generation is invalid")
        status = self.execution.status
        if status not in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            raise ValueError("terminal commit requires a terminal Execution")
        if status is ExecutionStatus.SUCCEEDED and self.result.output is None:
            raise ValueError("successful terminal result requires output")
        if status is not ExecutionStatus.SUCCEEDED and self.result.output is not None:
            raise ValueError("failed terminal result cannot contain output")
        if status is ExecutionStatus.FAILED:
            if self.terminal_event_type is not ExecutionEventType.EXECUTION_FAILED:
                raise ValueError("failed terminal commit requires failed event")
            if (
                self.terminal_event_payload.get("error_code")
                != self.execution.error_code
            ):
                raise ValueError("failed terminal error code mismatch")
            safe_details = self.terminal_event_payload.get("safe_error_details")
            if not isinstance(safe_details, Mapping) or dict(safe_details) != dict(
                self.execution.safe_error_details
            ):
                raise ValueError("failed terminal safe error details mismatch")
            diagnostics = self.execution.error_diagnostics
            if diagnostics is None:
                if "error_diagnostics" in self.terminal_event_payload:
                    raise ValueError(
                        "historical failed terminal cannot invent diagnostics"
                    )
            elif self.terminal_event_payload.get(
                "error_diagnostics"
            ) != diagnostics.to_payload():
                raise ValueError("failed terminal error diagnostics mismatch")
        elif "error_diagnostics" in self.terminal_event_payload:
            raise ValueError("non-failed terminal cannot carry error diagnostics")


@dataclass(frozen=True, slots=True)
class IdempotencyTerminalUpdate:
    scope: str
    idempotency_key: str = field(repr=False)
    expected_status: IdempotencyStatus
    next_status: IdempotencyStatus
    request_digest: str
    result_digest: str | None
    error_code: str | None

    def __post_init__(self) -> None:
        try:
            validate_resource_id(self.scope)
            validate_idempotency_key(self.idempotency_key)
        except AIError as error:
            raise ValueError("idempotency identity is invalid") from error


@dataclass(frozen=True, slots=True)
class OperationTerminalUpdate:
    operation_id: str = field(repr=False)
    expected_status: OperationStatus
    next_status: OperationStatus
    result_ref: str | None
    result_digest: str | None
    error_code: str | None


@dataclass(frozen=True, slots=True)
class ExecutionTerminalCommitResult:
    execution: ExecutionRecord
    result: ResultRecord


@dataclass(frozen=True, slots=True)
class PendingDeferredCall:
    tool_call_id: str
    tool_name: str
    arguments_payload: StoredPayload
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    capability_ref: CapabilityRef | None = None

    def __post_init__(self) -> None:
        if not self.tool_call_id or not self.tool_name:
            raise ValueError("deferred call identity is required")
        if self.capability_ref is not None and not isinstance(self.capability_ref, CapabilityRef):
            raise ValueError("deferred call capability reference is invalid")
        try:
            object.__setattr__(self, "metadata", dict(self.metadata))
        except (TypeError, ValueError) as error:
            raise ValueError("deferred call metadata is invalid") from error


@dataclass(frozen=True, slots=True)
class PendingToolContinuation:
    source_agent_run_id: str
    approvals: tuple[PendingDeferredCall, ...] = ()
    calls: tuple[PendingDeferredCall, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_agent_run_id:
            raise ValueError("deferred continuation identity is invalid")
        values = (*self.approvals, *self.calls)
        ids = tuple(item.tool_call_id for item in values)
        if not values or len(ids) != len(set(ids)):
            raise ValueError("deferred continuation calls must be unique")


@dataclass(frozen=True, slots=True)
class RepositoryInstructionBarrier:
    agent_run_id: str
    tool_call_id: str
    arguments_digest: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.agent_run_id,
                self.tool_call_id,
                self.arguments_digest,
            )
        ) or not (len(self.arguments_digest) == 64 and all(value in "0123456789abcdef" for value in self.arguments_digest)):
            raise ValueError("repository instruction barrier is invalid")


class ExecutionRecoveryState(str, Enum):
    __str__ = str.__str__
    __format__ = str.__format__
    ADMITTED = "admitted"
    ACTIVE = "active"
    WAITING = "waiting"
    HANDOFF = "handoff"
    COMPLETED = "completed"


class RecoveryHandoffPhase(str, Enum):
    __str__ = str.__str__
    __format__ = str.__format__
    NONE = "none"
    PREPARED = "prepared"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class RecoveryTerminalOutcome:
    terminal_status: ExecutionStatus
    error_code: str | None
    safe_error_details: Mapping[str, JsonValue]
    stop_reason: StopReason
    output: StoredPayload | None
    object_source_domain: PersistenceDomain | None
    usage: UsageMetrics
    terminal_event_type: ExecutionEventType
    terminal_event_payload: Mapping[str, JsonValue]
    result_created_at: datetime
    error_diagnostics: ErrorDiagnostics | None = None

    def __post_init__(self) -> None:
        if self.terminal_status is ExecutionStatus.SUCCEEDED and self.output is None:
            raise ValueError("successful recovery outcome requires output")
        if (
            self.terminal_status is not ExecutionStatus.SUCCEEDED
            and self.output is not None
        ):
            raise ValueError("failed terminal result cannot contain output")
        if self.output is None and self.object_source_domain is not None:
            raise ValueError("recovery object source requires output")
        if (
            self.output is not None
            and self.output.kind == "inline"
            and self.object_source_domain is not None
        ):
            raise ValueError("inline recovery output cannot have an object source")
        if (
            self.output is not None
            and self.output.kind == "object"
            and self.object_source_domain is None
        ):
            raise ValueError("object recovery output requires an object source")
        if self.object_source_domain not in {
            None,
            PersistenceDomain.EXECUTION,
            PersistenceDomain.CHECKPOINT,
        }:
            raise ValueError("recovery object source domain is invalid")
        if self.result_created_at.tzinfo is None:
            raise ValueError("recovery outcome requires an aware timestamp")
        if self.error_diagnostics is not None and not isinstance(
            self.error_diagnostics, ErrorDiagnostics
        ):
            raise TypeError("recovery error diagnostics are invalid")
        if (
            self.terminal_status is not ExecutionStatus.FAILED
            and self.error_diagnostics is not None
        ):
            raise ValueError("only failed recovery outcome can carry error diagnostics")


@dataclass(frozen=True, slots=True)
class RecoveryConversationIntent:
    session_id: str
    expected_cursor: ConversationCursor | None
    next_cursor: ConversationCursor


@dataclass(frozen=True, slots=True)
class RecoveryTerminalHandoff:
    outcome: RecoveryTerminalOutcome
    source_agent_run_id: str | None
    conversation: RecoveryConversationIntent | None

    def __post_init__(self) -> None:
        if (
            self.outcome.terminal_status is ExecutionStatus.SUCCEEDED
            and self.source_agent_run_id is None
        ):
            raise ValueError("successful recovery handoff requires a source attempt")
        if self.conversation is not None and self.source_agent_run_id is None:
            raise ValueError("conversation recovery intent requires a source attempt")


@dataclass(frozen=True, slots=True)
class ExecutionRecoveryDetail:
    """Recoverable execution details owned by the execution control revision."""

    agent_run_id: str | None = None
    source_agent_run_id: str | None = None
    pending_tools: PendingToolContinuation | None = None
    repository_instruction_overlay: RuntimePayloadRef | None = None
    repository_instruction_barriers: tuple[RepositoryInstructionBarrier, ...] = ()
    handoff_phase: RecoveryHandoffPhase = RecoveryHandoffPhase.NONE
    terminal_handoff: RecoveryTerminalHandoff | None = None

    def __post_init__(self) -> None:
        for identity in (self.agent_run_id, self.source_agent_run_id):
            if identity is not None and (not isinstance(identity, str) or not identity):
                raise ValueError("recovery attempt identity cannot be empty")
        if self.agent_run_id is None and (self.source_agent_run_id is not None or self.pending_tools is not None):
            raise ValueError("recovery source requires an admitted attempt")
        barriers = tuple(self.repository_instruction_barriers)
        identities = tuple((item.agent_run_id, item.tool_call_id) for item in barriers)
        if len(identities) != len(set(identities)):
            raise ValueError("repository instruction barriers must be unique")
        object.__setattr__(self, "repository_instruction_barriers", barriers)
        if self.handoff_phase is RecoveryHandoffPhase.NONE and self.terminal_handoff is not None:
            raise ValueError("unprepared recovery cannot contain a terminal handoff")
        if self.handoff_phase is RecoveryHandoffPhase.PREPARED and self.terminal_handoff is None:
            raise ValueError("prepared recovery requires a terminal handoff")
        if self.handoff_phase is not RecoveryHandoffPhase.NONE and self.pending_tools is not None:
            raise ValueError("terminal recovery cannot retain deferred continuation")


__all__ = ['ExecutionTimelineRecord', 'PendingDeferredCall', 'PendingToolContinuation', 'RepositoryInstructionBarrier', 'ExecutionRecoveryState', 'RecoveryHandoffPhase', 'RecoveryTerminalOutcome', 'RecoveryConversationIntent', 'RecoveryTerminalHandoff', 'ExecutionRecoveryDetail', 'RuntimePayloadRef', 'StoredUserInput', 'ExecutionRecord', 'ExecutionStartClaim', 'ExecutionStartUnknownCommit', 'ExecutionCancelRequestCommit', 'ExecutionStartReservation', 'ExecutionStartReservationResult', 'IdempotencyRecord', 'ResultRecord', 'ExecutionTerminalCommit', 'IdempotencyTerminalUpdate', 'OperationTerminalUpdate', 'ExecutionTerminalCommitResult']
