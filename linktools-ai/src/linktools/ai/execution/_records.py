#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution-owned durable records and state transition contracts."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from ..agent import AgentBindingContract
from ..asset import AssetVersionRef
from ..core import (PersistenceDomain, RunBudget, ExecutionEventType, ExecutionLineageKind,
    ExecutionMode, ExecutionStatus, IdempotencyStatus, JsonValue, OperationStatus,
    ResourceKind, StopReason, ThinkingValue, UsageMetrics, canonical_sha256,
    normalize_correlation, normalize_execution_mode, normalize_thinking)
from ..errors import ErrorDiagnostics
from ..storage import StoredPayload
from ..model import StoredUserInput
from ..task import TaskBindingContract


@dataclass(frozen=True, slots=True)
class RuntimePayloadRef:
    payload: StoredPayload
    source_domain: PersistenceDomain | None


@dataclass(frozen=True, slots=True)
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
    restart_checkpoints: bool = False
    durable_trace: bool = False
    # Whether this execution retains conversation and fork context.
    conversation: bool = True

    def __post_init__(self) -> None:
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
        agent_binding = isinstance(self.binding, AgentBindingContract)
        task_binding = isinstance(self.binding, TaskBindingContract)
        if agent_binding == task_binding:
            raise TypeError("execution binding contract is invalid")
        if agent_binding:
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
            if self.mode is not None or self.planning is not None or self.thinking is not None:
                raise ValueError("task execution cannot carry agent execution policy")
            if (
                self.session_id is not None
                or self.memory_scope is not None
                or self.input_context is not None
                or self.context_imported
                or self.tool_response_ref is not None
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
class ExecutionStartClaim:
    execution_id: str
    expected_revision: int
    expected_event_seq: int
    scope: str
    idempotency_key_digest: str
    request_digest: str
    started_at: datetime


@dataclass(frozen=True, slots=True)
class ExecutionStartUnknownCommit:
    execution_id: str
    expected_revision: int
    expected_event_seq: int
    scope: str
    idempotency_key_digest: str
    request_digest: str
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class ExecutionCancelRequestCommit:
    execution_id: str
    expected_revision: int
    expected_event_seq: int
    operation_id: str
    requested_at: datetime


@dataclass(frozen=True, slots=True)
class ExecutionStartReservation:
    execution: ExecutionRecord
    idempotency: IdempotencyRecord
    budget: RunBudget | None = None


@dataclass(frozen=True, slots=True)
class ExecutionStartReservationResult:
    execution: ExecutionRecord
    idempotency: IdempotencyRecord
    created: bool


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    scope: str
    idempotency_key_digest: str
    request_digest: str
    resource_kind: ResourceKind
    resource_id: str
    status: IdempotencyStatus
    result_digest: str | None
    error_code: str | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
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
    idempotency_key_digest: str
    expected_status: IdempotencyStatus
    next_status: IdempotencyStatus
    request_digest: str
    result_digest: str | None
    error_code: str | None


@dataclass(frozen=True, slots=True)
class OperationTerminalUpdate:
    operation_id: str
    expected_status: OperationStatus
    next_status: OperationStatus
    result_ref: str | None
    result_digest: str | None
    error_code: str | None


@dataclass(frozen=True, slots=True)
class ExecutionTerminalCommitResult:
    execution: ExecutionRecord
    result: ResultRecord


__all__ = ['RuntimePayloadRef', 'StoredUserInput', 'ExecutionRecord', 'ExecutionStartClaim', 'ExecutionStartUnknownCommit', 'ExecutionCancelRequestCommit', 'ExecutionStartReservation', 'ExecutionStartReservationResult', 'IdempotencyRecord', 'ResultRecord', 'ExecutionTerminalCommit', 'IdempotencyTerminalUpdate', 'OperationTerminalUpdate', 'ExecutionTerminalCommitResult']
