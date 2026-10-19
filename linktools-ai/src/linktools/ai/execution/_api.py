#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution control and result values."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, cast
from ..core import (ExecutionStatus, ExecutionLineageKind, JsonValue, UsageMetrics,
    Principal, validate_idempotency_key)
from ..errors import AIError, ErrorCode, ErrorDiagnostics
from ._records import ResultRecord

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
    started_at: datetime | None
    result: "ResultRecord | None"


@dataclass(frozen=True, slots=True)
class ExecutionView:
    """Execution state, with the original start and recorded result timestamps.

    ``terminal_at`` is the durable result's creation time, not the wall-clock
    instant when its terminal storage transaction became visible.
    """

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
    started_at: datetime | None = None
    terminal_at: datetime | None = None


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
        started_at=value.started_at,
        terminal_at=None if value.result is None else value.result.created_at,
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


__all__ = ['CancelExecutionRequest', 'CancelExecutionResult', 'ExecutionHandle', 'ExecutionView', 'project_execution_view', 'ExecutionResult']
