#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime domain state contracts."""

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from .._domain import RuntimeDomain

if TYPE_CHECKING:
    from .._persistence import (
        ApprovalRepository,
        ArtifactRepository,
        EvaluationRepository,
        EventRepository,
        ExecutionRepository,
        ExternalCallRepository,
        IdempotencyRepository,
        MemoryRepository,
        OperationLedgerRepository,
        RecoveryCheckpointRepository,
        SessionRepository,
        TaskRepository,
    )
    from .._tool import ToolStateRepository


class RuntimeRetentionMode(StrEnum):
    DURABLE = "durable"
    VOLATILE = "volatile"
    TRANSIENT = "transient"


@dataclass(frozen=True, slots=True)
class ConversationState:
    sessions: "SessionRepository"
    operations: "OperationLedgerRepository"


@dataclass(frozen=True, slots=True)
class ExecutionState:
    executions: "ExecutionRepository"
    events: "EventRepository"
    idempotency: "IdempotencyRepository"
    operations: "OperationLedgerRepository"


@dataclass(frozen=True, slots=True)
class MemoryState:
    records: "MemoryRepository"
    operations: "OperationLedgerRepository"


@dataclass(frozen=True, slots=True)
class ArtifactState:
    records: "ArtifactRepository"
    operations: "OperationLedgerRepository"


@dataclass(frozen=True, slots=True)
class TaskState:
    tasks: "TaskRepository"
    operations: "OperationLedgerRepository"


@dataclass(frozen=True, slots=True)
class EvaluationState:
    records: "EvaluationRepository"
    idempotency: "IdempotencyRepository"
    operations: "OperationLedgerRepository"


@dataclass(frozen=True, slots=True)
class RecoveryState:
    approvals: "ApprovalRepository"
    external_calls: "ExternalCallRepository"
    checkpoints: "RecoveryCheckpointRepository"
    operations: "OperationLedgerRepository"
    tools: "ToolStateRepository"


__all__ = [
    "ArtifactState",
    "ConversationState",
    "EvaluationState",
    "ExecutionState",
    "MemoryState",
    "RecoveryState",
    "RuntimeDomain",
    "RuntimeRetentionMode",
    "TaskState",
]
