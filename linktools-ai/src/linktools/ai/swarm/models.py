#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Swarm domain models: SwarmRun/SwarmStep state, AgentRef, TaskInput, TokenUsage,
and the SwarmStatus/SwarmStepStatus enums + transition table. Mirrors the
frozen-dataclass + str-Enum conventions of run/models.py and session/models.py."""

from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ..run.models import RunErrorInfo, RunResult


class SwarmStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    RECOVERABLE = "recoverable"
    # CANCELLING distinguishes "cancel requested" from "actually cancelled"
    # (mirrors RunStatus.CANCELLING):
    # SwarmEngine.cancel() flips to CANCELLING while an in-flight swarm
    # coroutine is still unwinding; the CancelledError handler in
    # SwarmEngine.run() transitions CANCELLING -> CANCELLED once actually
    # stopped.
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SwarmStepStatus(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AttemptStatus(str, Enum):
    """Lifecycle of a single task execution attempt (SwarmStepAttempt.status).

    Each (re)try of a SwarmStep records one SwarmStepAttempt so
    retries, agent migrations, and failure recovery are fully auditable. A task
    that succeeds on the second try leaves attempt #1 = FAILED and #2 = SUCCEEDED.
    """

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


ALLOWED_SWARM_TRANSITIONS: "Mapping[SwarmStatus, frozenset[SwarmStatus]]" = {
    SwarmStatus.PENDING: frozenset({SwarmStatus.RUNNING}),
    SwarmStatus.RUNNING: frozenset(
        {
            SwarmStatus.PAUSED,
            SwarmStatus.SUCCEEDED,
            SwarmStatus.FAILED,
            SwarmStatus.CANCELLING,
            SwarmStatus.CANCELLED,
        }
    ),
    SwarmStatus.PAUSED: frozenset(
        {
            SwarmStatus.RUNNING,
            SwarmStatus.CANCELLING,
            SwarmStatus.CANCELLED,
        }
    ),
    SwarmStatus.RECOVERABLE: frozenset({SwarmStatus.RUNNING}),
    SwarmStatus.CANCELLING: frozenset({SwarmStatus.CANCELLED, SwarmStatus.FAILED}),
    SwarmStatus.SUCCEEDED: frozenset(),
    SwarmStatus.FAILED: frozenset(),
    SwarmStatus.CANCELLED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class AgentRef:
    agent_id: str
    role: "str | None" = None


@dataclass(frozen=True, slots=True)
class TaskInput:
    prompt: str
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: "Decimal" = field(default_factory=lambda: Decimal("0"))

    def add(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_cost=self.total_cost + other.total_cost,
        )

    @classmethod
    def from_mapping(cls, m: "Mapping[str, Any]") -> "TokenUsage":
        """Strictly parse token usage from a mapping. Non-int / negative values
        are rejected rather than silently coerced via int()."""
        raw_input = m.get("input_tokens", 0)
        raw_output = m.get("output_tokens", 0)
        if (
            not isinstance(raw_input, int)
            or isinstance(raw_input, bool)
            or raw_input < 0
        ):
            raise ValueError(
                f"input_tokens must be a non-negative int, got {raw_input!r}"
            )
        if (
            not isinstance(raw_output, int)
            or isinstance(raw_output, bool)
            or raw_output < 0
        ):
            raise ValueError(
                f"output_tokens must be a non-negative int, got {raw_output!r}"
            )
        return cls(input_tokens=raw_input, output_tokens=raw_output)


@dataclass(frozen=True, slots=True)
class SwarmRun:
    id: str
    run_id: str
    round: int
    status: SwarmStatus
    version: int
    token_usage: TokenUsage
    cost: "Decimal"
    created_at: datetime
    updated_at: datetime
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SwarmCheckpoint:
    """Durable strategy input for resume; it never replays completed tasks."""

    completed_task_ids: "tuple[str, ...]"
    failed_task_ids: "tuple[str, ...]"
    pending_task_ids: "tuple[str, ...]"
    active_task_ids: "tuple[str, ...]"
    task_outputs: "Mapping[str, RunResult]" = field(default_factory=dict)
    strategy_state: "Mapping[str, Any]" = field(default_factory=dict)
    aggregate_state: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SwarmStep:
    id: str
    swarm_run_id: str
    parent_task_id: "str | None"
    assigned_agent_id: "str | None"
    description: str
    status: SwarmStepStatus
    dependencies: "tuple[str, ...]"
    input: TaskInput
    result: "RunResult | None"
    error: "RunErrorInfo | None"
    attempts: int
    version: int
    claimed_at: "datetime | None"
    lease_expires_at: "datetime | None"
    created_at: datetime
    updated_at: datetime
    # The id of the child RunRecord this task's execution creates (set in
    # strategy._run_task right after claim_task succeeds). The child identity:
    # task.id IS NOT its child RunRecord.id; each (re)execution mints a fresh
    # run_id and stores it here. None until claimed, or after a reclaim reset.
    active_run_id: "str | None" = None


@dataclass(frozen=True, slots=True)
class SwarmStepAttempt:
    """One execution attempt of a SwarmStep.

    A single SwarmStep may produce several SwarmStepAttempts over its life:
    retries inside one ``_run_task`` call each record their own attempt, as does
    a re-invocation of ``_run_task`` after a prior FAILED. The ``run_id`` is the
    child RunRecord id for that execution (NOT the task id), and the
    ``attempt`` field is 1-based and monotonically increments per task. The
    status transitions RUNNING -> SUCCEEDED | FAILED so the audit trail records
    both the failures and the eventual success (or final failure).
    """

    id: str
    task_id: str
    run_id: str
    agent_id: str
    attempt: int
    status: AttemptStatus
    started_at: datetime
    finished_at: "datetime | None"
    error: "RunErrorInfo | None"
