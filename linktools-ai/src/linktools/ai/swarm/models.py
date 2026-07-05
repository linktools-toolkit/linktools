#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Swarm domain models: SwarmRun/SwarmTask state, AgentRef, TaskInput, TokenUsage,
and the SwarmStatus/SwarmTaskStatus enums + transition table. Mirrors the
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
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SwarmTaskStatus(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


ALLOWED_SWARM_TRANSITIONS: "Mapping[SwarmStatus, frozenset[SwarmStatus]]" = {
    SwarmStatus.PENDING: frozenset({SwarmStatus.RUNNING}),
    SwarmStatus.RUNNING: frozenset({
        SwarmStatus.PAUSED, SwarmStatus.SUCCEEDED, SwarmStatus.FAILED,
        SwarmStatus.CANCELLED,
    }),
    SwarmStatus.PAUSED: frozenset({SwarmStatus.RUNNING, SwarmStatus.CANCELLED}),
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
        return cls(
            input_tokens=int(m.get("input_tokens", 0) or 0),
            output_tokens=int(m.get("output_tokens", 0) or 0),
        )


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
class SwarmTask:
    id: str
    swarm_run_id: str
    parent_task_id: "str | None"
    assigned_agent_id: "str | None"
    description: str
    status: SwarmTaskStatus
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
