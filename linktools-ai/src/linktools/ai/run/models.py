#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run domain models: RunStatus/RunnableType/RunInput/RunResult/RunErrorInfo/RunRecord,
and the allowed-transition table RunStore.transition() validates against."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunnableType(str, Enum):
    AGENT = "agent"
    SWARM = "swarm"


ALLOWED_RUN_TRANSITIONS: "Mapping[RunStatus, frozenset]" = {
    RunStatus.PENDING: frozenset({RunStatus.RUNNING}),
    RunStatus.RUNNING: frozenset({
        RunStatus.WAITING_APPROVAL, RunStatus.PAUSED, RunStatus.SUCCEEDED,
        RunStatus.FAILED, RunStatus.CANCELLED,
    }),
    RunStatus.WAITING_APPROVAL: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED}),
    RunStatus.PAUSED: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED}),
    RunStatus.SUCCEEDED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class RunInput:
    prompt: str
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunResult:
    output: Any
    token_usage: "Mapping[str, Any]" = field(default_factory=dict)
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunErrorInfo:
    error_type: str
    message: str
    detail: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunRecord:
    id: str
    root_run_id: str
    parent_run_id: "str | None"
    session_id: str
    runnable_id: str
    runnable_type: RunnableType
    status: RunStatus
    input: RunInput
    result: "RunResult | None"
    error: "RunErrorInfo | None"
    version: int
    created_at: datetime
    started_at: "datetime | None"
    finished_at: "datetime | None"
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunCheckpoint:
    id: str
    run_id: str
    sequence: int
    format: str
    schema_version: int
    payload: bytes
    created_at: datetime
    metadata: "Mapping[str, Any]" = field(default_factory=dict)
