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
    # CANCELLING distinguishes "cancel requested" from "actually cancelled"
    # (review doc §6.1). Runtime.cancel flips a run to CANCELLING while the
    # in-flight task is still draining; the runner's CancelledError handler
    # then transitions CANCELLING -> CANCELLED once the task has actually
    # stopped. Going through CANCELLING first avoids falsely advertising a run
    # as cancelled the instant cancel is requested -- the underlying asyncio
    # Task may still be mid-await.
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunnableType(str, Enum):
    AGENT = "agent"
    SWARM = "swarm"


# Review doc §6.2 transition table. CANCELLING is the canonical route to
# CANCELLED for any in-flight run: RUNNING/WAITING_APPROVAL/PAUSED -> CANCELLING
# -> CANCELLED (or -> FAILED if cancellation itself times out / errors).
#
# The direct {RUNNING, WAITING_APPROVAL, PAUSED} -> CANCELLED edges are RETAINED
# as a fallback for the no-task path: when Runtime.cancel is called on a run
# with no live asyncio.Task registered (a stale record from a crashed worker, a
# test seed, or any path where the runner is not driving execute()), there is
# nothing to actually stop -- going straight to CANCELLED is correct. Removing
# these edges would break the seeded-cancel tests in test_runtime_cancel.py
# (those runs never have an in-flight task). Document and keep them.
ALLOWED_RUN_TRANSITIONS: "Mapping[RunStatus, frozenset]" = {
    RunStatus.PENDING: frozenset({RunStatus.RUNNING}),
    RunStatus.RUNNING: frozenset({
        RunStatus.WAITING_APPROVAL, RunStatus.PAUSED, RunStatus.SUCCEEDED,
        RunStatus.FAILED, RunStatus.CANCELLING, RunStatus.CANCELLED,
    }),
    RunStatus.WAITING_APPROVAL: frozenset({
        RunStatus.RUNNING, RunStatus.CANCELLING, RunStatus.CANCELLED,
    }),
    RunStatus.PAUSED: frozenset({
        RunStatus.RUNNING, RunStatus.CANCELLING, RunStatus.CANCELLED,
    }),
    RunStatus.CANCELLING: frozenset({RunStatus.CANCELLED, RunStatus.FAILED}),
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
