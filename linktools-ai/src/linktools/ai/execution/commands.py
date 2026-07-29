"""Validated commands accepted by the execution lifecycle port."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .run import RunApproval, RunDefinition, RunKind


@dataclass(frozen=True, slots=True)
class CreateSession:
    session_id: str
    user_id: str | None
    tenant_id: str | None


@dataclass(frozen=True, slots=True)
class StartRun:
    run_id: str
    session_id: str
    kind: RunKind
    definition: RunDefinition
    user_prompt: object
    root_run_id: str | None = None
    parent_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class ClaimRun:
    run_id: str
    owner: str
    now: datetime
    duration: timedelta


@dataclass(frozen=True, slots=True)
class HeartbeatRun:
    run_id: str
    owner: str
    fence: int
    now: datetime
    duration: timedelta


@dataclass(frozen=True, slots=True)
class PauseRun:
    run_id: str
    owner: str
    fence: int
    snapshot: object
    pending_approval: RunApproval


@dataclass(frozen=True, slots=True)
class DecideRunApproval:
    run_id: str
    approval_id: str
    decision: str
    decided_by: str


@dataclass(frozen=True, slots=True)
class ResumeRun:
    run_id: str


@dataclass(frozen=True, slots=True)
class RequestRunCancel:
    run_id: str
    owner: str
    fence: int
    requested_at: datetime


@dataclass(frozen=True, slots=True)
class CompleteRun:
    run_id: str
    owner: str
    fence: int
    snapshot: object


@dataclass(frozen=True, slots=True)
class FailRun:
    run_id: str
    owner: str
    fence: int
    snapshot: object


@dataclass(frozen=True, slots=True)
class AcknowledgeRunCancel:
    run_id: str
    owner: str
    fence: int
    snapshot: object


__all__ = [
    "AcknowledgeRunCancel",
    "ClaimRun",
    "CompleteRun",
    "CreateSession",
    "DecideRunApproval",
    "FailRun",
    "HeartbeatRun",
    "PauseRun",
    "RequestRunCancel",
    "ResumeRun",
    "StartRun",
]
