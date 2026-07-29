"""Execution aggregates and canonical JSON-shaped persistence values."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Generic, Literal, TypeVar

from ..storage.json import JsonScalar, JsonValue
from .run import (
    RunApproval,
    RunDefinition,
    RunError,
    RunErrorInfo,
    RunKind,
    RunRecord,
    RunResult,
    RunStatus,
    RunUsage,
    RunnableType,
)

@dataclass(frozen=True, slots=True)
class SessionRecord:
    id: str
    user_id: str | None
    tenant_id: str | None
    next_turn_sequence: int
    latest_completed_run_id: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class SessionTurn:
    session_id: str
    sequence: int
    run_id: str
    user_prompt: JsonValue
    assistant_summary: JsonValue | None
    status: RunStatus
    created_at: datetime
    completed_at: datetime | None


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    schema: Literal["run-snapshot.v1"]
    run_id: str
    revision: int
    resume_messages: tuple[JsonValue, ...]
    final_output: JsonValue | None
    status: RunStatus
    usage: RunUsage
    trace_end_sequence: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class NewRunTraceStep:
    kind: Literal["model_interaction", "tool_result"]
    payload: JsonValue
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RunTraceStep:
    run_id: str
    sequence: int
    kind: Literal["model_interaction", "tool_result"]
    payload: JsonValue
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RunEvent:
    run_id: str
    sequence: int
    type: str
    payload: JsonValue
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RunEvaluation:
    evaluation_id: str
    run_id: str
    evaluator: str
    score: float | None
    result: JsonValue
    created_at: datetime


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Page(Generic[T]):
    items: tuple[T, ...]
    has_more: bool
    next_cursor: int | None = None


__all__ = [
    "JsonScalar",
    "JsonValue",
    "Page",
    "RunApproval",
    "RunDefinition",
    "RunError",
    "RunErrorInfo",
    "RunEvaluation",
    "RunEvent",
    "RunKind",
    "RunRecord",
    "RunResult",
    "RunSnapshot",
    "RunStatus",
    "RunTraceStep",
    "RunUsage",
    "RunnableType",
    "SessionRecord",
    "SessionTurn",
    "NewRunTraceStep",
]
