"""Canonical semantic trace values, and the persisted trace-step / event
append-log records that carry them."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal

from ..json import JsonValue
from .domain import RunUsage


class InteractionStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class TraceError:
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class ModelRequestTrace:
    messages: tuple[JsonValue, ...]
    settings: JsonValue
    tools: tuple[JsonValue, ...]


@dataclass(frozen=True, slots=True)
class ModelResponseTrace:
    parts: tuple[JsonValue, ...]
    finish_reason: str | None
    provider_response_id: str | None
    usage: RunUsage


@dataclass(frozen=True, slots=True)
class ModelInteractionTrace:
    sequence: int
    model_name: str
    request: ModelRequestTrace
    response: ModelResponseTrace | None
    status: InteractionStatus
    error: TraceError | None
    started_at: datetime
    completed_at: datetime | None


@dataclass(frozen=True, slots=True)
class ToolResultTrace:
    tool_call_id: str
    tool_name: str
    operation_id: str
    status: Literal[
        "completed",
        "failed",
        "denied",
        "result_denied",
        "indeterminate",
    ]
    result: JsonValue | None
    error: JsonValue | None
    replayed: bool
    started_at: datetime | None
    completed_at: datetime


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


__all__ = [
    "InteractionStatus",
    "ModelInteractionTrace",
    "ModelRequestTrace",
    "ModelResponseTrace",
    "NewRunTraceStep",
    "RunEvent",
    "RunTraceStep",
    "ToolResultTrace",
    "TraceError",
]
