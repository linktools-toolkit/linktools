"""Canonical semantic trace values."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from ..storage.json import JsonValue
from .run import RunUsage


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
class ToolCallTrace:
    call_id: str
    interaction_sequence: int
    tool_name: str
    arguments: JsonValue
    result: JsonValue | None
    status: str
    error: TraceError | None
    started_at: datetime | None
    completed_at: datetime | None


__all__ = ["InteractionStatus", "ModelInteractionTrace", "ModelRequestTrace", "ModelResponseTrace", "ToolCallTrace", "TraceError"]
