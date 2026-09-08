#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned step, event, and snapshot contracts."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Protocol

from pydantic_ai.messages import ModelMessage

EventKind = Literal[
    "run_started",
    "run_completed",
    "run_interrupted",
    "run_failed",
    "model_request_started",
    "model_request_completed",
    "model_request_failed",
    "tool_call_started",
    "tool_call_completed",
    "tool_call_failed",
]
SnapshotState = Literal["complete", "interrupted"]


@dataclass(slots=True)
class RunRecord:
    run_id: str
    conversation_id: str | None = None
    parent_run_id: str | None = None
    agent_name: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    registration_id: str | None = None


@dataclass(slots=True)
class StepEvent:
    run_id: str
    kind: EventKind
    step_index: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    conversation_id: str | None = None
    parent_run_id: str | None = None
    agent_name: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    error: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    idempotency_key: str | None = None
    event_index: int = 0


@dataclass(slots=True)
class ContinuableSnapshot:
    run_id: str
    step_index: int
    messages: list[ModelMessage]
    conversation_id: str | None = None
    parent_run_id: str | None = None
    agent_name: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    state: SnapshotState = "complete"
    idempotency_key: str | None = None
    context_messages: list[ModelMessage] | None = None


class StepStore(Protocol):
    async def initialize(self) -> None: ...

    async def close(self) -> None: ...

    async def register_run(
        self, record: RunRecord, *, execution_id: str | None = None
    ) -> None: ...

    async def get_run(self, *, run_id: str) -> RunRecord | None: ...

    async def list_runs(
        self,
        *,
        parent_run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[RunRecord]: ...

    async def append_event(
        self, event: StepEvent, *, execution_id: str | None = None
    ) -> None: ...

    async def list_events(self, *, run_id: str) -> list[StepEvent]: ...

    async def iter_messages(self, *, run_id: str) -> AsyncIterator[object]: ...

    async def list_snapshots(self, *, run_id: str) -> list[ContinuableSnapshot]: ...

    async def save_snapshot(
        self, snapshot: ContinuableSnapshot, *, execution_id: str | None = None
    ) -> None: ...

    async def latest_snapshot(
        self, *, run_id: str, include_interrupted: bool = False
    ) -> ContinuableSnapshot | None: ...

    async def release_run(
        self, run_id: str, *, execution_id: str | None = None
    ) -> None: ...


__all__ = [
    "ContinuableSnapshot",
    "EventKind",
    "RunRecord",
    "SnapshotState",
    "StepEvent",
    "StepStore",
]
