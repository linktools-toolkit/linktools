#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned step, event, and checkpoint contracts."""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Protocol

from pydantic_ai.messages import ModelMessage, ModelRequest

EventKind = Literal[
    "run_started",
    "run_completed",
    "run_interrupted",
    "run_failed",
    "model_request_started",
    "model_request_completed",
    "model_request_failed",
    "model_request_cancelled",
    "tool_call_started",
    "tool_call_completed",
    "tool_call_failed",
]
CheckpointState = Literal["complete", "interrupted"]


@dataclass(slots=True)
class AgentRunRecord:
    agent_run_id: str
    agent_conversation_id: str | None = None
    parent_agent_run_id: str | None = None
    agent_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class StepEvent:
    agent_run_id: str
    kind: EventKind
    step_index: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    agent_conversation_id: str | None = None
    parent_agent_run_id: str | None = None
    agent_id: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    error: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    idempotency_key: str | None = None
    event_index: int = 0


@dataclass(slots=True)
class AgentRunCheckpoint:
    agent_run_id: str
    step_index: int
    messages: list[ModelMessage]
    agent_conversation_id: str | None = None
    parent_agent_run_id: str | None = None
    agent_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    state: CheckpointState = "complete"
    idempotency_key: str | None = None
    context_messages: list[ModelMessage] | None = None
    transcript_message_count_before: int | None = None
    pending_request_index: int | None = None

    def __post_init__(self) -> None:
        if (
            self.transcript_message_count_before is not None
            and (
                isinstance(self.transcript_message_count_before, bool)
                or not isinstance(self.transcript_message_count_before, int)
                or self.transcript_message_count_before < 0
                or self.transcript_message_count_before > len(self.messages)
            )
        ):
            raise ValueError("checkpoint transcript boundary is invalid")
        if self.pending_request_index is None:
            return
        if (
            isinstance(self.pending_request_index, bool)
            or not isinstance(self.pending_request_index, int)
            or self.pending_request_index < 0
        ):
            raise ValueError("checkpoint pending request index is invalid")
        context = self.messages if self.context_messages is None else self.context_messages
        if (
            self.pending_request_index >= len(context)
            or not isinstance(context[self.pending_request_index], ModelRequest)
        ):
            raise ValueError("checkpoint pending request must identify a model request")


class AgentRunStore(Protocol):
    async def initialize(self) -> None: ...

    async def close(self) -> None: ...

    async def register_agent_run(
        self, record: AgentRunRecord, *, execution_id: str | None = None
    ) -> None: ...

    async def get_agent_run(self, *, agent_run_id: str) -> AgentRunRecord | None: ...

    async def list_agent_runs(
        self,
        *,
        parent_agent_run_id: str | None = None,
        agent_conversation_id: str | None = None,
    ) -> list[AgentRunRecord]: ...

    async def append_event(
        self, event: StepEvent, *, execution_id: str | None = None
    ) -> None: ...

    async def list_events(self, *, agent_run_id: str) -> list[StepEvent]: ...

    async def iter_messages(self, *, agent_run_id: str) -> AsyncIterator[object]: ...

    async def list_checkpoints(self, *, agent_run_id: str) -> list[AgentRunCheckpoint]: ...

    async def save_checkpoint(
        self, checkpoint: AgentRunCheckpoint, *, execution_id: str | None = None
    ) -> None: ...

    async def latest_checkpoint(
        self, *, agent_run_id: str, include_interrupted: bool = False
    ) -> AgentRunCheckpoint | None: ...

    async def list_model_interactions(
        self,
        *,
        agent_run_id: str,
        after_request_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[object]: ...

    async def model_interaction_count(self, *, agent_run_id: str) -> int: ...

    async def resolve_model_interaction(
        self,
        interaction: object,
    ) -> object: ...

    async def resolve_model_interactions(
        self,
        interactions: Sequence[object],
    ) -> list[object]: ...

    async def release_agent_run(
        self, agent_run_id: str, *, execution_id: str | None = None
    ) -> None: ...


__all__ = [
    "AgentRunCheckpoint",
    "EventKind",
    "AgentRunRecord",
    "CheckpointState",
    "StepEvent",
    "AgentRunStore",
]
