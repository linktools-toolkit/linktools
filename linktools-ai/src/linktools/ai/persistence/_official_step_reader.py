#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read existing Runtime views directly from the official step-persistence owner."""

from collections.abc import AsyncIterator, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace, field

from pydantic_ai.messages import ModelMessage
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord, StepStore

from ..model.messages import native_checkpoint_messages
from ..errors import AIError, ErrorCode
from ._native_checkpoint import NativeCheckpoint
from ._step_contracts import AgentRunCheckpoint, AgentRunRecord, StepEvent, StepEventType
from ..steps import NativeLifecycleEvent
from .contracts import LoadedContextMessage, LoadedModelContext


_EVENT_TYPES: dict[str, StepEventType] = {
    "run_started": "AGENT_RUN_STARTED",
    "run_completed": "AGENT_RUN_SUCCEEDED",
    "run_failed": "AGENT_RUN_FAILED",
    "model_request_started": "MODEL_REQUEST_STARTED",
    "model_request_completed": "MODEL_REQUEST_SUCCEEDED",
    "model_request_failed": "MODEL_REQUEST_FAILED",
    "tool_call_started": "TOOL_CALL_STARTED",
    "tool_call_completed": "TOOL_CALL_SUCCEEDED",
    "tool_call_failed": "TOOL_CALL_FAILED",
}


@dataclass
class _ConfirmedStepRun:
    record: RunRecord
    latest: ContinuableSnapshot | None = None
    complete: ContinuableSnapshot | None = None
    snapshot_keys: set[str] = field(default_factory=set)


class OfficialStepReader:
    """Read-only DTO projection; no Runtime checkpoint bytes are stored here."""

    def __init__(self, store: StepStore | None = None) -> None:
        self._step_store = store
        self._confirmed: dict[str, _ConfirmedStepRun] = {}

    def bind_step_store(self, store: StepStore) -> None:
        if self._step_store is not None and self._step_store is not store:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._step_store = store

    @property
    def step_store(self) -> StepStore:
        if self._step_store is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        return self._step_store

    def confirm_run(self, record: RunRecord) -> None:
        """Retain only a run registration already acknowledged by its official owner."""
        previous = self._confirmed.get(record.run_id)
        if previous is not None and previous.record != record:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if previous is None:
            self._confirmed[record.run_id] = _ConfirmedStepRun(deepcopy(record))

    def confirm_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        confirmed = self._confirmed.get(snapshot.run_id)
        if confirmed is None:
            return
        key = snapshot.idempotency_key
        if key is None:
            self.invalidate_run(snapshot.run_id)
            return
        if key in confirmed.snapshot_keys:
            return
        confirmed.snapshot_keys.add(key)
        confirmed.latest = self.restore_snapshot(deepcopy(snapshot))
        if snapshot.state == "complete":
            confirmed.complete = confirmed.latest

    def invalidate_run(self, run_id: str) -> None:
        self._confirmed.pop(run_id, None)

    def clear_confirmed_runs(self) -> None:
        self._confirmed.clear()

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        confirmed = self._confirmed.get(run_id)
        if confirmed is not None:
            return deepcopy(confirmed.record)
        return await self.step_store.get_run(run_id=run_id)

    async def latest_snapshot(
        self, *, run_id: str, include_interrupted: bool = False,
    ) -> ContinuableSnapshot | None:
        confirmed = self._confirmed.get(run_id)
        if confirmed is not None:
            return deepcopy(confirmed.latest if include_interrupted else confirmed.complete)
        snapshot = await self.step_store.latest_snapshot(run_id=run_id, include_interrupted=include_interrupted)
        return None if snapshot is None else self.restore_snapshot(snapshot)

    @classmethod
    def restore_snapshot(cls, snapshot: ContinuableSnapshot) -> ContinuableSnapshot:
        restored = NativeCheckpoint.decode(snapshot)
        return replace(restored, messages=native_checkpoint_messages(restored.messages, restore=True))

    async def get_agent_run(self, *, agent_run_id: str) -> AgentRunRecord | None:
        record = await self.get_run(run_id=agent_run_id)
        return None if record is None else self.project_run(record)

    async def list_agent_runs(
        self, *, parent_agent_run_id: str | None = None,
        agent_conversation_id: str | None = None,
    ) -> list[AgentRunRecord]:
        return [self.project_run(record) for record in await self.step_store.list_runs(
            parent_run_id=parent_agent_run_id, conversation_id=agent_conversation_id,
        )]

    @classmethod
    def project_run(cls, record: RunRecord) -> AgentRunRecord:
        return AgentRunRecord(
            agent_run_id=record.run_id, agent_conversation_id=record.conversation_id,
            parent_agent_run_id=record.parent_run_id, agent_id=record.agent_name,
            metadata={key: value for key, value in record.metadata.items()
                      if key not in {"linktools.transcript_start", "linktools.transcript_run_id"}},
            started_at=record.started_at,
        )

    async def latest_checkpoint(
        self, *, agent_run_id: str, include_interrupted: bool = False, include_transcript: bool = True,
    ) -> AgentRunCheckpoint | None:
        snapshot = await self.latest_snapshot(
            run_id=agent_run_id, include_interrupted=include_interrupted,
        )
        if snapshot is None:
            return None
        run = await self.get_run(run_id=agent_run_id)
        if run is None or snapshot.run_id != run.run_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return self.project_checkpoint(run, snapshot, include_transcript=include_transcript)

    @classmethod
    def project_checkpoint(
        cls, run: RunRecord, snapshot: ContinuableSnapshot, *, include_transcript: bool = True,
    ) -> AgentRunCheckpoint:
        messages = []
        if include_transcript:
            try:
                raw_start = run.metadata["linktools.transcript_start"]
                start = int(raw_start)
                if str(start) != raw_start or start < 0:
                    raise ValueError("invalid snapshot transcript boundary")
            except (KeyError, TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            # Sticky history can trim SDK history. The original run's first
            # occurrence proves the boundary; absence does not invalidate context.
            transcript_run_id = run.metadata.get("linktools.transcript_run_id", run.run_id)
            first = next((index for index, message in enumerate(snapshot.messages)
                          if message.run_id == transcript_run_id), None)
            if first is None:
                if start != len(snapshot.messages):
                    raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
            else:
                # Native preparation can merge requests or refresh system parts.
                # The original run identity, not a pre-SDK offset, owns the slice.
                start = first
            messages = list(snapshot.messages[start:])
        return AgentRunCheckpoint(
            agent_run_id=snapshot.run_id, step_index=snapshot.step_index,
            messages=messages, context_messages=list(snapshot.messages),
            agent_conversation_id=snapshot.conversation_id,
            parent_agent_run_id=snapshot.parent_run_id, agent_id=snapshot.agent_name,
            timestamp=snapshot.timestamp, state=snapshot.state,
            idempotency_key=snapshot.idempotency_key,
            output_retries_used=snapshot.output_retries_used if isinstance(snapshot, NativeCheckpoint) else None,
        )

    async def list_events(self, *, agent_run_id: str) -> list[StepEvent]:
        return [StepEvent(
            agent_run_id=event.run_id, event_type=event.event_type if isinstance(event, NativeLifecycleEvent) else _EVENT_TYPES[event.kind],
            step_index=event.step_index, timestamp=event.timestamp,
            agent_conversation_id=event.conversation_id, parent_agent_run_id=event.parent_run_id,
            agent_id=event.agent_name, tool_call_id=event.tool_call_id, tool_name=event.tool_name,
            error=event.error, metadata=dict(event.metadata), idempotency_key=event.idempotency_key,
            event_index=index,
        ) for index, event in enumerate(await self.step_store.list_events(run_id=agent_run_id), 1)]

    async def iter_messages(self, *, agent_run_id: str) -> AsyncIterator[ModelMessage]:
        checkpoint = await self.latest_checkpoint(agent_run_id=agent_run_id, include_interrupted=True)
        if checkpoint is not None:
            for message in checkpoint.messages:
                yield message

    async def load_loaded_model_context(self, *, owner_id: str) -> LoadedModelContext:
        snapshot = await self.latest_snapshot(run_id=owner_id, include_interrupted=True)
        if snapshot is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        return LoadedModelContext(tuple(LoadedContextMessage(message, None) for message in snapshot.messages))

    async def list_model_interactions(
        self, *, agent_run_id: str, after_model_request_seq: int | None = None,
        limit: int | None = None,
    ) -> list[object]:
        return []

    async def model_interaction_count(self, *, agent_run_id: str) -> int:
        return 0

    async def resolve_model_interactions(
        self, interactions: Sequence[object], *, include_client_requests: bool = False,
        include_content: bool = True,
    ) -> list[object]:
        if interactions:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return []
