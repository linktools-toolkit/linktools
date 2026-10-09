#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read the existing process-local owner for explicit transient execution routes."""

from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import datetime
from typing import Protocol

from ..errors import AIError, ErrorCode
from .state._step_archive import StagingAgentRunStore
from .state._step_contracts import AgentRunHistoryCapture, AgentRunRecord, StepEvent


class _InteractionResolver(Protocol):
    async def resolve_model_interactions(self, interactions: Sequence[object]) -> Sequence[object]: ...


class TransientExecutionHistoryStore:
    """Adapt transient recorder facts without retaining another history copy."""

    def __init__(self, source: StagingAgentRunStore, resolver: _InteractionResolver) -> None:
        self._source = source
        self._resolver = resolver

    async def get_agent_run(self, *, agent_run_id: str) -> AgentRunRecord | None:
        return self._source.get_agent_run_local(agent_run_id)

    async def capture_history(
        self, agent_run_ids: Sequence[str], *, include_pending: bool = False,
    ) -> Mapping[str, AgentRunHistoryCapture]:
        # Local getters do not yield: every identity and high water belongs to
        # the same event-loop boundary, including a frozen pending message.
        result: dict[str, AgentRunHistoryCapture] = {}
        for run_id in dict.fromkeys(agent_run_ids):
            run = self._source.get_agent_run_local(run_id)
            events = tuple(self._source.list_events_local(run_id))
            transcript = self._source.staged_transcript(run_id)
            interactions = self._source.capture_interactions_local(run_id)
            if run is None and (events or transcript is not None or interactions):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result[run_id] = AgentRunHistoryCapture(
                run, len(events), 0 if transcript is None else len(transcript.messages),
                0 if not interactions else interactions[-1].model_request_seq,
                pending_keys=() if transcript is None or not include_pending else transcript.pending_keys,
                pending_message=None if transcript is None or not include_pending else transcript.pending,
                inline_events=events,
            )
        return result

    async def read_pending_message(self, capture: AgentRunHistoryCapture) -> object | None:
        return capture.pending_message

    async def iter_message_range(self, *, agent_run_id: str, start: int, end: int) -> AsyncIterator[object]:
        transcript = self._source.staged_transcript(agent_run_id)
        if transcript is None or start < 0 or end < start or end > len(transcript.messages):
            raise AIError(ErrorCode.CURSOR_INVALID)
        for message in transcript.messages[start:end]:
            yield message

    async def list_event_range(self, *, agent_run_id: str, start: int, end: int) -> list[StepEvent]:
        events = self._source.list_events_local(agent_run_id)
        if start < 0 or end < start or end > len(events):
            raise AIError(ErrorCode.CURSOR_INVALID)
        return events[start:end]

    async def read_history_associations(
        self, *, agent_run_id: str, message_seqs: Sequence[int],
        tool_call_ids: Sequence[str], event_high_water: int,
    ) -> list[StepEvent]:
        del message_seqs, tool_call_ids
        return await self.list_event_range(agent_run_id=agent_run_id, start=0, end=event_high_water)

    async def list_trace_events(
        self, *, agent_run_id: str, event_high_water: int,
        after_timestamp: datetime | None, after_sequence: int, limit: int,
        model_request_seq: int | None = None, step_index: int | None = None,
        tool_call_id: str | None = None,
    ) -> list[tuple[int, StepEvent]]:
        events = await self.list_event_range(agent_run_id=agent_run_id, start=0, end=event_high_water)
        selected: list[tuple[int, StepEvent]] = []
        for sequence, event in enumerate(events, 1):
            if not event.event_type.startswith(("MODEL_REQUEST_", "TOOL_CALL_")):
                continue
            request = event.metadata.get("linktools.ai.model_request_seq")
            if request is not None and (not request.isdigit() or int(request) < 1):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if (after_timestamp is not None and (event.timestamp, sequence) <= (after_timestamp, after_sequence)
                    or model_request_seq is not None and (request is None or int(request) != model_request_seq)
                    or step_index is not None and event.step_index != step_index
                    or tool_call_id is not None and event.tool_call_id != tool_call_id):
                continue
            selected.append((sequence, event))
        selected.sort(key=lambda value: (value[1].timestamp, value[0]))
        return selected[:limit]

    async def list_model_interactions(
        self, *, agent_run_id: str, after_model_request_seq: int | None = None,
        limit: int | None = None,
    ) -> list[object]:
        return await self._source.list_model_interactions(
            agent_run_id=agent_run_id, after_model_request_seq=after_model_request_seq, limit=limit,
        )

    async def resolve_model_interactions(self, interactions: Sequence[object]) -> Sequence[object]:
        return await self._resolver.resolve_model_interactions(interactions)


__all__ = ["TransientExecutionHistoryStore"]
