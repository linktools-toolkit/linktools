#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded, independent Trace delivery for interactive Session attempts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING

from linktools import environ

from ..persistence.contracts import TranscriptSpanRef
from ..persistence.runs import ModelInteractionStateStepArchive, StagingAgentRunStore, StepEvent

if TYPE_CHECKING:
    from ._canonical_trace_interactions import _CapturedTurnInteractions

_logger = environ.get_logger(__name__)
_MAX_BATCH_PAYLOAD_BYTES = 4 * 1024 * 1024
_MAX_BATCHES = 16
_MAX_BATCH_ENTRIES = 256


class BoundedTraceStaging(StagingAgentRunStore):
    """Bound the optional request and tool journal while the turn runs."""

    def __init__(self) -> None:
        super().__init__()
        self.trace_discarded = False
        self._trace_bytes = 0
        self._trace_events_seen = 0

    def _discard_trace(self) -> None:
        self.trace_discarded = True
        self._payloads.clear()
        self._interactions.clear()
        self._interaction_dirty.clear()
        self._prepared_interactions.clear()
        for events in self._events.values():
            events[:] = [event for event in events if not event.event_type.startswith(("MODEL_REQUEST_", "TOOL_CALL_"))]

    def intern_payload(self, agent_run_id: str, payload: bytes) -> tuple[str, int]:
        digest, size = sha256(payload).hexdigest(), len(payload)
        if not self.trace_discarded:
            interned = self._payloads.setdefault(agent_run_id, {})
            if digest not in interned:
                if self._trace_bytes + size > _MAX_BATCH_PAYLOAD_BYTES:
                    self._discard_trace()
                else:
                    interned[digest] = bytes(payload)
                    self._trace_bytes += size
        return digest, size

    def append_event_local(self, event: StepEvent) -> None:
        if event.event_type.startswith(("MODEL_REQUEST_", "TOOL_CALL_")):
            if self.trace_discarded:
                return
            if self._trace_events_seen >= _MAX_BATCH_ENTRIES:
                self._discard_trace()
                return
            self._trace_events_seen += 1
        super().append_event_local(event)

    def stage_model_interaction(self, interaction: object) -> None:
        if not self.trace_discarded:
            super().stage_model_interaction(interaction)

    def prepare_model_interaction(self, interaction: object) -> None:
        if not self.trace_discarded:
            super().prepare_model_interaction(interaction)


@dataclass(frozen=True, slots=True)
class SessionTraceBatch:
    capture: _CapturedTurnInteractions
    outcome: str
    error_code: str | None


class LossySessionTrace:
    """Serialize bounded detached batches. A failed write is never retried."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[ModelInteractionStateStepArchive, SessionTraceBatch]] = asyncio.Queue(_MAX_BATCHES)
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    def offer(self, archive: object, batch: SessionTraceBatch) -> None:
        captured = batch.capture
        if (self._closed or self._queue.full() or not isinstance(archive, ModelInteractionStateStepArchive)
                or len(captured.events) + len(captured.interactions) + len(captured.local_messages) > _MAX_BATCH_ENTRIES
                or sum(len(payload) for payload in captured.payloads.values()) > _MAX_BATCH_PAYLOAD_BYTES):
            return
        self._queue.put_nowait((archive, batch))
        if self._task is None:
            self._task = asyncio.create_task(self._write(), name="linktools-session-trace")

    async def close(self) -> None:
        self._closed = True
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        while not self._queue.empty():
            self._queue.get_nowait()

    async def _write(self) -> None:
        while True:
            archive, batch = await self._queue.get()
            try:
                await self._append(archive, batch)
            except Exception as error:
                # Trace may carry model/tool payloads. Never log exception text.
                _logger.warning("Optional Session Trace discarded: %s", type(error).__name__)
            finally:
                del archive, batch

    @staticmethod
    async def _append(archive: ModelInteractionStateStepArchive, batch: SessionTraceBatch) -> None:
        capture = batch.capture
        # Optional evidence must not resolve bodies through the business
        # Conversation/Execution archives. Drop any non-local dependency.
        for interaction in capture.interactions:
            for projection in (interaction.request_context, interaction.response_context,
                               *(attempt.response_context for attempt in interaction.stream_attempts or ())):
                if projection is not None and any(isinstance(part, TranscriptSpanRef) for part in projection.items):
                    return
        event = StepEvent(
            capture.run.agent_run_id,
            {"succeeded": "AGENT_RUN_SUCCEEDED", "waiting": "AGENT_RUN_INTERRUPTED"}.get(
                batch.outcome, "AGENT_RUN_FAILED",
            ),
            0,
            agent_conversation_id=capture.run.agent_conversation_id, agent_id=capture.run.agent_id,
            metadata={"outcome": batch.outcome, **({} if batch.error_code is None else {"error_code": batch.error_code})},
        )
        content_store, append = await capture.prepare_terminal_append(archive)
        stores = tuple(dict.fromkeys((archive.state_store, content_store)))

        async def write(transaction):
            run = await archive._append_lossy_session_trace_in_transaction(
                transaction.transaction(archive.state_store), capture.run, (*capture.events, event),
            )
            if run is not None:
                await append(transaction, run)

        await archive.state_store.storage_group.mutate(stores, write)
