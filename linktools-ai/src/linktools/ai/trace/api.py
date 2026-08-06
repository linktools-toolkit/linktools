#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Trace recorder and reader protocols with sequence validation."""

from datetime import datetime
from typing import Protocol

from linktools.core import environ

from ..domain.trace import RunSnapshot, TraceEvent
from ..foundation.errors import ErrorCode, LinktoolsAIError

logger = environ.get_logger("ai.trace.recorder")


class TraceRecorder(Protocol):
    async def append(self, event: TraceEvent) -> TraceEvent: ...
    async def list(self, execution_id: str, after_sequence: int = 0) -> "tuple[TraceEvent, ...]": ...
    async def snapshot(self, snapshot: RunSnapshot) -> RunSnapshot: ...


class InMemoryTraceRecorder:
    """Small deterministic recorder useful for local runs and contract tests."""

    def __init__(self) -> None:
        self._events: "dict[str, list[TraceEvent]]" = {}
        self._snapshots: "dict[str, RunSnapshot]" = {}

    async def append(self, event: TraceEvent) -> TraceEvent:
        events = self._events.setdefault(event.execution_id, [])
        for existing in events:
            if existing.sequence == event.sequence:
                if existing == event:
                    return existing
                raise LinktoolsAIError(ErrorCode.TRACE_SEQUENCE_CONFLICT, "trace sequence conflict")
        if not events and event.sequence != 1:
            raise LinktoolsAIError(ErrorCode.TRACE_SEQUENCE_CONFLICT, "trace sequence must start at one")
        if events and event.sequence <= events[-1].sequence:
            raise LinktoolsAIError(ErrorCode.TRACE_SEQUENCE_CONFLICT, "trace sequence must increase")
        events.append(event)
        logger.debug("trace event appended execution=%s sequence=%s", event.execution_id, event.sequence)
        return event

    async def list(self, execution_id: str, after_sequence: int = 0) -> "tuple[TraceEvent, ...]":
        return tuple(event for event in self._events.get(execution_id, ()) if event.sequence > after_sequence)

    async def snapshot(self, snapshot: RunSnapshot) -> RunSnapshot:
        existing = self._snapshots.get(snapshot.snapshot_id)
        if existing is not None and existing != snapshot:
            raise LinktoolsAIError(ErrorCode.TRACE_SEQUENCE_CONFLICT, "snapshot is immutable")
        self._snapshots[snapshot.snapshot_id] = snapshot
        return snapshot


__all__ = ["InMemoryTraceRecorder", "TraceRecorder"]
