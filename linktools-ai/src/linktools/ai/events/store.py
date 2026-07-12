#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EventStore: append-only. Events are never overwritten or deleted.

The EventStore is the SOLE owner of event sequence
assignment -- callers pass the payload plus the run/stream context, and the
store constructs the EventEnvelope (assigning the next sequence atomically,
minting event_id, and stamping occurred_at)."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .envelope import EventEnvelope
from .payloads import EventPayload


@dataclass(frozen=True, slots=True)
class EventPage:
    items: "tuple[EventEnvelope, ...]"
    cursor: "str | None"


@runtime_checkable
class EventStore(Protocol):
    async def append(
        self,
        *,
        stream_id: str,
        run_id: str,
        root_run_id: str,
        parent_run_id: "str | None",
        session_id: str,
        runnable_id: str,
        payload: EventPayload,
    ) -> EventEnvelope: ...

    async def list(
        self, stream_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> EventPage: ...
