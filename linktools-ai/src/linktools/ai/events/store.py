#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EventStore: append-only. Events are never overwritten or deleted."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .envelope import EventEnvelope


@dataclass(frozen=True, slots=True)
class EventPage:
    items: "tuple[EventEnvelope, ...]"
    cursor: "str | None"


@runtime_checkable
class EventStore(Protocol):
    async def append(self, event: EventEnvelope, *, expected_sequence: "int | None" = None) -> EventEnvelope:
        ...

    async def list(self, run_id: str, *, after_sequence: int = 0, limit: int = 100) -> EventPage:
        ...
