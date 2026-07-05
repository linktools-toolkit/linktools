#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EventEnvelope[TEvent]: the strongly-typed wrapper every event payload travels
in. Generic (not PEP-695 `class Foo[T]`) for 3.10 compatibility."""

from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar

TEvent = TypeVar("TEvent")


@dataclass(frozen=True, slots=True)
class EventEnvelope(Generic[TEvent]):
    event_id: str
    sequence: int
    occurred_at: datetime
    run_id: str
    root_run_id: str
    parent_run_id: "str | None"
    session_id: str
    runnable_id: str
    payload: TEvent
