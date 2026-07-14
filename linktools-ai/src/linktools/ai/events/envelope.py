#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EventEnvelope[TEvent]: the strongly-typed wrapper every event payload travels
in. Generic (not PEP-695 `class Foo[T]`) for Python 3.10 support."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, Mapping, TypeVar

TEvent = TypeVar("TEvent")


@dataclass(frozen=True, slots=True)
class EventEnvelope(Generic[TEvent]):
    event_id: str
    # stream_id is the sequence-uniqueness boundary (a
    # session/audit/root-run/swarm stream, not necessarily the run itself);
    # every current caller passes stream_id == run_id, but the field is now
    # first-class rather than conflated with run_id.
    stream_id: str
    sequence: int
    occurred_at: datetime
    run_id: str
    root_run_id: str
    parent_run_id: "str | None"
    session_id: str
    runnable_id: str
    payload: TEvent
    # Free-form per-event metadata. The FileRunCommitCoordinator tags the
    # critical pause/complete events with ``commit_id`` so it can dedup by
    # (run_id, commit_id, event_type) -- a run may legitimately pause more
    # than once (one event per approval), so deduping by event type alone
    # would either drop a legitimate second pause or duplicate on recovery.
    metadata: "Mapping[str, Any]" = field(default_factory=dict)
