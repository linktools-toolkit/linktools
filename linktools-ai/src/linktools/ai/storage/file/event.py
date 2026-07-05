#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FileEventStore: root/{run_id}/{sequence:010d}.json, one file per event,
never overwritten -- append-only per spec docs/linktools-ai.md section 23.3.
The payload's concrete type name is stored alongside the payload's __dict__
so list() can reconstruct the exact dataclass."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ...errors import EventSequenceConflictError
from ...events import payloads as _payloads_module
from ...events.envelope import EventEnvelope
from ...events.store import EventPage


class FileEventStore:
    def __init__(self, *, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: str) -> Path:
        d = self._root / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _event_path(self, run_id: str, sequence: int) -> Path:
        return self._run_dir(run_id) / f"{sequence:010d}.json"

    async def append(self, event: EventEnvelope, *, expected_sequence: "int | None" = None) -> EventEnvelope:
        path = self._event_path(event.run_id, event.sequence)
        if expected_sequence is not None and path.exists():
            raise EventSequenceConflictError(
                f"event already exists at sequence {event.sequence} for run {event.run_id}"
            )
        payload_type = type(event.payload).__name__
        raw = {
            "event_id": event.event_id, "sequence": event.sequence, "occurred_at": event.occurred_at.isoformat(),
            "run_id": event.run_id, "root_run_id": event.root_run_id, "parent_run_id": event.parent_run_id,
            "session_id": event.session_id, "runnable_id": event.runnable_id,
            "payload_type": payload_type, "payload": asdict(event.payload),
        }
        path.write_text(json.dumps(raw))
        return event

    def _load(self, path: Path) -> EventEnvelope:
        raw = json.loads(path.read_text())
        payload_cls = getattr(_payloads_module, raw["payload_type"])
        payload = payload_cls(**raw["payload"])
        return EventEnvelope(
            event_id=raw["event_id"], sequence=raw["sequence"], occurred_at=datetime.fromisoformat(raw["occurred_at"]),
            run_id=raw["run_id"], root_run_id=raw["root_run_id"], parent_run_id=raw["parent_run_id"],
            session_id=raw["session_id"], runnable_id=raw["runnable_id"], payload=payload,
        )

    async def list(self, run_id: str, *, after_sequence: int = 0, limit: int = 100) -> EventPage:
        run_dir = self._root / run_id
        if not run_dir.exists():
            return EventPage(items=(), cursor=None)
        items = []
        for path in sorted(run_dir.glob("*.json")):
            envelope = self._load(path)
            if envelope.sequence <= after_sequence:
                continue
            items.append(envelope)
        return EventPage(items=tuple(items[:limit]), cursor=None)
