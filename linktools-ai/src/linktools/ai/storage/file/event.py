#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FileEventStore: root/{stream_id}/{sequence:010d}.json, one file per event,
never overwritten -- append-only per spec docs/linktools-ai.md section 23.3.
The payload's concrete type name is stored alongside the payload's __dict__
so list() can reconstruct the exact dataclass.

Per review doc §8.1/§8.5, the store is the SOLE owner of sequence assignment.
append() takes the payload plus run/stream context and assigns the next
sequence atomically under a per-stream lock (single-process file mode per
§19.5 -- an asyncio.Lock per stream serializes appends within one event loop)."""

import asyncio
import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ...events import payloads as _payloads_module
from ...events.envelope import EventEnvelope
from ...events.payloads import EventPayload
from ...events.store import EventPage


def _validate_id_segment(value: str, *, kind: str) -> str:
    if not value or "/" in value or "\\" in value or value in (".", ".."):
        raise ValueError(f"invalid {kind}: {value!r}")
    return value


class FileEventStore:
    def __init__(self, *, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        # Per-stream asyncio.Lock so concurrent appends to the same stream
        # serialize their read-max-then-write and never collide on a sequence.
        self._stream_locks: "dict[str, asyncio.Lock]" = {}
        self._streams_guard = asyncio.Lock()

    async def _stream_lock(self, stream_id: str) -> asyncio.Lock:
        async with self._streams_guard:
            lock = self._stream_locks.get(stream_id)
            if lock is None:
                lock = asyncio.Lock()
                self._stream_locks[stream_id] = lock
            return lock

    def _stream_dir(self, stream_id: str) -> Path:
        d = self._root / _validate_id_segment(stream_id, kind="stream_id")
        d.mkdir(parents=True, exist_ok=True)
        return d

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
    ) -> EventEnvelope:
        # Atomic per-stream sequence assignment: hold the lock across the
        # read-max + write so a concurrent append to the same stream cannot
        # observe the same max and reuse a sequence number.
        lock = await self._stream_lock(stream_id)
        async with lock:
            stream_dir = self._stream_dir(stream_id)
            existing = list(stream_dir.glob("*.json"))
            next_seq = max((int(p.stem) for p in existing), default=0) + 1
            event_id = str(uuid.uuid4())
            occurred_at = datetime.now(timezone.utc)
            payload_type = type(payload).__name__
            raw = {
                "event_id": event_id, "sequence": next_seq, "occurred_at": occurred_at.isoformat(),
                "run_id": run_id, "root_run_id": root_run_id, "parent_run_id": parent_run_id,
                "session_id": session_id, "runnable_id": runnable_id,
                "payload_type": payload_type, "payload": asdict(payload),
            }
            path = stream_dir / f"{next_seq:010d}.json"
            path.write_text(json.dumps(raw))
            return EventEnvelope(
                event_id=event_id, sequence=next_seq, occurred_at=occurred_at,
                run_id=run_id, root_run_id=root_run_id, parent_run_id=parent_run_id,
                session_id=session_id, runnable_id=runnable_id, payload=payload,
            )

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
        run_dir = self._root / _validate_id_segment(run_id, kind="run_id")
        if not run_dir.exists():
            return EventPage(items=(), cursor=None)
        items = []
        for path in sorted(run_dir.glob("*.json")):
            envelope = self._load(path)
            if envelope.sequence <= after_sequence:
                continue
            items.append(envelope)
        return EventPage(items=tuple(items[:limit]), cursor=None)
