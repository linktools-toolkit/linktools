#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FilesystemEventStore: root/{stream_id}/{sequence:010d}.json, one file per event,
never overwritten -- append-only. The payload's concrete type name is stored
alongside the payload's __dict__ so list() can reconstruct the exact dataclass.

The store is the SOLE owner of sequence assignment. append() takes the
payload plus run/stream context and assigns the next sequence atomically
under a per-stream lock (single-process file mode; an asyncio.Lock per
stream serializes appends within one event loop).

Each public async method delegates to a ``_*_sync`` private method via
``asyncio.to_thread`` so blocking file I/O never runs on the event loop.
The per-stream ``asyncio.Lock`` is held in the async wrapper and spans the
``to_thread`` call, so concurrent appends to the same stream still serialize."""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ...events.envelope import EventEnvelope
from ...events.payloads import EventPayload
from ...events.registry import EventCodec, default_codec
from ...events.store import EventPage


def _validate_id_segment(value: str, *, kind: str) -> str:
    if not value or "/" in value or "\\" in value or value in (".", ".."):
        raise ValueError(f"invalid {kind}: {value!r}")
    return value


class FilesystemEventStore:
    def __init__(self, *, root: Path, codec: "EventCodec | None" = None) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        # Per-stream asyncio.Lock so concurrent appends to the same stream
        # serialize their read-max-then-write and never collide on a sequence.
        self._stream_locks: "dict[str, asyncio.Lock]" = {}
        self._streams_guard = asyncio.Lock()
        # Event wire contract: encode/decode by stable event_type, never by
        # the payload class name. File and SQL stores share one codec.
        self._codec: EventCodec = codec or default_codec

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

    def _append_sync(
        self,
        *,
        stream_id: str,
        run_id: str,
        root_run_id: str,
        parent_run_id: "str | None",
        session_id: str,
        runnable_id: str,
        payload: EventPayload,
        metadata: "Mapping[str, Any] | None" = None,
    ) -> EventEnvelope:
        stream_dir = self._stream_dir(stream_id)
        existing = list(stream_dir.glob("*.json"))
        next_seq = max((int(p.stem) for p in existing), default=0) + 1
        event_id = str(uuid.uuid4())
        occurred_at = datetime.now(timezone.utc)
        event_type, schema_version, payload_data = self._codec.encode(payload)
        meta = dict(metadata) if metadata else {}
        raw = {
            "event_id": event_id,
            "stream_id": stream_id,
            "sequence": next_seq,
            "occurred_at": occurred_at.isoformat(),
            "run_id": run_id,
            "root_run_id": root_run_id,
            "parent_run_id": parent_run_id,
            "session_id": session_id,
            "runnable_id": runnable_id,
            "event_type": event_type,
            "schema_version": schema_version,
            "payload": payload_data,
            "metadata": meta,
        }
        path = stream_dir / f"{next_seq:010d}.json"
        path.write_text(json.dumps(raw))
        return EventEnvelope(
            event_id=event_id,
            stream_id=stream_id,
            sequence=next_seq,
            occurred_at=occurred_at,
            run_id=run_id,
            root_run_id=root_run_id,
            parent_run_id=parent_run_id,
            session_id=session_id,
            runnable_id=runnable_id,
            payload=payload,
            metadata=meta,
        )

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
        metadata: "Mapping[str, Any] | None" = None,
    ) -> EventEnvelope:
        # Atomic per-stream sequence assignment: hold the lock across the
        # read-max + write so a concurrent append to the same stream cannot
        # observe the same max and reuse a sequence number. The lock spans
        # the to_thread call (not the other way around) -- holding an
        # asyncio.Lock across a to_thread await is fine; holding a sync lock
        # across an await would block the loop.
        lock = await self._stream_lock(stream_id)
        async with lock:
            return await asyncio.to_thread(
                self._append_sync,
                stream_id=stream_id,
                run_id=run_id,
                root_run_id=root_run_id,
                parent_run_id=parent_run_id,
                session_id=session_id,
                runnable_id=runnable_id,
                payload=payload,
                metadata=metadata,
            )

    def _load(self, path: Path) -> EventEnvelope:
        raw = json.loads(path.read_text())
        # Read by stable event_type. Envelopes written before the registry
        # existed carry ``payload_type`` (the class name) and no
        # ``event_type``/``schema_version``; fall back to that tag so history
        # remains readable. Each payload's event_type ClassVar literal matches
        # its class name, so the legacy tag is a valid event_type.
        event_type = raw.get("event_type") or raw.get("payload_type")
        schema_version = raw.get("schema_version")
        payload = self._codec.decode(event_type, schema_version, raw["payload"])
        return EventEnvelope(
            event_id=raw["event_id"],
            # fall back to run_id for files written before stream_id
            # became a first-class field -- every caller has always passed
            # stream_id == run_id, so this default is exact, not a guess.
            stream_id=raw.get("stream_id", raw["run_id"]),
            sequence=raw["sequence"],
            occurred_at=datetime.fromisoformat(raw["occurred_at"]),
            run_id=raw["run_id"],
            root_run_id=raw["root_run_id"],
            parent_run_id=raw["parent_run_id"],
            session_id=raw["session_id"],
            runnable_id=raw["runnable_id"],
            payload=payload,
            metadata=raw.get("metadata") or {},
        )

    def _list_sync(
        self, stream_id: str, *, after_sequence: int, limit: int
    ) -> EventPage:
        stream_dir = self._root / _validate_id_segment(stream_id, kind="stream_id")
        if not stream_dir.exists():
            return EventPage(items=(), cursor=None)
        items = []
        for path in sorted(stream_dir.glob("*.json")):
            envelope = self._load(path)
            if envelope.sequence <= after_sequence:
                continue
            items.append(envelope)
        return EventPage(items=tuple(items[:limit]), cursor=None)

    async def list(
        self, stream_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> EventPage:
        return await asyncio.to_thread(
            self._list_sync,
            stream_id,
            after_sequence=after_sequence,
            limit=limit,
        )
