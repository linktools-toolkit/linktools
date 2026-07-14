#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FileCheckpointStore: root/{run_id}/{sequence}.bin (raw payload) + a JSON
sidecar for id/format/schema_version/created_at/metadata.

The Store owns sequence assignment: callers submit a NewRunCheckpoint and
receive the persisted RunCheckpoint with id/sequence/created_at filled in.
append() takes a per-run asyncio.Lock so two concurrent appends for the same
run cannot race on the max-sequence read, then writes both files via a .tmp +
os.replace so a crash never leaves a half-written checkpoint visible. Each
public async method delegates to a ``_*_sync`` private method via
``asyncio.to_thread`` so blocking file I/O never runs on the event loop."""

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ...run.models import NewRunCheckpoint, RunCheckpoint


def _validate_id_segment(value: str, *, kind: str) -> str:
    if not value or "/" in value or "\\" in value or value in (".", ".."):
        raise ValueError(f"invalid {kind}: {value!r}")
    return value


class FileCheckpointStore:
    def __init__(self, *, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._locks: "dict[str, asyncio.Lock]" = {}

    def _lock_for(self, run_id: str) -> asyncio.Lock:
        lock = self._locks.get(run_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[run_id] = lock
        return lock

    def _run_dir(self, run_id: str) -> Path:
        d = self._root / _validate_id_segment(run_id, kind="run_id")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _payload_path(self, run_id: str, sequence: int) -> Path:
        return self._run_dir(run_id) / f"{sequence}.bin"

    def _meta_path(self, run_id: str, sequence: int) -> Path:
        return self._run_dir(run_id) / f"{sequence}.json"

    def _next_sequence(self, run_id: str) -> int:
        run_dir = self._root / _validate_id_segment(run_id, kind="run_id")
        if not run_dir.exists():
            return 1
        existing = [int(p.stem) for p in run_dir.glob("*.json")]
        return (max(existing) + 1) if existing else 1

    def _append_sync(self, new: NewRunCheckpoint) -> RunCheckpoint:
        sequence = self._next_sequence(new.run_id)
        checkpoint_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        payload_path = self._payload_path(new.run_id, sequence)
        meta_path = self._meta_path(new.run_id, sequence)
        meta = {
            "id": checkpoint_id,
            "format": new.format,
            "schema_version": new.schema_version,
            "created_at": created_at.isoformat(),
            "metadata": dict(new.metadata),
        }
        # Write both files via .tmp + os.replace (atomic on POSIX) so a crash
        # never leaves a half-written checkpoint visible: _load requires BOTH
        # files, and _latest globs the .json sidecar, so a checkpoint appears
        # only once its sidecar is in place.
        payload_tmp = payload_path.with_suffix(".bin.tmp")
        meta_tmp = meta_path.with_suffix(".json.tmp")
        payload_tmp.write_bytes(new.payload)
        meta_tmp.write_text(json.dumps(meta))
        os.replace(payload_tmp, payload_path)
        os.replace(meta_tmp, meta_path)
        return RunCheckpoint(
            id=checkpoint_id,
            run_id=new.run_id,
            sequence=sequence,
            format=new.format,
            schema_version=new.schema_version,
            payload=new.payload,
            created_at=created_at,
            metadata=dict(new.metadata),
        )

    async def append(self, checkpoint: NewRunCheckpoint) -> RunCheckpoint:
        async with self._lock_for(checkpoint.run_id):
            return await asyncio.to_thread(self._append_sync, checkpoint)

    def _load(self, run_id: str, sequence: int) -> "RunCheckpoint | None":
        meta_path = self._meta_path(run_id, sequence)
        payload_path = self._payload_path(run_id, sequence)
        if not meta_path.exists() or not payload_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        return RunCheckpoint(
            id=meta["id"],
            run_id=run_id,
            sequence=sequence,
            format=meta["format"],
            schema_version=meta["schema_version"],
            payload=payload_path.read_bytes(),
            created_at=datetime.fromisoformat(meta["created_at"]),
            metadata=meta["metadata"],
        )

    def _latest_sync(self, run_id: str) -> "RunCheckpoint | None":
        run_dir = self._root / _validate_id_segment(run_id, kind="run_id")
        if not run_dir.exists():
            return None
        sequences = sorted((int(p.stem) for p in run_dir.glob("*.json")), reverse=True)
        if not sequences:
            return None
        return self._load(run_id, sequences[0])

    async def latest(self, run_id: str) -> "RunCheckpoint | None":
        return await asyncio.to_thread(self._latest_sync, run_id)

    def _get_sync(self, checkpoint_id: str) -> "RunCheckpoint | None":
        for run_dir in self._root.iterdir():
            if not run_dir.is_dir():
                continue
            for meta_path in run_dir.glob("*.json"):
                meta = json.loads(meta_path.read_text())
                if meta["id"] == checkpoint_id:
                    return self._load(run_dir.name, int(meta_path.stem))
        return None

    async def get(self, checkpoint_id: str) -> "RunCheckpoint | None":
        return await asyncio.to_thread(self._get_sync, checkpoint_id)
