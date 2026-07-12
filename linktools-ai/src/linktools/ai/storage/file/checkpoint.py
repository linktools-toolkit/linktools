#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FileCheckpointStore: root/{run_id}/{sequence}.bin (raw payload) + a JSON
sidecar for id/format/schema_version/created_at/metadata.

Each public async method delegates to a ``_*_sync`` private method via
``asyncio.to_thread`` so blocking file I/O never runs on the event loop."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from ...run.models import RunCheckpoint


def _validate_id_segment(value: str, *, kind: str) -> str:
    if not value or "/" in value or "\\" in value or value in (".", ".."):
        raise ValueError(f"invalid {kind}: {value!r}")
    return value


class FileCheckpointStore:
    def __init__(self, *, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: str) -> Path:
        d = self._root / _validate_id_segment(run_id, kind="run_id")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _payload_path(self, run_id: str, sequence: int) -> Path:
        return self._run_dir(run_id) / f"{sequence}.bin"

    def _meta_path(self, run_id: str, sequence: int) -> Path:
        return self._run_dir(run_id) / f"{sequence}.json"

    def _save_sync(self, checkpoint: RunCheckpoint) -> None:
        self._payload_path(checkpoint.run_id, checkpoint.sequence).write_bytes(
            checkpoint.payload
        )
        meta = {
            "id": checkpoint.id,
            "format": checkpoint.format,
            "schema_version": checkpoint.schema_version,
            "created_at": checkpoint.created_at.isoformat(),
            "metadata": dict(checkpoint.metadata),
        }
        self._meta_path(checkpoint.run_id, checkpoint.sequence).write_text(
            json.dumps(meta)
        )

    async def save(self, checkpoint: RunCheckpoint) -> None:
        await asyncio.to_thread(self._save_sync, checkpoint)

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
