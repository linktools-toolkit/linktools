#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FileCheckpointStore: local filesystem CheckpointStore implementation."""

import asyncio
from pathlib import Path


class FileCheckpointStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, session_id: str, seq: int) -> Path:
        return self.root / session_id / f"{seq}.bin"

    async def save(self, session_id: str, seq: int, content: bytes) -> str:
        path = self._path(session_id, seq)
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, content)
        return f"{session_id}:{seq}"

    async def list(self, session_id: str) -> "list[str]":
        session_dir = self.root / session_id
        if not await asyncio.to_thread(session_dir.exists):
            return []
        files = await asyncio.to_thread(lambda: sorted(session_dir.glob("*.bin"), key=lambda p: int(p.stem)))
        return [f"{session_id}:{path.stem}" for path in files]

    async def restore(self, checkpoint_id: str) -> bytes:
        session_id, _, seq_text = checkpoint_id.rpartition(":")
        if not session_id:
            raise KeyError(checkpoint_id)
        path = self._path(session_id, int(seq_text))
        if not await asyncio.to_thread(path.exists):
            raise KeyError(checkpoint_id)
        return await asyncio.to_thread(path.read_bytes)
