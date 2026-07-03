#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CheckpointStore Protocol: opaque byte-snapshot storage, keyed by (session_id, seq).

The stored bytes' meaning is decided by the caller (a FileSession's context.json
content, a serialized SessionTranscriptHead for a RemoteSession, etc.) -- this
Protocol only manages storage/retrieval, not session reconstruction."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class CheckpointStore(Protocol):
    async def save(self, session_id: str, seq: int, content: bytes) -> str:
        """Store a snapshot, returning a checkpoint_id that `restore()` accepts."""
        ...

    async def list(self, session_id: str) -> "list[str]":
        """Return this session's checkpoint_ids, oldest first."""
        ...

    async def restore(self, checkpoint_id: str) -> bytes:
        """Return the raw bytes saved under checkpoint_id. Raises KeyError if unknown."""
        ...
