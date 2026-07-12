#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CheckpointStore: keyed by (run_id, sequence) -- NOT by session_id, fixing
the durability bug in the pre-vNext checkpoint/ module (a fresh agent instance
for the same session restarted the sequence counter at 1, silently
overwriting that session's prior checkpoint)."""

from typing import Protocol, runtime_checkable

from .models import RunCheckpoint


@runtime_checkable
class CheckpointStore(Protocol):
    async def save(self, checkpoint: RunCheckpoint) -> None: ...

    async def latest(self, run_id: str) -> "RunCheckpoint | None": ...

    async def get(self, checkpoint_id: str) -> "RunCheckpoint | None": ...
