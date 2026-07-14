#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CheckpointStore: keyed by (run_id, sequence). The Store owns sequence
assignment -- callers submit a NewRunCheckpoint (no id/sequence/created_at) and
receive the persisted RunCheckpoint. A caller can therefore never collide with
an existing checkpoint by hardcoding the sequence (the prior bug: a fresh
agent instance for the same session restarted the counter at the first sequence
and overwrote that session's prior checkpoint, and the SQL unique constraint on
(run_id, sequence) rejected the duplicate)."""

from typing import Protocol, runtime_checkable

from .models import NewRunCheckpoint, RunCheckpoint


@runtime_checkable
class CheckpointStore(Protocol):
    async def append(self, checkpoint: NewRunCheckpoint) -> RunCheckpoint: ...

    async def latest(self, run_id: str) -> "RunCheckpoint | None": ...

    async def get(self, checkpoint_id: str) -> "RunCheckpoint | None": ...
