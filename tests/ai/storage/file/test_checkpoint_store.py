#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/file/test_checkpoint_store.py"""
import pytest

from linktools.ai.run.models import RunCheckpoint
from linktools.ai.storage.file.checkpoint import FileCheckpointStore
from datetime import datetime, timezone


def _checkpoint(run_id="run-1", sequence=1) -> RunCheckpoint:
    return RunCheckpoint(
        id=f"{run_id}-{sequence}", run_id=run_id, sequence=sequence, format="msgpack",
        schema_version=1, payload=b"snapshot-bytes", created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_save_then_get_roundtrip(tmp_path):
    store = FileCheckpointStore(root=tmp_path)
    await store.save(_checkpoint())
    fetched = await store.get("run-1-1")
    assert fetched is not None
    assert fetched.payload == b"snapshot-bytes"
    assert fetched.sequence == 1


@pytest.mark.asyncio
async def test_get_missing_returns_none(tmp_path):
    store = FileCheckpointStore(root=tmp_path)
    assert await store.get("nope") is None


@pytest.mark.asyncio
async def test_latest_returns_highest_sequence(tmp_path):
    store = FileCheckpointStore(root=tmp_path)
    await store.save(_checkpoint(sequence=1))
    await store.save(_checkpoint(sequence=3))
    await store.save(_checkpoint(sequence=2))
    latest = await store.latest("run-1")
    assert latest.sequence == 3


@pytest.mark.asyncio
async def test_latest_for_unknown_run_returns_none(tmp_path):
    store = FileCheckpointStore(root=tmp_path)
    assert await store.latest("nope") is None


@pytest.mark.asyncio
async def test_checkpoints_for_different_runs_are_isolated(tmp_path):
    store = FileCheckpointStore(root=tmp_path)
    await store.save(_checkpoint(run_id="run-a", sequence=1))
    await store.save(_checkpoint(run_id="run-b", sequence=1))
    latest_a = await store.latest("run-a")
    latest_b = await store.latest("run-b")
    assert latest_a.run_id == "run-a"
    assert latest_b.run_id == "run-b"
