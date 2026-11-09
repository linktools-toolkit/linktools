#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpoint state does not multiply durable business message bodies."""

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, ModelMessagesTypeAdapter, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai_harness.media import DiskMediaStore
from pydantic_ai_harness.step_persistence import ContinuableSnapshot
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.persistence import SqlStepStore
from linktools.ai.persistence._step_store import _snapshot_descriptor_json
from linktools.ai.persistence.runs import NativeCheckpoint, OfficialStepReader

from .test_atomic_step_sql import atomic_store as atomic_store


def test_snapshot_descriptor_preserves_boundary_payload_and_rejects_overflow() -> None:
    checkpoint = {"output_retries_used": 0, "message_sha256": "0" * 64}
    descriptor = {"stored_messages": 1, "checkpoint": checkpoint}
    overhead = len(json.dumps(descriptor).encode("utf-8")) - 1
    checkpoint["output_retries_used"] = 10 ** (1024 - overhead - 1)
    encoded = _snapshot_descriptor_json(descriptor)
    assert len(encoded.encode("utf-8")) == 1024
    assert json.loads(encoded) == descriptor
    checkpoint["output_retries_used"] *= 10
    with pytest.raises(ValueError, match="descriptor exceeds 1024 UTF-8 bytes"):
        _snapshot_descriptor_json(descriptor)


@pytest.mark.asyncio
async def test_oversized_checkpoint_descriptor_fails_before_snapshot_write(atomic_store) -> None:
    store, _ = atomic_store
    snapshot = NativeCheckpoint.encode(ContinuableSnapshot(run_id="run", step_index=1,
        messages=[ModelRequest(parts=[UserPromptPart("question")])]), output_retries_used=10 ** 1024)
    with pytest.raises(ValueError, match="descriptor exceeds 1024 UTF-8 bytes"):
        await store.save_snapshot(snapshot)
    assert await store.list_snapshots(run_id="run", include_interrupted=True) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("messages", (
    [],
    [ModelRequest(parts=[UserPromptPart("问题：保留中文与数值")], metadata={"nested": [1.0, -0.0, True]})],
    [ModelResponse(parts=[TextPart("答复" * 16384)], metadata={
        "linktools.ai.checkpoint-state": {"protocol": "linktools.ai.native-checkpoint", "application": True},
    })],
))
async def test_checkpoint_changes_share_message_bodies_and_restore_exact_state(atomic_store, messages) -> None:
    store, engine = atomic_store
    first = ContinuableSnapshot(run_id="run", step_index=1, messages=messages, idempotency_key="first")
    second = replace(first, step_index=2, timestamp=first.timestamp + timedelta(seconds=1),
                     idempotency_key="second", state="interrupted")
    encoded = [NativeCheckpoint.encode(first, output_retries_used=1),
               NativeCheckpoint.encode(second, output_retries_used=2)]
    for snapshot in encoded:
        await store.save_snapshot(snapshot)
    await store.save_snapshot(encoded[-1])
    cold = SqlStepStore(engine, namespace="steps", tenant_id="one")
    try:
        assert await cold.list_snapshots(run_id="run", include_interrupted=True) == encoded
        restored = await OfficialStepReader(cold).latest_snapshot(run_id="run", include_interrupted=True)
        assert restored.messages == messages
        assert restored.output_retries_used == 2
        async with engine.connect() as connection:
            bodies = (await connection.execute(text("SELECT message FROM ai_step_messages"))).scalars().all()
            assert len(bodies) == len(messages)
            assert all("\\u" not in body for body in bodies)
        # Re-publishing an ordinary business message reuses the same body.
        if messages and not messages[-1].metadata.get("linktools.ai.checkpoint-state"):
            await cold.save_snapshot(replace(first, idempotency_key="ordinary"))
            async with engine.connect() as connection:
                assert await connection.scalar(text("SELECT COUNT(*) FROM ai_step_messages")) == len(messages)
    finally:
        await cold.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ("budget", "boolean", "binding", "digest", "shape"))
async def test_checkpoint_descriptor_rejects_corruption_and_cross_snapshot_binding(atomic_store, damage) -> None:
    store, engine = atomic_store
    snapshot = NativeCheckpoint.encode(ContinuableSnapshot(run_id="run", step_index=1,
        messages=[ModelRequest(parts=[UserPromptPart("question")])], idempotency_key="first"),
        output_retries_used=2)
    await store.save_snapshot(snapshot)
    async with engine.begin() as connection:
        descriptor = json.loads(await connection.scalar(text("SELECT messages FROM ai_step_snapshots")))
        if damage == "binding":
            await connection.execute(text("UPDATE ai_step_snapshots SET step_index = 9"))
        else:
            checkpoint = descriptor["checkpoint"]
            if damage in {"budget", "boolean"}:
                checkpoint["output_retries_used"] = 0 if damage == "budget" else True
            elif damage == "digest":
                checkpoint["message_sha256"] = "0" * 64
            else:
                checkpoint["extra"] = 1
            await connection.execute(text("UPDATE ai_step_snapshots SET messages = :value"),
                                     {"value": json.dumps(descriptor)})
    with pytest.raises(ValueError, match="checkpoint state"):
        await store.latest_snapshot(run_id="run")


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", (
    {"application": "保持"},
    {"payload": {"data": "YWJj", "kind": "binary", "media_type": "image/png"}},
))
async def test_checkpoint_digest_roundtrips_externalized_media(tmp_path: Path, metadata) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'media.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="one",
                         media_store=DiskMediaStore(directory=tmp_path / "media"), media_threshold_bytes=1)
    snapshot = NativeCheckpoint.encode(ContinuableSnapshot(run_id="run", step_index=2,
        messages=[ModelRequest(parts=[UserPromptPart([
            "查看图像", BinaryContent(data=b"synthetic-image", media_type="image/png"),
        ])], metadata=metadata)]), output_retries_used=3)
    try:
        await store.initialize(local=True)
        await store.save_snapshot(snapshot)
        restored = await store.latest_snapshot(run_id="run")
        assert replace(restored, messages=snapshot.messages) == snapshot
        assert json.loads(ModelMessagesTypeAdapter.dump_json(restored.messages)) == json.loads(
            ModelMessagesTypeAdapter.dump_json(snapshot.messages))
        assert NativeCheckpoint.decode(restored).output_retries_used == 3
    finally:
        await store.close()
        await engine.dispose()
