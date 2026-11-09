#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Output retry state belongs to the same durable boundary as model context."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSessionTransaction

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import SqlStepStore
from linktools.ai.persistence.contracts import ContextProjection, StoredAgentRunCheckpoint
from linktools.ai.persistence.runs import AgentRunCheckpoint, NativeCheckpoint, OfficialStepReader

from .test_atomic_step_sql import atomic_store as atomic_store
from .test_state_codec_protocol import _decode_step_envelope, _encode_step_envelope


@pytest.mark.parametrize("message", (
    ModelRequest(parts=[UserPromptPart("question")], metadata=None),
    ModelResponse(parts=[TextPart("answer")], metadata={
        "arbitrary": [None, 1, {"nested": True}],
        "linktools.ai.checkpoint-state": {"version": 999, "output_retries_used": "application data"},
    }),
))
def test_native_checkpoint_state_roundtrips_original_message_metadata(message) -> None:
    original = deepcopy(message)
    snapshot = ContinuableSnapshot(run_id="run", step_index=2, messages=[message], idempotency_key="point")
    encoded = NativeCheckpoint.encode(snapshot, output_retries_used=3)
    restored = NativeCheckpoint.decode(encoded)
    assert message == original
    assert restored.messages == [original]
    assert restored.output_retries_used == 3
    assert restored.run_id == snapshot.run_id and restored.idempotency_key == snapshot.idempotency_key
    assert NativeCheckpoint.encode(restored, output_retries_used=3) == encoded


@pytest.mark.parametrize("extra_metadata", ({}, {"arbitrary": 1}))
def test_raw_native_snapshot_preserves_same_named_application_metadata(extra_metadata) -> None:
    metadata = {**extra_metadata, "linktools.ai.checkpoint-state": {"version": 999}}
    snapshot = ContinuableSnapshot(run_id="run", step_index=2,
        messages=[ModelRequest(parts=[UserPromptPart("question")], metadata=metadata)])
    restored = OfficialStepReader.restore_snapshot(snapshot)
    assert restored == snapshot
    assert restored.messages[-1].metadata == metadata
    checkpoint = OfficialStepReader.project_checkpoint(RunRecord(run_id="run"), restored, include_transcript=False)
    assert checkpoint.output_retries_used is None


@pytest.mark.parametrize(("field", "value"), (
    ("version", 2), ("run_id", "other"), ("step_index", 9), ("state", "interrupted"),
    ("idempotency_key", "other"), ("output_retries_used", -1), ("output_retries_used", True),
    ("output_retries_used", "1"), ("metadata", []), ("extra", 1),
))
def test_native_checkpoint_rejects_corrupt_or_misbound_state(field, value) -> None:
    snapshot = ContinuableSnapshot(run_id="run", step_index=2,
        messages=[ModelRequest(parts=[UserPromptPart("question")])], idempotency_key="point")
    encoded = NativeCheckpoint.encode(snapshot, output_retries_used=3)
    encoded.messages[-1].metadata["linktools.ai.checkpoint-state"][field] = value
    with pytest.raises(AIError) as raised:
        NativeCheckpoint.decode(encoded)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_native_checkpoint_does_not_infer_missing_retry_state() -> None:
    snapshot = ContinuableSnapshot(run_id="run", step_index=2,
        messages=[ModelRequest(parts=[UserPromptPart("question")])])
    restored = NativeCheckpoint.decode(snapshot)
    assert restored.messages == snapshot.messages
    assert restored == snapshot
    projected = OfficialStepReader.project_checkpoint(RunRecord(run_id="run"), restored, include_transcript=False)
    assert projected.output_retries_used is None


@pytest.mark.parametrize("count", (0, 3))
def test_archive_checkpoint_codec_preserves_output_retry_count(count: int) -> None:
    checkpoint = StoredAgentRunCheckpoint("run", 1, datetime.now(timezone.utc), "interrupted",
                                         ContextProjection(()), 0, output_retries_used=count)
    restored = _decode_step_envelope(_encode_step_envelope(checkpoint))
    assert restored == checkpoint and restored.output_retries_used == count


@pytest.mark.parametrize("count", (-1, True, "1"))
def test_checkpoint_rejects_invalid_output_retry_count(count) -> None:
    with pytest.raises(ValueError):
        AgentRunCheckpoint("run", 1, [], output_retries_used=count)


@pytest.mark.asyncio
async def test_native_checkpoint_retry_state_survives_lost_commit_ack(atomic_store, monkeypatch) -> None:
    store, engine = atomic_store
    snapshot = ContinuableSnapshot(run_id="run", step_index=2,
        messages=[ModelRequest(parts=[UserPromptPart("question")], run_id="run", metadata={"original": 1})],
        idempotency_key="point")
    encoded = NativeCheckpoint.encode(snapshot, output_retries_used=3)
    run = RunRecord(run_id="run", registration_id="registration", metadata={"linktools.transcript_start": "0"})
    original = AsyncSessionTransaction.__aexit__
    lost = False

    async def lose_ack(self, error_type, error, traceback):
        nonlocal lost
        result = await original(self, error_type, error, traceback)
        if error_type is None and not lost:
            lost = True
            raise OperationalError("commit acknowledgement lost", {}, RuntimeError("lost"),
                                   connection_invalidated=True)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(AsyncSessionTransaction, "__aexit__", lose_ack)
        with pytest.raises(AIError) as raised:
            await store.commit_step(run=run, snapshot=encoded)
        assert raised.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
    cold = SqlStepStore(engine, namespace="steps", tenant_id="one")
    try:
        assert await cold.latest_snapshot(run_id="run") == encoded
        reader = OfficialStepReader(cold)
        checkpoint = await reader.latest_checkpoint(agent_run_id="run")
        assert checkpoint is not None and checkpoint.output_retries_used == 3
        assert checkpoint.messages == snapshot.messages
        restored = await reader.latest_snapshot(run_id="run")
        assert restored.messages == snapshot.messages
        await cold.commit_step(snapshot=encoded)
        assert (await reader.latest_checkpoint(agent_run_id="run")).output_retries_used == 3
        next_snapshot = NativeCheckpoint.encode(replace(snapshot, idempotency_key="next"), output_retries_used=4)
        await cold.commit_step(snapshot=next_snapshot)
        assert (await reader.latest_checkpoint(agent_run_id="run")).output_retries_used == 4
    finally:
        await cold.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("known", (False, True))
@pytest.mark.parametrize("messages", ([], [ModelRequest(parts=[UserPromptPart("question")], metadata={
    "linktools.ai.checkpoint-state": {"protocol": "linktools.ai.native-checkpoint", "application": True},
})]))
async def test_runtime_snapshot_reader_preserves_retry_state_and_decoded_metadata(known, messages) -> None:
    from pydantic_ai_harness.step_persistence import InMemoryStepStore
    from linktools.ai.runtime._step_store import RuntimeStepStore

    # Unknown upstream snapshots use ordinary application metadata, not an
    # envelope claiming the Runtime protocol.
    if not known and messages:
        messages = [replace(messages[0], metadata={"linktools.ai.checkpoint-state": {"application": True}})]
    snapshot = ContinuableSnapshot(run_id="run", step_index=2, messages=messages, idempotency_key="point")
    store = InMemoryStepStore()
    await store.register_run(RunRecord(run_id="run"))
    await store.save_snapshot(NativeCheckpoint.encode(snapshot, output_retries_used=3) if known else snapshot)
    runtime_store = RuntimeStepStore(store, None)
    restored = await runtime_store.latest_snapshot(run_id="run")
    assert restored.messages == messages
    assert isinstance(restored, NativeCheckpoint) is known
    projected = OfficialStepReader.project_checkpoint(RunRecord(run_id="run"), restored, include_transcript=False)
    assert projected.output_retries_used == (3 if known else None)
    assert OfficialStepReader.restore_snapshot(restored) == restored
    assert (await OfficialStepReader(runtime_store).latest_snapshot(run_id="run")) == restored
