#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Opaque pending checkpoint bytes do not depend on Trace retention or mirrors."""

from dataclasses import replace
import json

import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, UserPromptPart

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model.messages import encode_model_messages
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence._projection_payload import decode_projection_payload, encode_projection_payload
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint, AgentRunRecord

from ._runtime_test_helpers import native_checkpoint, seed_native_snapshot
from .test_step_projection import _execution
from .test_projection_payload import _capture


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpoint_state", [None, "complete", "interrupted"])
@pytest.mark.parametrize("include_interrupted", [False, True])
async def test_checkpoint_payload_freezes_latest_context_without_trace_or_source_io(tmp_path, monkeypatch, checkpoint_state, include_interrupted):
    state = Persistence(PersistenceStorage.sqlite(tmp_path / "context.db"))
    await state.initialize(namespace="checkpoint-payload", tenant_id="tenant")
    try:
        store = state.run_store
        store.bind_execution_producer(
            "run", execution_id="execution", producer_generation=1,
            conversation=True, restart_checkpoints=False, durable_trace=False,
        )
        run = AgentRunRecord("run", metadata={"value": "captured"})
        await store.register_agent_run(run, execution_id="execution")
        checkpoint = None
        if checkpoint_state is not None:
            checkpoint = AgentRunCheckpoint(
                "run", 1, [ModelRequest(parts=[UserPromptPart(["input", BinaryContent(b"input bytes", media_type="text/plain")])])],
                context_messages=[ModelRequest(parts=[UserPromptPart("full context")])],
                state=checkpoint_state,
            )
            store._staging.save_checkpoint_local(checkpoint)
            # Compacted Trace deltas do not remove the owning latest checkpoint.
            store._staging.trim_checkpoints_local("run", checkpoint_offset=1)
        store.trace_persistence.dirty.clear()
        archive = store.read_store(PersistenceDomain.TRACE)

        async def unexpected(*args, **kwargs):
            raise AssertionError("checkpoint byte capture must not perform source I/O")

        monkeypatch.setattr(archive.state_store, "read", unexpected)
        monkeypatch.setattr(archive.state_store, "mutate", unexpected)
        content = await store.capture_agent_run_checkpoint_payload(
            "run", execution_id="execution", producer_generation=1, include_interrupted=include_interrupted,
        )
        assert isinstance(content, bytes) and content
        run.metadata["value"] = "later"
        store._staging.release_agent_run_local("run")
        restored_run, restored = await store.decode_agent_run_checkpoint_payload(
            content, agent_run_id="run", producer_generation=1,
        )
        assert restored_run.agent_run_id == "run" and restored_run.metadata == {"value": "captured"}
        if checkpoint is None or checkpoint_state == "interrupted" and not include_interrupted:
            assert restored is None
        else:
            assert restored.state == checkpoint_state
            assert encode_model_messages(restored.messages) == encode_model_messages(checkpoint.messages)
            assert encode_model_messages(restored.context_messages) == encode_model_messages(checkpoint.context_messages)
        assert not store._durability_flights
    finally:
        monkeypatch.undo()
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpointed", (False, True))
async def test_checkpoint_payload_rejects_foreign_owner_and_trace_envelope(tmp_path, checkpointed):
    state = Persistence(PersistenceStorage.sqlite(tmp_path / "payload.db"))
    await state.initialize(namespace="checkpoint-payload", tenant_id="tenant")
    try:
        store = state.run_store
        execution = replace(_execution(), revision=1, producer_generation=1, producer_claim_id="owner-claim", restart_checkpoints=checkpointed)
        repository = state.execution.executions
        await repository.create_execution(execution)
        store.bind_execution_producer(
            "run", execution_id="execution", producer_generation=1,
            conversation=True, restart_checkpoints=checkpointed, durable_trace=True,
        )
        await store.register_agent_run(AgentRunRecord("run"), execution_id="execution")
        native_messages = [ModelRequest(parts=[UserPromptPart("native context")], run_id="run")]
        if checkpointed:
            store.bind_step_store(native_checkpoint().store)
            assert await store.capture_agent_run_checkpoint_payload(
                "run", execution_id="execution", producer_generation=1,
            ) is None
            await seed_native_snapshot(store.step_store, run_id="run", messages=native_messages)
        with pytest.raises(AIError) as ownership:
            await store.capture_agent_run_checkpoint_payload("run", execution_id="foreign", producer_generation=1)
        assert ownership.value.code is ErrorCode.STORAGE_CONFLICT
        content = await store.capture_agent_run_checkpoint_payload("run", execution_id="execution", producer_generation=1)
        for run_id, generation in (("foreign", 1), ("run", 2)):
            with pytest.raises(AIError) as invalid:
                await store.decode_agent_run_checkpoint_payload(content, agent_run_id=run_id, producer_generation=generation)
            assert invalid.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        _, projection = _capture(store._staging)
        context_content = await store.capture_agent_run_checkpoint_payload(
            "run", execution_id="execution", producer_generation=1,
        )
        if checkpointed:
            assert context_content == content
            snapshot = await store.step_store.latest_snapshot(run_id="run")
            assert json.loads(context_content) == {
                "format": "step-snapshot-reference", "run_id": "run", "producer_generation": 1,
                "snapshot_key": snapshot.idempotency_key, "snapshot_state": "complete",
            }
            restored_run, checkpoint = await store.decode_agent_run_checkpoint_payload(
                context_content, agent_run_id="run", producer_generation=1,
            )
            assert restored_run.agent_run_id == "run" and checkpoint is not None
            assert checkpoint.messages == []
            assert encode_model_messages(checkpoint.context_messages) == encode_model_messages(native_messages)
        else:
            context = decode_projection_payload(context_content)
            assert context.checkpoints and not context.events and not context.interactions
            assert not context.payloads and context.transcript is None and not context.local_messages
        with pytest.raises(AIError) as trace_envelope:
            await store.decode_agent_run_checkpoint_payload(
                encode_projection_payload(replace(projection, producer_generation=1)),
                agent_run_id="run", producer_generation=1,
            )
        assert trace_envelope.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()
