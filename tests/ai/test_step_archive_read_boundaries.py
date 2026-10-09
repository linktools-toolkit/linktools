#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Projection writes reuse immutable owners and read coherent history heads."""

from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state import RuntimeDomain, _step_archive
from linktools.ai.runtime.state._codec import decode_envelope, encode_envelope
from linktools.ai.runtime.state._memory_transaction import _MemoryTransaction
from linktools.ai.runtime.state._step_archive import PreparedExecutionProjection
from linktools.ai.runtime.state._step_contracts import AgentRunCheckpoint, AgentRunRecord, StepEvent

from .test_incremental_projection_boundaries import _storage
from .test_step_projection import _execution


@asynccontextmanager
async def _archive(tmp_path: Path, backend: str = "memory"):
    state = _storage(backend, tmp_path)
    await state.initialize(namespace="archive-boundaries", tenant_id="tenant")
    try:
        execution = replace(_execution(), revision=1)
        repository = state.execution.executions
        await repository.create_with_history_head(execution)
        await repository.state_store.mutate(
            lambda tx: repository.admit_history_producer_in_transaction(tx, execution)
        )
        archive = state.run_store.read_store(RuntimeDomain.EXECUTION)
        yield state, archive, AgentRunRecord("run", metadata={"context": "x" * 16_384})
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("facts", ("none", "event", "checkpoint"))
async def test_observation_commit_bounds_immutable_owner_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, facts: str,
) -> None:
    async with _archive(tmp_path) as (_state, archive, run):
        await archive.register_agent_run(run, execution_id="execution")
        first = ModelResponse(parts=[TextPart("observed")])
        second = ModelResponse(parts=[TextPart("checkpoint suffix")])
        observation = await archive.transcript_repository.prepare_observation(
            "run", (first,), first_message_index=0, pending=None, pending_keys=(),
        )
        events = (StepEvent("run", "MODEL_REQUEST_SUCCEEDED", 1),) if facts == "event" else ()
        checkpoints = ()
        if facts == "checkpoint":
            batch = await archive.prepare_checkpoints(
                run, (AgentRunCheckpoint(
                    "run", 1, [first, second], transcript_message_count_before=0,
                ),), observed_message_count=1,
            )
            checkpoints = batch.checkpoints

        owner_key = archive._agent_run_key("run")
        owner_reads = 0
        owner_decodes = 0
        original_get = _MemoryTransaction.get_record
        original_batch = _MemoryTransaction.get_records
        original_decode = _step_archive._decode_step

        async def read_one(self, key):
            nonlocal owner_reads
            owner_reads += key == owner_key
            return await original_get(self, key)

        async def read_many(self, keys):
            nonlocal owner_reads
            owner_reads += sum(key == owner_key for key in keys)
            return await original_batch(self, keys)

        def decode(data):
            nonlocal owner_decodes
            value = original_decode(data)
            owner_decodes += isinstance(value, AgentRunRecord)
            return value

        with monkeypatch.context() as patch:
            patch.setattr(_MemoryTransaction, "get_record", read_one)
            patch.setattr(_MemoryTransaction, "get_records", read_many)
            patch.setattr(_step_archive, "_decode_step", decode)
            await archive.sync_prepared_projection(
                run, events=events, checkpoints=checkpoints, observation=observation,
                execution_id="execution", producer_generation=1,
            )
        assert owner_reads <= 2, "one projection must not repeatedly fetch its immutable owner"
        assert owner_decodes == 1, "one transaction must not repeatedly decode large run metadata"
        expected = [first, second] if facts == "checkpoint" else [first]
        assert [value async for value in archive.iter_messages(agent_run_id="run")] == expected
        assert await archive.list_events(agent_run_id="run") == list(events)
        if facts == "checkpoint":
            checkpoint = await archive.latest_checkpoint(agent_run_id="run")
            assert checkpoint is not None and checkpoint.messages == expected


@pytest.mark.asyncio
async def test_empty_projection_advances_history_only_when_admitting_a_run(tmp_path: Path) -> None:
    async with _archive(tmp_path) as (state, archive, run):
        repository = state.execution.executions
        before = await repository.get_history_head("execution", tenant_id="tenant")
        await archive.sync_prepared_projection(
            run, events=(), checkpoints=(), execution_id="execution", producer_generation=1,
        )
        admitted = await repository.get_history_head("execution", tenant_id="tenant")
        assert admitted.revision == before.revision + 1
        assert await archive.get_agent_run(agent_run_id="run") == run
        await archive.sync_prepared_projection(
            run, events=(), checkpoints=(), execution_id="execution", producer_generation=1,
        )
        assert await repository.get_history_head("execution", tenant_id="tenant") == admitted


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("failure", ("run", "head", "stale", "generation"))
async def test_observation_commit_keeps_validation_and_fencing(
    tmp_path: Path, backend: str, failure: str,
) -> None:
    async with _archive(tmp_path, backend) as (state, archive, run):
        await archive.register_agent_run(run, execution_id="execution")
        message = ModelResponse(parts=[TextPart("observed")])
        observation = await archive.transcript_repository.prepare_observation(
            "run", (message,), first_message_index=0, pending=None, pending_keys=(),
        )
        expected_error = ErrorCode.STORAGE_CONFLICT
        if failure in {"run", "head"}:
            key = (archive._agent_run_key("run") if failure == "run"
                   else archive.transcript_repository.head_key("run"))

            async def corrupt(transaction) -> None:
                record = await transaction.get_record(key)
                envelope = dict(decode_envelope(record.data).value)
                payload = dict(envelope["payload"])
                fields = dict(payload["fields"])
                fields["metadata" if failure == "run" else "message_count"] = False
                payload["fields"] = fields
                envelope["payload"] = payload
                assert await transaction.replace_record(
                    replace(record, data=encode_envelope(envelope), storage_version=record.storage_version + 1),
                    expected_storage_version=record.storage_version,
                )

            await archive.state_store.mutate(corrupt)
            expected_error = ErrorCode.STORAGE_INTEGRITY_ERROR
        elif failure == "stale":
            await archive.sync_prepared_projection(
                run, events=(), checkpoints=(), observation=observation,
                execution_id="execution", producer_generation=1,
            )
        else:
            repository = state.execution.executions
            await repository.state_store.mutate(
                lambda tx: repository.admit_history_producer_in_transaction(
                    tx, replace(_execution(), revision=2),
                )
            )
        with pytest.raises(AIError) as raised:
            await archive.sync_prepared_projection(
                run, events=(StepEvent("run", "MODEL_REQUEST_SUCCEEDED", 1),),
                checkpoints=(), observation=observation,
                execution_id="execution", producer_generation=1,
            )
        assert raised.value.code is expected_error
        assert await archive.list_events(agent_run_id="run") == []
        if failure in {"stale", "generation"}:
            assert [value async for value in archive.iter_messages(agent_run_id="run")] == (
                [message] if failure == "stale" else []
            )


@pytest.mark.asyncio
async def test_projection_verification_does_not_combine_different_history_captures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _archive(tmp_path) as (_state, archive, run):
        await archive.register_agent_run(run, execution_id="execution")
        original = archive.execution_history_heads
        captures = 0

        async def advancing_heads(agent_run_ids):
            nonlocal captures
            captured = await original(agent_run_ids)
            advanced = captures > 0
            captures += 1
            return {
                key: replace(value, event_count=int(advanced), interaction_count=int(advanced))
                for key, value in captured.items()
            }

        monkeypatch.setattr(archive, "execution_history_heads", advancing_heads)
        projection = PreparedExecutionProjection(
            run, (), (), 0, 0, 0, 0, 0, target_interaction_offset=1,
        )
        assert not await archive.verify_execution_projection_head(projection)


@pytest.mark.asyncio
async def test_projection_verification_reads_each_context_projection_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _archive(tmp_path) as (_state, archive, run):
        message = ModelResponse(parts=[TextPart("checkpoint")])
        batch = await archive.prepare_checkpoints(
            run, (AgentRunCheckpoint("run", 1, [message], transcript_message_count_before=0),),
        )
        await archive.sync_prepared_projection(
            run, events=(), checkpoints=batch.checkpoints,
            execution_id="execution", producer_generation=1,
        )
        head = await archive.execution_history_head_record("run")
        projection = PreparedExecutionProjection(
            run, (), (), 0, 0, head.event_count, head.checkpoint_count,
            head.transcript_message_count, durable_projection_digest=head.projection_digest,
        )
        projection_key = archive.transcript_repository.projection_key("run")
        reads = 0
        original = _MemoryTransaction.get_records

        async def read_many(self, keys):
            nonlocal reads
            reads += sum(key == projection_key for key in keys)
            return await original(self, keys)

        monkeypatch.setattr(_MemoryTransaction, "get_records", read_many)
        assert await archive.verify_execution_projection_head(projection)
        assert reads == 1, "verification must use one coherent context projection capture"
