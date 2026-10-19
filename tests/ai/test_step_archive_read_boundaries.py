#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Projection writes reuse immutable owners and read coherent history heads."""

from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state import RuntimeDomain, RuntimeState, _step_archive
from linktools.ai.runtime.state._codec import decode_envelope, encode_envelope
from linktools.ai.runtime.state._memory_transaction import _MemoryTransaction
from linktools.ai.runtime.state._step_archive import PreparedExecutionProjection
from linktools.ai.runtime.state._step_contracts import AgentRunCheckpoint, AgentRunRecord, StepEvent

from .test_incremental_projection_boundaries import _storage
from .test_step_projection import _execution


@asynccontextmanager
async def _archive(tmp_path: Path, backend: str = "memory"):
    state = RuntimeState(_storage(backend, tmp_path))
    await state.initialize(namespace="archive-boundaries", tenant_id="tenant")
    try:
        execution = replace(_execution(), revision=1)
        repository = state.execution.executions
        await repository.create_with_history_head(execution)
        await repository.state_store.mutate(
            lambda tx: repository.admit_history_producer_in_transaction(tx, execution)
        )
        archive = state.run_store.read_store(RuntimeDomain.TRACE)
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
                    replace(record, data=encode_envelope(envelope), storage_revision=record.storage_revision + 1),
                    expected_storage_revision=record.storage_revision,
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


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_projection_admission_advances_history_once_and_rolls_back_invalid_requests(
    tmp_path: Path, backend: str,
) -> None:
    from .test_model_interaction_records import _running

    async with _archive(tmp_path, backend) as (state, archive, run):
        repository = state.execution.executions
        before = await repository.get_history_head("execution", tenant_id="tenant")
        invalid = replace(_running(), agent_run_id="different")
        with pytest.raises(AIError) as raised:
            await archive.sync_prepared_projection(
                run, events=(), checkpoints=(), interactions=(invalid,),
                execution_id="execution", producer_generation=1,
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await archive.get_agent_run(agent_run_id=run.agent_run_id) is None
        assert await repository.get_history_head("execution", tenant_id="tenant") == before
        assert await archive.sync_prepared_projection(
            run, events=(), checkpoints=(), execution_id="execution", producer_generation=1,
        ) is None
        admitted = await repository.get_history_head("execution", tenant_id="tenant")
        assert admitted.revision == before.revision + 1
        assert await archive.get_agent_run(agent_run_id=run.agent_run_id) == run
        assert await archive.sync_prepared_projection(
            run, events=(), checkpoints=(), execution_id="execution", producer_generation=1,
        ) is None
        assert await repository.get_history_head("execution", tenant_id="tenant") == admitted


@pytest.mark.asyncio
async def test_projection_uses_batched_owner_admission_without_existence_preread(tmp_path: Path) -> None:
    from sqlalchemy import event

    async with _archive(tmp_path, "sqlite") as (_state, archive, run):
        await archive.register_agent_run(run, execution_id="execution")
        engine = archive.state_store.storage_group.context.engine.sync_engine
        counts = []
        for preread in (True, False):
            statements = []
            def capture(_connection, _cursor, statement, _parameters, _context, _many):
                statements.append(statement.lstrip().upper())
            async def project(transaction):
                if preread:
                    assert await transaction.get_record(archive._agent_run_key(run.agent_run_id)) is not None
                await archive.sync_projection_in_transaction(
                    transaction, run, events=(), checkpoints=(), execution_id="execution",
                    producer_generation=1,
                )
            event.listen(engine, "before_cursor_execute", capture)
            try:
                await archive.state_store.storage_group.mutate(
                    archive.transaction_stores,
                    lambda group: project(group.transaction(archive.state_store)),
                )
            finally:
                event.remove(engine, "before_cursor_execute", capture)
            counts.append(sum(sql.startswith("SELECT") for sql in statements))
        assert counts[1] < counts[0]
        print("projection SELECT preread, batched:", counts)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_concurrent_empty_projection_admissions_advance_history_once(tmp_path: Path, backend: str) -> None:
    import asyncio

    async with _archive(tmp_path, backend) as (state, archive, run):
        repository = state.execution.executions
        before = await repository.get_history_head("execution", tenant_id="tenant")
        async def project():
            return await archive.sync_prepared_projection(
                run, events=(), checkpoints=(), execution_id="execution", producer_generation=1,
            )
        results = await asyncio.gather(project(), project(), return_exceptions=True)
        assert any(result is None for result in results)
        for result in results:
            if result is not None:
                assert isinstance(result, AIError) and result.code is ErrorCode.STORAGE_CONFLICT
                assert await project() is None
        after = await repository.get_history_head("execution", tenant_id="tenant")
        assert after.revision == before.revision + 1
        assert await archive.get_agent_run(agent_run_id=run.agent_run_id) == run


@pytest.mark.asyncio
async def test_projection_keeps_custom_private_override_and_public_none_result(tmp_path: Path) -> None:
    from linktools.ai.runtime.state._model_interaction_store import ModelInteractionStateStepArchive

    async with _archive(tmp_path) as (_state, archive, run):
        calls = []
        class CustomArchive(ModelInteractionStateStepArchive):
            async def _sync_projection_in_transaction(self, *args, **kwargs):
                calls.append(True)
                await super()._sync_projection_in_transaction(*args, **kwargs)
                # Existing overrides may discard the private implementation result.
                return None
        recovery = _state.run_store.read_store(RuntimeDomain.CHECKPOINT)
        custom = CustomArchive(
            recovery.state_store, object_store=None, namespace="archive-boundaries",
            tenant_id="tenant", runtime_domain=RuntimeDomain.CHECKPOINT,
        )
        assert await custom.sync_prepared_projection(run, events=(), checkpoints=()) is None
        assert calls == [True]
        assert await custom.get_agent_run(agent_run_id=run.agent_run_id) == run


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_checkpoint_null_boundary_rejects_reads_instead_of_using_current_head(
    tmp_path: Path, backend: str,
) -> None:
    from linktools.ai.runtime.state._codec import _encode_step_envelope
    from linktools.ai.runtime.state._store import FactQuery

    async with _archive(tmp_path, backend) as (_state, archive, run):
        message = ModelResponse(parts=[TextPart("checkpoint")])
        checkpoint = AgentRunCheckpoint("run", 1, [message], transcript_message_count_before=0)
        prepared, = (await archive.prepare_checkpoints(run, (checkpoint,))).checkpoints
        assert prepared.stored.transcript_message_count == 1
        await archive.sync_prepared_projection(
            run, events=(), checkpoints=(prepared,),
            execution_id="execution", producer_generation=1,
        )
        encoded = _encode_step_envelope(prepared.stored)
        encoded["value"]["payload"]["fields"]["transcript_message_count"] = None

        async def corrupt(transaction) -> None:
            facts = await transaction.list_facts(FactQuery(archive._stream("run", "checkpoint")))
            assert len(facts) == 1
            owner = await transaction.get_record(archive._agent_run_key("run"))
            assert owner is not None
            await transaction.guard_record(owner.key_digest, expected_storage_revision=owner.storage_revision)
            await transaction.delete_fact_streams(owner.key_digest)
            await transaction.insert_fact(replace(facts[0], data=encoded))

        await archive.state_store.mutate(corrupt)
        with pytest.raises(AIError) as latest:
            await archive.latest_checkpoint(agent_run_id="run")
        assert latest.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        with pytest.raises(AIError) as verify:
            await archive.verify_checkpoint_projection(agent_run_id="run", checkpoint=checkpoint)
        assert verify.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
