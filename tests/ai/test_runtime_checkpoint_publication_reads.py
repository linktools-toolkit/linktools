#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpoint publication batches reads without weakening its producer fence."""

from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from sqlalchemy import update

from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain
from linktools.ai.persistence._sql import _SqlGroupTransaction
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint, AgentRunRecord
from .test_checkpoint_write_batches import _snapshot
from .test_incremental_projection_boundaries import _storage
from .test_step_projection import _execution

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def _producer(tmp_path, backend, *, checkpointed=True, unresolved=False):
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="checkpoint-publication", tenant_id="tenant")
    try:
        repository = state.execution.executions
        execution = replace(_execution(), revision=1)
        await repository.create_with_history_head(execution)
        await repository.state_store.mutate(
            lambda tx: repository.admit_history_producer_in_transaction(tx, execution)
        )
        store = state.run_store
        store.bind_execution_producer(
            "run", execution_id="execution", producer_generation=1,
            conversation=False, restart_checkpoints=checkpointed, durable_trace=False,
        )
        await store.register_agent_run(AgentRunRecord("run"), execution_id="execution")
        yield state, store, repository, execution
    finally:
        if unresolved:
            with pytest.raises(AIError) as error:
                await state.close()
            assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        else:
            await state.close()


def _checkpoint(count):
    return AgentRunCheckpoint("run", count, [
        ModelRequest(parts=[UserPromptPart(str(index), timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))], timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc)) for index in range(count)
    ], transcript_message_count_before=0)


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("scalar", (False, True))
async def test_publication_preserves_scalar_group_fallback_and_replay(tmp_path, monkeypatch, backend, scalar):
    async with _producer(tmp_path, backend) as (state, store, repository, _execution):
        archive = store.read_store(PersistenceDomain.CHECKPOINT)
        storage_group = repository.state_store.storage_group
        original = storage_group.mutate
        reads = []

        class ScalarGroup:
            def __init__(self, group):
                self.group = group

            def transaction(self, member):
                return self.group.transaction(member)

        class ReadingGroup(ScalarGroup):
            async def get_records(self, requests):
                reads.append(tuple(requests))
                return await self.group.get_records(requests)

        async def mutate(stores, callback):
            return await original(stores, lambda group: callback(
                ScalarGroup(group) if scalar else ReadingGroup(group)
            ))

        with monkeypatch.context() as patch:
            patch.setattr(storage_group, "mutate", mutate)
            checkpoint = _checkpoint(1)
            await store.save_checkpoint(checkpoint, execution_id="execution", producer_generation=1)
            before = await _snapshot(archive.state_store)
            await store.save_checkpoint(checkpoint, execution_id="execution", producer_generation=1)
        assert await _snapshot(archive.state_store) == before
        assert await archive.checkpoint_count(agent_run_id="run") == 1
        assert (await archive.latest_checkpoint(agent_run_id="run")).messages == checkpoint.messages
        assert bool(reads) is (not scalar and backend == "sqlite")


@pytest.mark.parametrize("condition", ("cancel", "takeover", "no_checkpoint"))
async def test_publication_prefetch_follows_producer_fence(tmp_path, monkeypatch, condition):
    async with _producer(tmp_path, "sqlite", checkpointed=condition != "no_checkpoint") as (state, store, repository, execution):
        reads = []
        original = _SqlGroupTransaction.get_records

        async def read(group, requests):
            reads.append(requests)
            return await original(group, requests)

        if condition == "cancel":
            await repository.compare_and_swap("execution", tenant_id="tenant", expected_revision=1,
                next_record=replace(execution, revision=2, status=ExecutionStatus.CANCELLING))
        elif condition == "takeover":
            current = await repository.compare_and_swap("execution", tenant_id="tenant", expected_revision=1,
                next_record=replace(execution, revision=2))
            await repository.state_store.mutate(lambda tx: repository.admit_history_producer_in_transaction(tx, current))
        with monkeypatch.context() as patch:
            patch.setattr(_SqlGroupTransaction, "get_records", read)
            if condition == "no_checkpoint":
                await store.save_checkpoint(_checkpoint(1), execution_id="execution", producer_generation=1)
            else:
                with pytest.raises(AIError) as raised:
                    await store.save_checkpoint(_checkpoint(1), execution_id="execution", producer_generation=1)
                assert raised.value.code is (ErrorCode.EXECUTION_CANCELLED if condition == "cancel" else ErrorCode.STORAGE_CONFLICT)
        assert reads == []


@pytest.mark.parametrize("target", ("agent_run", "transcript_head", "context_projection", "failure"))
async def test_prefetched_owner_conflict_rolls_back_complete_checkpoint(tmp_path, monkeypatch, target):
    async with _producer(tmp_path, "sqlite") as (state, store, repository, _execution):
        archive = store.read_store(PersistenceDomain.CHECKPOINT)
        await store.save_checkpoint(_checkpoint(1), execution_id="execution", producer_generation=1)
        content = await archive.transcript_repository.content_archive()
        members = (repository.state_store, archive.state_store, content.store)
        before = [await _snapshot(member) for member in members]
        original = _SqlGroupTransaction.get_records
        injections = []

        async def read(group, requests):
            values = await original(group, requests)
            if archive.state_store in requests and not injections:
                injections.append(target)
                if target == "failure":
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                record = next(record for record in values[archive.state_store].values() if record.kind == target)
                tx = group.transaction(archive.state_store)
                table = tx._table("ai_state_records")
                await tx._execute(update(table).where(
                    table.c.store_digest == tx._store_hex,
                    table.c.key_digest == record.key_digest.hex(),
                ).values(storage_revision=record.storage_revision + 1))
            return values

        with monkeypatch.context() as patch:
            patch.setattr(_SqlGroupTransaction, "get_records", read)
            with pytest.raises(AIError) as raised:
                await store.save_checkpoint(_checkpoint(2), execution_id="execution", producer_generation=1)
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert injections == [target]
        assert [await _snapshot(member) for member in members] == before


async def test_confirmed_preparation_reuses_metadata_until_commit(tmp_path, monkeypatch):
    from sqlalchemy import event
    from .test_checkpoint_metadata_snapshot_io import _assert_prepared_values_equal

    async with _producer(tmp_path, "sqlite") as (state, store, _repository, _execution):
        archive = store.read_store(PersistenceDomain.CHECKPOINT)
        await store.save_checkpoint(_checkpoint(1), execution_id="execution", producer_generation=1)
        run = await store.get_agent_run(agent_run_id="run")
        receipt = archive._checkpoint_prefix_receipts["run"]
        assert receipt.records is not None
        engine = archive.state_store.storage_group.context.engine.sync_engine
        statements = []

        def capture(_connection, _cursor, sql, _parameters, _context, _many):
            if sql.startswith("SELECT"):
                statements.append(sql)

        event.listen(engine, "before_cursor_execute", capture)
        try:
            checkpoint = _checkpoint(2)
            relocated, hot = await archive.prepare_run_checkpoint(run, checkpoint)
            hot_count = len(statements)
            statements.clear()
            archive._checkpoint_prefix_receipts["run"] = replace(receipt, records=None)
            _, observed = await archive.prepare_run_checkpoint(run, checkpoint)
            assert len(statements) == hot_count + 2
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        assert hot.checkpoints[0].preparation_receipt is receipt
        assert observed.checkpoints[0].preparation_receipt is None
        await _assert_prepared_values_equal(archive, hot, observed)
        assert relocated.transcript_message_count_before == 1


@pytest.mark.parametrize("changed", ("append", "projection", "checkpoint"))
@pytest.mark.parametrize("with_chunks", (False, True))
async def test_stale_preparation_rejects_all_changed_baselines(tmp_path, changed, with_chunks):
    async with _producer(tmp_path, "sqlite") as (state, store, _repository, _execution):
        archive = store.read_store(PersistenceDomain.CHECKPOINT)
        await store.save_checkpoint(_checkpoint(1), execution_id="execution", producer_generation=1)
        run = await store.get_agent_run(agent_run_id="run")
        target = _checkpoint(2 if with_chunks else 1)
        _, batch = await archive.prepare_run_checkpoint(run, target)
        assert batch.checkpoints[0].preparation_receipt is not None
        other = _checkpoint(2 if changed == "append" else 1)
        if changed == "projection":
            other = replace(other, context_messages=_checkpoint(2).messages)
        elif changed == "checkpoint":
            other = replace(other, state="interrupted")
        await archive.materialize_checkpoint(run, other)
        before = await _snapshot(archive.state_store)
        with pytest.raises(AIError) as raised:
            await archive.state_store.mutate(lambda tx: archive.materialize_checkpoint_in_transaction(
                tx, run, batch.checkpoints[0],
            ))
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert await _snapshot(archive.state_store) == before
        assert "run" not in archive._checkpoint_prefix_receipts


@pytest.mark.parametrize("with_chunks", (False, True))
async def test_cached_preparation_materialization_is_idempotent(tmp_path, with_chunks):
    async with _producer(tmp_path, "sqlite") as (state, store, _repository, _execution):
        archive = store.read_store(PersistenceDomain.CHECKPOINT)
        await store.save_checkpoint(_checkpoint(1), execution_id="execution", producer_generation=1)
        run = await store.get_agent_run(agent_run_id="run")
        target = _checkpoint(2 if with_chunks else 1)
        _, batch = await archive.prepare_run_checkpoint(run, target)
        prepared = batch.checkpoints[0]
        assert prepared.preparation_receipt is not None

        async def materialize(tx):
            await archive.materialize_checkpoint_in_transaction(tx, run, prepared)

        await archive.state_store.mutate(materialize)
        committed = await _snapshot(archive.state_store)
        await archive.state_store.mutate(materialize)
        assert await _snapshot(archive.state_store) == committed
        await archive.materialize_checkpoint(run, replace(target, state="interrupted"))
        changed = await _snapshot(archive.state_store)
        with pytest.raises(AIError) as conflict:
            await archive.state_store.mutate(materialize)
        assert conflict.value.code is ErrorCode.STORAGE_CONFLICT
        assert await _snapshot(archive.state_store) == changed


async def test_lost_commit_reply_does_not_confirm_guessed_metadata(tmp_path, monkeypatch):
    async with _producer(tmp_path, "sqlite") as (state, store, repository, _execution):
        group = repository.state_store.storage_group
        original = group.mutate
        lost = []

        async def mutate(stores, callback):
            result = await original(stores, callback)
            if "_save_checkpoint" in callback.__qualname__ and not lost:
                lost.append(True)
                raise OSError("lost checkpoint acknowledgement")
            return result

        with monkeypatch.context() as patch:
            patch.setattr(group, "mutate", mutate)
            await store.save_checkpoint(_checkpoint(1), execution_id="execution", producer_generation=1)
        archive = store.read_store(PersistenceDomain.CHECKPOINT)
        assert lost == [True]
        assert await archive.checkpoint_count(agent_run_id="run") == 1
        assert archive._checkpoint_prefix_receipts["run"].records is None


async def test_custom_prefix_confirmation_keeps_its_existing_signature(tmp_path):
    from linktools.ai.persistence._model_interaction_store import ModelInteractionStateStepArchive

    async with _producer(tmp_path, "sqlite") as (state, store, _repository, _execution):
        original = store.read_store(PersistenceDomain.CHECKPOINT)
        calls = []

        class CustomArchive(ModelInteractionStateStepArchive):
            def confirm_checkpoint_prefix(self, checkpoint, prepared):
                calls.append(checkpoint.step_index)
                super().confirm_checkpoint_prefix(checkpoint, prepared)

        custom = CustomArchive(original.state_store, object_store=None,
            namespace="checkpoint-publication", tenant_id="tenant", runtime_domain=PersistenceDomain.CHECKPOINT)
        store._archives[PersistenceDomain.CHECKPOINT] = custom
        try:
            await store.save_checkpoint(_checkpoint(1), execution_id="execution", producer_generation=1)
            assert calls == [1]
            assert custom._checkpoint_prefix_receipts["run"].records is None
        finally:
            store._archives[PersistenceDomain.CHECKPOINT] = original
            await custom.close()


@pytest.mark.parametrize("outcome", ("cancelled", "unknown"))
async def test_non_normal_commit_cannot_authorize_a_stale_cached_baseline(tmp_path, monkeypatch, outcome):
    import asyncio

    async with _producer(tmp_path, "sqlite", unresolved=outcome == "unknown") as (state, store, repository, _execution):
        archive = store.read_store(PersistenceDomain.CHECKPOINT)
        await store.save_checkpoint(_checkpoint(1), execution_id="execution", producer_generation=1)
        receipt = archive._checkpoint_prefix_receipts["run"]
        assert receipt.records is not None
        run = await store.get_agent_run(agent_run_id="run")
        _, stale = await archive.prepare_run_checkpoint(run, _checkpoint(1))
        group = repository.state_store.storage_group
        original = group.mutate
        verify = archive.verify_checkpoint_range
        committed, release = asyncio.Event(), asyncio.Event()
        injected = []

        async def mutate(stores, callback):
            result = await original(stores, callback)
            if "_save_checkpoint" in callback.__qualname__ and not injected:
                injected.append(True)
                if outcome == "cancelled":
                    committed.set()
                    await release.wait()
                else:
                    raise OSError("checkpoint reply lost")
            return result

        async def unknown_readback(*args, **kwargs):
            if injected:
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return await verify(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(group, "mutate", mutate)
            if outcome == "unknown":
                patch.setattr(archive, "verify_checkpoint_range", unknown_readback)
                with pytest.raises(AIError) as raised:
                    await store.save_checkpoint(_checkpoint(2), execution_id="execution", producer_generation=1)
                assert raised.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
            else:
                task = asyncio.create_task(store.save_checkpoint(
                    _checkpoint(2), execution_id="execution", producer_generation=1,
                ))
                try:
                    await asyncio.wait_for(committed.wait(), 5)
                    task.cancel()
                    release.set()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                finally:
                    release.set()
                    await asyncio.gather(task, return_exceptions=True)
        assert injected == [True]
        if outcome == "unknown":
            assert archive._checkpoint_prefix_receipts["run"] is receipt
            with pytest.raises(AIError) as fenced:
                await store.save_checkpoint(_checkpoint(3), execution_id="execution", producer_generation=1)
            assert fenced.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        else:
            confirmed = archive._checkpoint_prefix_receipts["run"]
            assert confirmed.checkpoint.step_index == 2 and confirmed.records is not None
            before = await _snapshot(archive.state_store)
            with pytest.raises(AIError) as conflict:
                await archive.state_store.mutate(lambda tx: archive.materialize_checkpoint_in_transaction(
                    tx, run, stale.checkpoints[0],
                ))
            assert conflict.value.code is ErrorCode.STORAGE_CONFLICT
            assert await _snapshot(archive.state_store) == before


async def test_checkpoint_readback_uses_actual_committed_sequence(tmp_path, monkeypatch):
    async with _producer(tmp_path, "sqlite") as (state, store, repository, _execution):
        archive = store.read_store(PersistenceDomain.CHECKPOINT)
        prepare = store.checkpoint_persistence.prepare_checkpoint
        mutate = repository.state_store.storage_group.mutate
        verify = archive.verify_checkpoint_range
        verified = []
        inserted = []
        target = _checkpoint(0)

        async def prepare_then_append(*args, **kwargs):
            result = await prepare(*args, **kwargs)
            if not inserted:
                inserted.append(True)
                await archive.materialize_checkpoint(args[1], replace(target, state="interrupted"))
            return result

        async def lose_reply(stores, callback):
            result = await mutate(stores, callback)
            if "_save_checkpoint" in callback.__qualname__:
                raise OSError("lost commit reply after sequence changed")
            return result

        async def track_verification(**kwargs):
            verified.append(kwargs["after_sequence"])
            return await verify(**kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(store.checkpoint_persistence, "prepare_checkpoint", prepare_then_append)
            patch.setattr(repository.state_store.storage_group, "mutate", lose_reply)
            patch.setattr(archive, "verify_checkpoint_range", track_verification)
            await store.save_checkpoint(target, execution_id="execution", producer_generation=1)
        assert verified == [1]
        assert await archive.checkpoint_count(agent_run_id="run") == 2
        assert (await archive.latest_checkpoint(agent_run_id="run")).state == "complete"


async def test_checkpoint_readback_keeps_exact_deduplicated_fact_position(tmp_path, monkeypatch):
    async with _producer(tmp_path, "sqlite") as (state, store, repository, _execution):
        archive = store.read_store(PersistenceDomain.CHECKPOINT)
        run = await store.get_agent_run(agent_run_id="run")
        target = _checkpoint(0)
        await archive.materialize_checkpoint(run, target)
        other = (await archive.prepare_checkpoints(run, (replace(target, state="interrupted"),))).checkpoints[0]
        materialize = archive.materialize_checkpoint_in_transaction
        mutate = repository.state_store.storage_group.mutate
        verify = archive.verify_checkpoint_range
        verified = []

        async def materialize_then_append(tx, owner, prepared, **kwargs):
            position = await materialize(tx, owner, prepared, **kwargs)
            await materialize(tx, owner, other)
            return position

        async def lose_reply(stores, callback):
            result = await mutate(stores, callback)
            if "_save_checkpoint" in callback.__qualname__:
                raise OSError("lost deduplicated commit reply")
            return result

        async def track_verification(**kwargs):
            verified.append(kwargs["after_sequence"])
            return await verify(**kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(archive, "materialize_checkpoint_in_transaction", materialize_then_append)
            patch.setattr(repository.state_store.storage_group, "mutate", lose_reply)
            patch.setattr(archive, "verify_checkpoint_range", track_verification)
            await store.save_checkpoint(target, execution_id="execution", producer_generation=1)
        assert verified == [0]
        assert await archive.checkpoint_count(agent_run_id="run") == 2


async def test_legacy_custom_materialization_keeps_prepared_readback_position(tmp_path, monkeypatch):
    from linktools.ai.persistence._model_interaction_store import ModelInteractionStateStepArchive

    async with _producer(tmp_path, "sqlite") as (state, store, repository, _execution):
        original = store.read_store(PersistenceDomain.CHECKPOINT)
        positions = []
        proofs = []

        class CustomArchive(ModelInteractionStateStepArchive):
            async def capture_checkpoint_progress(self, **kwargs):
                progress = await super().capture_checkpoint_progress(**kwargs)
                positions.append(progress[0])
                return progress

            async def materialize_checkpoint_in_transaction(self, tx, run, checkpoint, **kwargs):
                await super().materialize_checkpoint_in_transaction(tx, run, checkpoint, **kwargs)
                await super().materialize_checkpoint_in_transaction(tx, run, self.following)
                # Existing extensions need not adopt the new optional result.

            async def verify_checkpoint_range(self, **kwargs):
                proofs.append(kwargs["after_sequence"])
                return await super().verify_checkpoint_range(**kwargs)

        custom = CustomArchive(original.state_store, object_store=None, namespace="checkpoint-publication",
                               tenant_id="tenant", runtime_domain=PersistenceDomain.CHECKPOINT)
        store._archives[PersistenceDomain.CHECKPOINT] = custom
        try:
            run = await store.get_agent_run(agent_run_id="run")
            target = _checkpoint(0)
            custom.following = (await custom.prepare_checkpoints(run, (replace(target, state="interrupted"),))).checkpoints[0]
            mutate = repository.state_store.storage_group.mutate

            async def lose_reply(stores, callback):
                result = await mutate(stores, callback)
                if "_save_checkpoint" in callback.__qualname__:
                    raise OSError("custom commit acknowledgement lost")
                return result

            with monkeypatch.context() as patch:
                patch.setattr(repository.state_store.storage_group, "mutate", lose_reply)
                await store.save_checkpoint(target, execution_id="execution", producer_generation=1)
            assert positions == [1]
            assert proofs == [0]
            assert await custom.checkpoint_count(agent_run_id="run") == 2
        finally:
            store._archives[PersistenceDomain.CHECKPOINT] = original
            await custom.close()
