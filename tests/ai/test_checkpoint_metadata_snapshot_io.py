#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpoint metadata snapshots keep fresh preparation and commit fences."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from sqlalchemy import event

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeDomain
from linktools.ai.runtime.state import RuntimeState
from linktools.ai.runtime.state._archive import PreparedArchive
from linktools.ai.runtime.state._step_archive import PreparedAgentRunCheckpointBatch, StateStepArchive
from linktools.ai.runtime.state._sql import _SqlTransaction
from linktools.ai.runtime.state._step_contracts import AgentRunCheckpoint, AgentRunRecord
from linktools.ai.runtime.state._store import active_state_scope

from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio


def _checkpoint(count: int) -> AgentRunCheckpoint:
    return AgentRunCheckpoint(
        "run", count,
        [ModelRequest(parts=[UserPromptPart(
            str(index), timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )]) for index in range(count)],
        transcript_message_count_before=0, timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


async def _assert_prepared_values_equal(
    archive: StateStepArchive,
    left: PreparedAgentRunCheckpointBatch,
    right: PreparedAgentRunCheckpointBatch,
) -> None:
    # Pending bundles may repeat an already durable immutable node. Compare the
    # resulting record values as well as the roots and checkpoint facts.
    def facts(batch: PreparedAgentRunCheckpointBatch) -> PreparedAgentRunCheckpointBatch:
        return replace(batch, checkpoints=tuple(
            replace(checkpoint, archive=PreparedArchive()) for checkpoint in batch.checkpoints
        ))

    assert facts(left) == facts(right)
    content = await archive.transcript_repository.content_archive()
    pending = [
        {key: record for checkpoint in batch.checkpoints
         for key, record in checkpoint.archive.records.items()}
        for batch in (left, right)
    ]
    durable = await content.store.read_records(tuple(pending[0].keys() | pending[1].keys()))
    resolved = [{**durable, **records} for records in pending]
    assert resolved[0].keys() == resolved[1].keys()
    for key in resolved[0]:
        # Physical CAS revisions are not part of immutable content semantics.
        assert replace(resolved[0][key], storage_revision=0) == replace(resolved[1][key], storage_revision=0)


async def test_checkpoint_metadata_batches_reads_without_changing_prepared_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeState(_storage("sqlite", tmp_path))
    await state.initialize(namespace="checkpoint-metadata", tenant_id="tenant")
    archive = state.run_store.read_store(RuntimeDomain.CHECKPOINT)
    run = AgentRunRecord("run")
    engine = archive.state_store.storage_group.context.engine.sync_engine
    statements: list[str] = []
    select_stores: list[str] = []

    def capture(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.split()[0].upper())
        if statement.startswith("SELECT"):
            select_stores.append(_parameters[0])

    async def without_prefetch(transaction, keys):
        records = [await transaction.get_record(key) for key in keys]
        return {record.key_digest: record for record in records if record is not None}

    load = archive.transcript_repository.load_message_span

    async def without_observed_head(*args, **kwargs):
        kwargs.pop("observed_head", None)
        return await load(*args, **kwargs)

    # Resolve the shared scope once so neither side pays a one-time cold lookup.
    await archive.transcript_repository.content_archive(create=True)
    try:
        results, counts, domain_counts = [], [], []
        for baseline in (True, False):
            statements.clear()
            select_stores.clear()
            with monkeypatch.context() as patch:
                if baseline:
                    patch.setattr(_SqlTransaction, "get_records", without_prefetch)
                event.listen(engine, "before_cursor_execute", capture)
                try:
                    results.append(await archive.prepare_checkpoints(run, (_checkpoint(1),)))
                finally:
                    event.remove(engine, "before_cursor_execute", capture)
            assert not set(statements) & {"INSERT", "UPDATE", "DELETE"}
            counts.append(statements.count("SELECT"))
            domain_counts.append(select_stores.count(archive.state_store.store_digest.hex()))
        await _assert_prepared_values_equal(archive, results[0], results[1])
        # Three domain metadata keys plus one content lookup, batched to two reads.
        assert domain_counts == [3, 1]
        assert counts == [4, 2]

        await archive.materialize_checkpoint(run, _checkpoint(1))
        results, counts, domain_counts = [], [], []
        for baseline in (True, False):
            statements.clear()
            select_stores.clear()
            with monkeypatch.context() as patch:
                if baseline:
                    patch.setattr(_SqlTransaction, "get_records", without_prefetch)
                    patch.setattr(archive.transcript_repository, "load_message_span", without_observed_head)
                event.listen(engine, "before_cursor_execute", capture)
                try:
                    results.append(await archive.relocate_run_checkpoint(run, _checkpoint(2)))
                finally:
                    event.remove(engine, "before_cursor_execute", capture)
            assert not set(statements) & {"INSERT", "UPDATE", "DELETE"}
            counts.append(statements.count("SELECT"))
            domain_counts.append(select_stores.count(archive.state_store.store_digest.hex()))
        assert results[0] == results[1]
        # Run/head metadata, chunk facts, immutable nodes and content; the
        # unbatched path also rereads the head discarded by its wrapper.
        assert domain_counts == [4, 2]
        assert counts == [6, 4]
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("failure", ("different_run", "missing_head", "short_prefix", "changed_prefix"))
@pytest.mark.parametrize("composed", (False, True))
async def test_checkpoint_relocation_keeps_integrity_failures(
    tmp_path: Path, backend: str, failure: str, composed: bool,
) -> None:
    state = RuntimeState(_storage(backend, tmp_path))
    await state.initialize(namespace="checkpoint-metadata", tenant_id="tenant")
    archive = state.run_store.read_store(RuntimeDomain.CHECKPOINT)
    run = AgentRunRecord("run")
    checkpoint = _checkpoint(1)
    try:
        await archive.materialize_checkpoint(run, checkpoint)
        if failure == "different_run":
            run = replace(run, metadata={"changed": "yes"})
        elif failure == "missing_head":
            await archive.state_store.mutate(
                lambda tx: tx.delete_record(archive.transcript_repository.head_key("run"))
            )
        elif failure == "short_prefix":
            checkpoint = _checkpoint(0)
        else:
            checkpoint = replace(checkpoint, messages=[ModelRequest(parts=[UserPromptPart("different")])])
        with pytest.raises(AIError) as raised:
            if composed:
                await archive.prepare_run_checkpoint(run, checkpoint)
            else:
                await archive.relocate_run_checkpoint(run, checkpoint)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        if failure == "missing_head":
            with pytest.raises(AIError) as raised:
                await archive.prepare_checkpoints(run, (checkpoint,))
            assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert active_state_scope() is None
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("composed", (False, True))
async def test_checkpoint_preparation_rereads_after_body_io_and_commit_rejects_stale_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, composed: bool,
) -> None:
    state = RuntimeState(_storage(backend, tmp_path))
    await state.initialize(namespace="checkpoint-metadata", tenant_id="tenant")
    archive = state.run_store.read_store(RuntimeDomain.CHECKPOINT)
    run = AgentRunRecord("run")
    try:
        await archive.materialize_checkpoint(run, _checkpoint(1))
        load = archive.transcript_repository.load_message_span

        async def append_during_body_read(*args, **kwargs):
            assert active_state_scope() is None
            await archive.materialize_checkpoint(run, _checkpoint(2))
            return await load(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(archive.transcript_repository, "load_message_span", append_during_body_read)
            if composed:
                prepared, = (await archive.prepare_run_checkpoint(run, _checkpoint(3)))[1].checkpoints
            else:
                relocated = await archive.relocate_run_checkpoint(run, _checkpoint(3))
        if not composed:
            assert relocated.transcript_message_count_before == 1
            prepared, = (await archive.prepare_checkpoints(run, (relocated,))).checkpoints
        assert prepared.chunks[0].first_message_index == 2
        await archive.materialize_checkpoint(run, _checkpoint(4))
        with pytest.raises(AIError) as raised:
            await archive.state_store.mutate(
                lambda tx: archive.materialize_checkpoint_in_transaction(tx, run, prepared)
            )
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        latest = await archive.latest_checkpoint(agent_run_id="run")
        assert latest is not None and latest.step_index == 4
        assert len(latest.messages) == 4
        assert active_state_scope() is None
    finally:
        await state.close()


@pytest.mark.parametrize("case", ("recovery_first", "recovery_next", "conversation", "execution"))
async def test_checkpoint_commit_reuses_projection_read_and_keeps_replay_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    from .test_step_projection import _execution

    counts = []
    for baseline in (True, False):
        path = tmp_path / str(baseline)
        path.mkdir()
        state = RuntimeState(_storage("sqlite", path))
        await state.initialize(namespace="checkpoint-commit", tenant_id="tenant")
        domain = (RuntimeDomain.CONVERSATION if case == "conversation" else
                  RuntimeDomain.TRACE if case == "execution" else RuntimeDomain.CHECKPOINT)
        archive = state.run_store.read_store(domain)
        run = AgentRunRecord("run")
        engine = archive.state_store.storage_group.context.engine.sync_engine
        statements = []
        get_records = _SqlTransaction.get_records
        projection_key = archive.transcript_repository.projection_key("run")

        async def without_projection_prefetch(transaction, keys):
            if archive._agent_run_key("run") in keys:
                keys = tuple(key for key in keys if key != projection_key)
            return await get_records(transaction, keys)

        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement.split()[0].upper())

        try:
            if domain is RuntimeDomain.TRACE:
                execution = replace(_execution(), revision=1)
                repository = state.execution.executions
                await repository.create_with_history_head(execution)
                await repository.state_store.mutate(
                    lambda tx: repository.admit_history_producer_in_transaction(tx, execution)
                )
            if case == "recovery_next":
                await archive.materialize_checkpoint(run, _checkpoint(0))
            prepared, = (await archive.prepare_checkpoints(run, (_checkpoint(1),))).checkpoints

            async def commit():
                if domain is RuntimeDomain.TRACE:
                    await archive.sync_prepared_projection(
                        run, events=(), checkpoints=(prepared,), execution_id="execution",
                        producer_generation=1,
                    )
                else:
                    await archive.state_store.mutate(
                        lambda tx: archive.materialize_checkpoint_in_transaction(tx, run, prepared)
                    )

            with monkeypatch.context() as patch:
                if baseline:
                    patch.setattr(_SqlTransaction, "get_records", without_projection_prefetch)
                event.listen(engine, "before_cursor_execute", capture)
                try:
                    await commit()
                finally:
                    event.remove(engine, "before_cursor_execute", capture)
            counts.append(statements.count("SELECT"))
            latest = await archive.latest_checkpoint(agent_run_id="run")
            assert latest is not None and latest.messages == _checkpoint(1).messages
            if domain is not RuntimeDomain.TRACE:
                statements.clear()
                event.listen(engine, "before_cursor_execute", capture)
                try:
                    await commit()
                finally:
                    event.remove(engine, "before_cursor_execute", capture)
                assert not set(statements) & {"INSERT", "UPDATE", "DELETE"}
        finally:
            await state.close()
    assert counts[1] == counts[0] - 1


async def test_checkpoint_projection_stale_version_rolls_back_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeState(_storage("sqlite", tmp_path))
    await state.initialize(namespace="checkpoint-stale-projection", tenant_id="tenant")
    archive = state.run_store.read_store(RuntimeDomain.CHECKPOINT)
    run = AgentRunRecord("run")
    try:
        await archive.materialize_checkpoint(run, _checkpoint(1))
        prepared, = (await archive.prepare_checkpoints(run, (_checkpoint(2),))).checkpoints
        projection_key = archive.transcript_repository.projection_key("run")
        replace_records = _SqlTransaction.replace_records
        injected = []

        async def stale_projection(transaction, replacements):
            projection = next((
                value for value in replacements if value.record.key_digest == projection_key
            ), None)
            if projection is not None and not injected:
                injected.append(True)
                await replace_records(transaction, (projection,))
            return await replace_records(transaction, replacements)

        with monkeypatch.context() as patch:
            patch.setattr(_SqlTransaction, "replace_records", stale_projection)
            with pytest.raises(AIError) as raised:
                await archive.state_store.mutate(
                    lambda tx: archive.materialize_checkpoint_in_transaction(tx, run, prepared)
                )
        assert injected and raised.value.code is ErrorCode.STORAGE_CONFLICT
        latest = await archive.latest_checkpoint(agent_run_id="run")
        assert latest is not None and latest.messages == _checkpoint(1).messages
        assert await archive.checkpoint_count(agent_run_id="run") == 1
    finally:
        await state.close()


async def test_run_admission_does_not_read_context_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeState(_storage("sqlite", tmp_path))
    await state.initialize(namespace="checkpoint-admission", tenant_id="tenant")
    archive = state.run_store.read_store(RuntimeDomain.CHECKPOINT)
    projection_key = archive.transcript_repository.projection_key("run")
    get_record, get_records = _SqlTransaction.get_record, _SqlTransaction.get_records

    async def read_one(transaction, key):
        assert key != projection_key
        return await get_record(transaction, key)

    async def read_many(transaction, keys):
        assert projection_key not in keys
        return await get_records(transaction, keys)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(_SqlTransaction, "get_record", read_one)
            patch.setattr(_SqlTransaction, "get_records", read_many)
            await archive.register_agent_run(AgentRunRecord("run"))
    finally:
        await state.close()
