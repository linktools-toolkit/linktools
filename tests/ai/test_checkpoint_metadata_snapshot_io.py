#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace and conversation metadata snapshots preserve preparation and commit fences."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from sqlalchemy import event

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence import Persistence
from linktools.ai.persistence._archive import PreparedArchive
from linktools.ai.persistence._native_run_reference import NativeRunReferences
from linktools.ai.persistence._step_archive import PreparedAgentRunCheckpointBatch, StateStepArchive
from linktools.ai.persistence._sql import _SqlTransaction
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint, AgentRunRecord
from linktools.ai.persistence.state import active_state_scope

from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio


async def _trace_archive(state: Persistence) -> StateStepArchive:
    from .test_step_projection import _execution

    repository = state.execution.executions
    execution = await repository.get("execution", tenant_id="tenant")
    if execution is None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        execution = replace(_execution(), revision=1, producer_generation=1, created_at=now, updated_at=now)
        await repository.create_execution(execution)
    return state.run_store.read_store(PersistenceDomain.TRACE)


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
    for first, second in zip(left.checkpoints, right.checkpoints, strict=True):
        for value in (first.archive, second.archive):
            assert value.owner in (None, (content.scope_id, content._namespace, content._tenant_id))
        assert first.archive.native_messages.keys() == second.archive.native_messages.keys()
        for root, value in first.archive.native_messages.items():
            other = second.archive.native_messages[root]
            assert (value.root, value.frontier, value.messages, value.nodes) == (
                other.root, other.frontier, other.messages, other.nodes,
            )
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
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="checkpoint-metadata", tenant_id="tenant")
    archive = await _trace_archive(state)
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
        return {record.key: record for record in records if record is not None}

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
            domain_counts.append(select_stores.count(archive.state_store.store_key))
        await _assert_prepared_values_equal(archive, results[0], results[1])
        # Native message preparation needs no body existence lookup. The three
        # domain metadata keys still collapse to one grouped read.
        assert domain_counts == [3, 1]
        assert counts == [3, 1]

    finally:
        await state.close()


@pytest.mark.parametrize("case", ("archive_first", "archive_next", "conversation", "execution"))
async def test_checkpoint_commit_reuses_projection_read_and_keeps_replay_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    counts = []
    for baseline in (True, False):
        path = tmp_path / str(baseline)
        path.mkdir()
        state = Persistence(_storage("sqlite", path))
        await state.initialize(namespace="checkpoint-commit", tenant_id="tenant")
        archive = (state.run_store.read_store(PersistenceDomain.CONVERSATION)
                   if case == "conversation" else await _trace_archive(state))
        run = AgentRunRecord("run")
        engine = archive.state_store.storage_group.context.engine.sync_engine
        statements = []
        get_records = _SqlTransaction.get_records
        projection_key = archive.transcript_repository.projection_key("run")
        read_physical = NativeRunReferences.read_physical

        async def without_native_projection_prefetch(references, transaction, keys):
            if archive._agent_run_key("run") in keys:
                keys = tuple(key for key in keys if key != projection_key)
            return await read_physical(references, transaction, keys)

        async def without_projection_prefetch(transaction, keys):
            if archive._agent_run_key("run") in keys:
                keys = tuple(key for key in keys if key != projection_key)
            return await get_records(transaction, keys)

        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement.split()[0].upper())

        try:
            if case == "archive_next":
                await archive.materialize_checkpoint(run, _checkpoint(0), execution_id="execution")
            prepared, = (await archive.prepare_checkpoints(run, (_checkpoint(1),))).checkpoints

            async def commit():
                if case == "execution":
                    await archive.sync_prepared_projection(
                        run, events=(), checkpoints=(prepared,), execution_id="execution",
                        producer_generation=1,
                    )
                else:
                    await archive._mutate(
                        lambda tx: archive.materialize_checkpoint_in_transaction(tx, run, prepared, execution_id="execution")
                    )

            with monkeypatch.context() as patch:
                if baseline:
                    patch.setattr(_SqlTransaction, "get_records", without_projection_prefetch)
                    patch.setattr(NativeRunReferences, "read_physical", without_native_projection_prefetch)
                event.listen(engine, "before_cursor_execute", capture)
                try:
                    await commit()
                finally:
                    event.remove(engine, "before_cursor_execute", capture)
            counts.append(statements.count("SELECT"))
            latest = await archive.latest_checkpoint(agent_run_id="run")
            assert latest is not None and latest.messages == _checkpoint(1).messages
            if case != "execution":
                from .test_checkpoint_write_batches import _snapshot

                before = await _snapshot(archive.state_store)
                statements.clear()
                writes = []

                def capture_replay(connection, cursor, statement, parameters, context, many):
                    capture(connection, cursor, statement, parameters, context, many)
                    if statement.split()[0] in {"INSERT", "UPDATE", "DELETE"}:
                        writes.append((statement, parameters))

                event.listen(engine, "before_cursor_execute", capture_replay)
                try:
                    await commit()
                finally:
                    event.remove(engine, "before_cursor_execute", capture_replay)
                assert await _snapshot(archive.state_store) == before
                if case == "conversation":
                    assert not writes
                else:
                    # Trace still fences its execution producer, but replay must
                    # not rewrite transcript records, checkpoint facts or content.
                    assert len(writes) == 1 and writes[0][0].startswith("UPDATE ai_state_records")
                    repository = state.execution.executions
                    assert repository.state_store.store_key in writes[0][1]
                    assert repository._key("execution", ("execution",)) in writes[0][1]
        finally:
            await state.close()
    assert counts[1] == counts[0] - 1


async def test_checkpoint_projection_stale_version_rolls_back_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="checkpoint-stale-projection", tenant_id="tenant")
    archive = await _trace_archive(state)
    run = AgentRunRecord("run")
    try:
        await archive.materialize_checkpoint(run, _checkpoint(1), execution_id="execution")
        prepared, = (await archive.prepare_checkpoints(run, (_checkpoint(2),))).checkpoints
        projection_key = archive.transcript_repository.projection_key("run")
        replace_records = _SqlTransaction.replace_records
        injected = []

        async def stale_projection(transaction, replacements):
            projection = next((
                value for value in replacements if value.record.key == projection_key
            ), None)
            if projection is not None and not injected:
                injected.append(True)
                await replace_records(transaction, (projection,))
            return await replace_records(transaction, replacements)

        with monkeypatch.context() as patch:
            patch.setattr(_SqlTransaction, "replace_records", stale_projection)
            with pytest.raises(AIError) as raised:
                await archive._mutate(
                    lambda tx: archive.materialize_checkpoint_in_transaction(tx, run, prepared, execution_id="execution")
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
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="checkpoint-admission", tenant_id="tenant")
    archive = await _trace_archive(state)
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
            await archive.register_agent_run(AgentRunRecord("run"), execution_id="execution")
    finally:
        await state.close()
