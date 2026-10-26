#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run checkpoint preparation reuses only metadata preceding body I/O."""

from linktools.ai.persistence import Persistence
import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence._model_interaction_store import ModelInteractionStateStepArchive
from linktools.ai.persistence._step_contracts import AgentRunRecord
from linktools.ai.persistence.state import active_state_scope

from .test_checkpoint_metadata_snapshot_io import _assert_prepared_values_equal, _checkpoint
from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("existing", ("absent", "empty", "prefix", "head_without_run"))
async def test_run_preparation_reduces_empty_prefix_reads_without_changing_values(
    tmp_path: Path, existing: str,
) -> None:
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="run-preparation", tenant_id="tenant")
    archive = state.run_store.read_store(PersistenceDomain.CHECKPOINT)
    run = AgentRunRecord("run")
    engine = archive.state_store.storage_group.context.engine.sync_engine
    statements = []
    select_stores: list[str] = []

    def capture(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.split()[0].upper())
        if statement.startswith("SELECT"):
            select_stores.append(_parameters[0])

    try:
        if existing == "empty":
            await archive.register_agent_run(run)
        elif existing in {"prefix", "head_without_run"}:
            await archive.materialize_checkpoint(run, _checkpoint(1))
            if existing == "head_without_run":
                await archive.state_store.mutate(lambda tx: tx.delete_record(archive._agent_run_key("run")))
        # Both alternatives start with the shared content scope already resolved.
        await archive.transcript_repository.content_archive(create=True)
        batches, counts, domain_counts = [], [], []
        for composed in (False, True):
            statements.clear()
            select_stores.clear()
            event.listen(engine, "before_cursor_execute", capture)
            try:
                if composed:
                    _relocated, batch = await archive.prepare_run_checkpoint(run, _checkpoint(2))
                else:
                    relocated = await archive.relocate_run_checkpoint(run, _checkpoint(2))
                    batch = await archive.prepare_checkpoints(run, (relocated,))
                batches.append(batch)
            finally:
                event.remove(engine, "before_cursor_execute", capture)
            assert not set(statements) & {"INSERT", "UPDATE", "DELETE"}
            counts.append(statements.count("SELECT"))
            domain_counts.append(select_stores.count(archive.state_store.store_digest.hex()))
        await _assert_prepared_values_equal(archive, batches[0], batches[1])
        # Empty prefixes reuse one metadata capture. Prefix validation performs
        # source reads, so both paths must recapture metadata after body I/O.
        assert domain_counts == ([3, 3] if existing == "prefix" else [2, 1])
        assert counts == {
            "absent": [3, 2], "empty": [3, 2], "prefix": [5, 5],
            "head_without_run": [3, 2],
        }[existing]
        expected_start = 1 if existing in {"prefix", "head_without_run"} else 0
        assert batches[1].checkpoints[0].chunks[0].first_message_index == expected_start
        assert active_state_scope() is None
    finally:
        await state.close()


@pytest.mark.parametrize("override", ("relocate", "prepare"))
async def test_run_preparation_preserves_custom_relocation_and_preparation(
    tmp_path: Path, override: str,
) -> None:
    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="run-preparation", tenant_id="tenant")
    original = state.run_store.read_store(PersistenceDomain.CHECKPOINT)
    run = AgentRunRecord("run")

    class RelocatingArchive(ModelInteractionStateStepArchive):
        async def relocate_run_checkpoint(self, run, checkpoint):
            relocated = await super().relocate_run_checkpoint(run, checkpoint)
            return replace(relocated, step_index=19)

    class PreparingArchive(ModelInteractionStateStepArchive):
        async def prepare_checkpoints(self, run, checkpoints, **kwargs):
            batch = await super().prepare_checkpoints(run, checkpoints, **kwargs)
            return replace(batch, target_transcript_message_count=19)

    custom = (RelocatingArchive if override == "relocate" else PreparingArchive)(
        original.state_store, object_store=None, namespace="run-preparation",
        tenant_id="tenant", runtime_domain=PersistenceDomain.CHECKPOINT,
    )
    try:
        checkpoint = _checkpoint(1)
        expected_checkpoint = await custom.relocate_run_checkpoint(run, checkpoint)
        expected = await custom.prepare_checkpoints(run, (expected_checkpoint,))
        relocated, actual = await custom.prepare_run_checkpoint(run, checkpoint)
        assert relocated == expected_checkpoint and actual == expected
        assert (actual.checkpoints[0].stored.step_index if override == "relocate"
                else actual.target_transcript_message_count) == 19
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("fault", ("append", "cancel", "failure"))
async def test_run_preparation_body_io_keeps_fresh_commit_and_releases_cancelled_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, fault: str,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="run-preparation", tenant_id="tenant")
    archive = state.run_store.read_store(PersistenceDomain.CHECKPOINT)
    run = AgentRunRecord("run")
    prepare = archive.transcript_repository.prepare_model_projection
    entered = False

    async def prepare_with_fault(*args, **kwargs):
        nonlocal entered
        assert active_state_scope() is None
        if not entered:
            entered = True
            if fault == "cancel":
                raise asyncio.CancelledError
            if fault == "failure":
                raise OSError("body write failed")
            await archive.materialize_checkpoint(run, _checkpoint(1))
        return await prepare(*args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(archive.transcript_repository, "prepare_model_projection", prepare_with_fault)
            if fault == "append":
                prepared, = (await archive.prepare_run_checkpoint(run, _checkpoint(2)))[1].checkpoints
            else:
                with pytest.raises(asyncio.CancelledError if fault == "cancel" else OSError):
                    await archive.prepare_run_checkpoint(run, _checkpoint(2))
        assert entered and active_state_scope() is None
        if fault == "append":
            assert prepared.chunks[0].first_message_index == 0
            with pytest.raises(AIError) as raised:
                await archive.state_store.mutate(
                    lambda tx: archive.materialize_checkpoint_in_transaction(tx, run, prepared)
                )
            assert raised.value.code is ErrorCode.STORAGE_CONFLICT
            latest = await archive.latest_checkpoint(agent_run_id="run")
            assert latest is not None and latest.messages == _checkpoint(1).messages
        else:
            assert await archive.get_agent_run(agent_run_id="run") is None
        prepared, = (await archive.prepare_run_checkpoint(run, _checkpoint(2)))[1].checkpoints
        await archive.state_store.mutate(
            lambda tx: archive.materialize_checkpoint_in_transaction(tx, run, prepared)
        )
        latest = await archive.latest_checkpoint(agent_run_id="run")
        assert latest is not None and latest.messages == _checkpoint(2).messages
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_run_preparation_rejects_corrupt_head_without_run(
    tmp_path: Path, backend: str,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="run-preparation", tenant_id="tenant")
    archive = state.run_store.read_store(PersistenceDomain.CHECKPOINT)
    run = AgentRunRecord("run")
    try:
        await archive.materialize_checkpoint(run, _checkpoint(1))
        key = archive.transcript_repository.head_key("run")
        async def corrupt(transaction):
            await transaction.delete_record(archive._agent_run_key("run"))
            record = await transaction.get_record(key)
            assert record is not None
            await transaction.replace_record(
                replace(record, data={}, storage_revision=record.storage_revision + 1),
                expected_storage_revision=record.storage_revision,
            )
        await archive.state_store.mutate(corrupt)
        relocated = await archive.relocate_run_checkpoint(run, _checkpoint(2))
        assert relocated.transcript_message_count_before == 0
        for operation in (
            lambda: archive.prepare_checkpoints(run, (relocated,)),
            lambda: archive.prepare_run_checkpoint(run, _checkpoint(2)),
        ):
            with pytest.raises(AIError) as raised:
                await operation()
            assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            assert active_state_scope() is None
    finally:
        await state.close()
