#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sealed preparation batches sources without changing publication authority."""
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.persistence import Persistence
from linktools.ai.persistence._archive import PreparedArchive
from linktools.ai.persistence._step_archive import ExecutionProjectionBatch, LockOrderError
from linktools.ai.persistence._step_contracts import AgentRunRecord
from linktools.ai.persistence._transcript_staging import StagedTranscript
from linktools.ai.persistence.state import active_state_scope

from .test_checkpoint_metadata_snapshot_io import _assert_prepared_values_equal, _checkpoint, _trace_archive
from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("run_count", (1, 2))
async def test_terminal_source_batch_preserves_values_and_releases_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, run_count: int,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="terminal-source", tenant_id="tenant")
    try:
        archive = await _trace_archive(state)
        projections = []
        for index in range(run_count):
            run = AgentRunRecord(f"run-{index}")
            checkpoint = replace(_checkpoint(1), agent_run_id=run.agent_run_id)
            projections.append(ExecutionProjectionBatch(
                run, (), (checkpoint,), 0, 0, 0, 1, 0, 1,
                transcript=StagedTranscript(checkpoint.messages),
            ))
        statements = []
        engine = archive.state_store.storage_group.context.engine.sync_engine if backend == "sqlite" else None

        def count(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement.split()[0].upper())

        prepare = archive.transcript_repository.prepare_model_projection
        async def outside_scope(*args, **kwargs):
            assert active_state_scope() is None
            return await prepare(*args, **kwargs)
        monkeypatch.setattr(archive.transcript_repository, "prepare_model_projection", outside_scope)
        if engine is not None:
            event.listen(engine, "before_cursor_execute", count)
        try:
            expected_heads = await archive.execution_history_heads(tuple(p.run.agent_run_id for p in projections))
            expected = []
            for projection in projections:
                bundle = PreparedArchive()
                observation = await archive.transcript_repository.prepare_observation(
                    projection.run.agent_run_id, projection.transcript.messages,
                    first_message_index=0, pending=None, pending_keys=(), archive=bundle,
                )
                await archive.prepare_run_registration(projection.run)
                batch = await archive.prepare_checkpoints_after_seal(
                    projection.run, projection.checkpoints,
                    observed_message_count=observation.target_message_count, archive=bundle,
                )
                expected.append(batch)
            before = statements.count("SELECT")
            assert not set(statements) & {"INSERT", "UPDATE", "DELETE"}
            statements.clear()
            heads, preparations = await archive.capture_terminal_projection_preparations(projections)
            results = [await prepare(projection.run) for projection, prepare in zip(projections, preparations, strict=True)]
            after = statements.count("SELECT")
            assert not set(statements) & {"INSERT", "UPDATE", "DELETE"}
        finally:
            if engine is not None:
                event.remove(engine, "before_cursor_execute", count)
        assert heads == expected_heads
        for first, (second, observation, _) in zip(expected, results, strict=True):
            assert observation is not None and observation.target_message_count == 1
            await _assert_prepared_values_equal(archive, first, second)
        if backend == "sqlite":
            assert (before, after) == (1 + 2 * run_count, 1)
        # A captured source is not permission to prepare inside a later state scope.
        with pytest.raises(LockOrderError):
            await archive.state_store.read(lambda tx: preparations[0](projections[0].run))
    finally:
        await state.close()


async def test_terminal_capture_honors_after_seal_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="terminal-override", tenant_id="tenant")
    try:
        archive = await _trace_archive(state)
        original = archive.prepare_checkpoints_after_seal
        calls = []
        async def custom(self, run, checkpoints, **kwargs):
            calls.append(run.agent_run_id)
            return await original(run, checkpoints, **kwargs)
        monkeypatch.setattr(type(archive), "prepare_checkpoints_after_seal", custom)
        projection = ExecutionProjectionBatch(AgentRunRecord("run"), (), (), 0, 0, 0, 0, 0, 0)
        _, preparations = await archive.capture_terminal_projection_preparations((projection,))
        await preparations[0](projection.run)
        assert calls == ["run"]
    finally:
        await state.close()


async def test_terminal_independent_registration_follows_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="terminal-order", tenant_id="tenant")
    try:
        archive = await _trace_archive(state)
        owner = archive._run_registrations
        assert owner is not None
        monkeypatch.setattr(owner, "joins", lambda store: False)
        order = []
        original_observation = archive.transcript_repository.prepare_observation
        original_registration = archive.prepare_run_registration
        async def observe(*args, **kwargs):
            order.append("observation")
            return await original_observation(*args, **kwargs)
        async def register(*args, **kwargs):
            assert active_state_scope() is None
            order.append("registration")
            return await original_registration(*args, **kwargs)
        monkeypatch.setattr(archive.transcript_repository, "prepare_observation", observe)
        monkeypatch.setattr(archive, "prepare_run_registration", register)
        projection = ExecutionProjectionBatch(
            AgentRunRecord("run"), (), (), 0, 0, 0, 0, 0, 0, transcript=StagedTranscript(()),
        )
        _, preparations = await archive.capture_terminal_projection_preparations((projection,))
        assert order == []
        await preparations[0](projection.run)
        assert order == ["observation", "registration"]
    finally:
        await state.close()
