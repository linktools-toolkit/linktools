#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpoint progress captures batch sequence reads without weakening replay proofs."""

from linktools.ai.runtime.state import RuntimeState
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._model_interaction_store import ModelInteractionStateStepArchive
from linktools.ai.runtime.state._step_archive import StateStepArchive
from linktools.ai.runtime.state._step_contracts import AgentRunCheckpoint, AgentRunRecord

from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_checkpoint_progress_keeps_repeated_checkpoint_position_and_fresh_high_water(
    tmp_path: Path, backend: str,
) -> None:
    state = RuntimeState(_storage(backend, tmp_path))
    await state.initialize(namespace="progress-io", tenant_id="tenant")
    try:
        archive = state.run_store.read_store(RuntimeDomain.CHECKPOINT)
        run = AgentRunRecord("run")
        checkpoint = AgentRunCheckpoint("run", 1, [], transcript_message_count_before=0)
        prepared, = (await archive.prepare_checkpoints(run, (checkpoint,))).checkpoints
        assert await archive.capture_checkpoint_progress(agent_run_id="run", checkpoint=prepared) == (1, 0)
        await archive.materialize_checkpoint(run, checkpoint)
        assert await archive.capture_checkpoint_progress(agent_run_id="run", checkpoint=prepared) == (1, 0)
        await archive.state_store.mutate(lambda tx: tx.reserve_sequence(archive._sequence("run", "interaction"), 3))
        assert await archive.capture_checkpoint_progress(agent_run_id="run", checkpoint=prepared) == (1, 3)
        different = replace(prepared, stored=replace(prepared.stored, step_index=2))
        assert await archive.capture_checkpoint_progress(agent_run_id="run", checkpoint=different) == (2, 3)
        await archive.state_store.mutate(lambda tx: tx.delete_fact_streams(archive._agent_run_key("run")))
        assert await archive.capture_checkpoint_progress(agent_run_id="run", checkpoint=prepared) == (2, 3)
    finally:
        await state.close()


async def test_checkpoint_progress_batches_sequence_queries_in_one_snapshot(tmp_path: Path) -> None:
    state = RuntimeState(_storage("sqlite", tmp_path))
    await state.initialize(namespace="progress-io", tenant_id="tenant")
    try:
        archive = state.run_store.read_store(RuntimeDomain.CHECKPOINT)
        prepared, = (await archive.prepare_checkpoints(
            AgentRunRecord("run"), (AgentRunCheckpoint("run", 1, [], transcript_message_count_before=0),),
        )).checkpoints
        engine = archive.state_store.storage_group.context.engine.sync_engine
        counts = []
        for method in (StateStepArchive.capture_checkpoint_progress, ModelInteractionStateStepArchive.capture_checkpoint_progress):
            statements = []
            def capture(_connection, _cursor, statement, _parameters, _context, _many):
                statements.append(statement.split()[0].upper())
            event.listen(engine, "before_cursor_execute", capture)
            try:
                assert await method(archive, agent_run_id="run", checkpoint=prepared) == (1, 0)
            finally:
                event.remove(engine, "before_cursor_execute", capture)
            assert not set(statements) & {"INSERT", "UPDATE", "DELETE"}
            counts.append(statements.count("SELECT"))
        assert counts[1] < counts[0]
    finally:
        await state.close()


async def test_checkpoint_progress_custom_archive_preserves_its_count_and_proof_overrides(tmp_path: Path) -> None:
    state = RuntimeState(_storage("memory", tmp_path))
    await state.initialize(namespace="progress-io", tenant_id="tenant")
    try:
        original = state.run_store.read_store(RuntimeDomain.CHECKPOINT)
        prepared, = (await original.prepare_checkpoints(
            AgentRunRecord("run"), (AgentRunCheckpoint("run", 1, [], transcript_message_count_before=0),),
        )).checkpoints
        calls = []
        class CustomArchive(ModelInteractionStateStepArchive):
            async def checkpoint_count(self, **kwargs):
                calls.append("checkpoint")
                return 7
            async def verify_checkpoint_range(self, **kwargs):
                assert kwargs["after_sequence"] == 6
                calls.append("proof")
                return True
            async def model_interaction_count(self, **kwargs):
                calls.append("interaction")
                return 11
        custom = CustomArchive(original.state_store, object_store=None, namespace="progress-io",
                               tenant_id="tenant", runtime_domain=RuntimeDomain.CHECKPOINT)
        assert await custom.capture_checkpoint_progress(agent_run_id="run", checkpoint=prepared) == (7, 11)
        assert calls == ["checkpoint", "proof", "interaction"]
    finally:
        await state.close()


async def test_checkpoint_progress_rejects_history_lock_before_storage_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.runtime.state._steps import _AgentRunHistoryLock
    from linktools.ai.runtime.state._store import StateLockOrderError
    state = RuntimeState(_storage("memory", tmp_path))
    await state.initialize(namespace="progress-io", tenant_id="tenant")
    try:
        archive = state.run_store.read_store(RuntimeDomain.CHECKPOINT)
        prepared, = (await archive.prepare_checkpoints(
            AgentRunRecord("run"), (AgentRunCheckpoint("run", 1, [], transcript_message_count_before=0),),
        )).checkpoints
        async def unexpected(*args, **kwargs):
            raise AssertionError("storage must not be opened under a history lock")
        monkeypatch.setattr(archive.state_store, "read", unexpected)
        async with _AgentRunHistoryLock().hold("run"):
            with pytest.raises(StateLockOrderError):
                await archive.capture_checkpoint_progress(agent_run_id="run", checkpoint=prepared)
    finally:
        await state.close()


async def test_checkpoint_progress_keeps_checkpoint_proof_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeState(_storage("memory", tmp_path))
    await state.initialize(namespace="progress-io", tenant_id="tenant")
    try:
        archive = state.run_store.read_store(RuntimeDomain.CHECKPOINT)
        checkpoint = AgentRunCheckpoint("run", 1, [], transcript_message_count_before=0)
        run = AgentRunRecord("run")
        prepared, = (await archive.prepare_checkpoints(run, (checkpoint,))).checkpoints
        await archive.materialize_checkpoint(run, checkpoint)
        async def fail(**kwargs):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        monkeypatch.setattr(archive, "verify_checkpoint_range", fail)
        with pytest.raises(AIError) as raised:
            await archive.capture_checkpoint_progress(agent_run_id="run", checkpoint=prepared)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()
