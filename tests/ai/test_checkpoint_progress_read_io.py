#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace progress and bounded fact proofs always reflect their durable source."""

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint, AgentRunRecord

from .test_checkpoint_metadata_snapshot_io import _trace_archive
from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_trace_progress_and_exact_fact_position_survive_replay(tmp_path: Path, backend: str) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="progress-io", tenant_id="tenant")
    try:
        archive = await _trace_archive(state)
        run = AgentRunRecord("run")
        checkpoint = AgentRunCheckpoint("run", 1, [], transcript_message_count_before=0)
        prepared, = (await archive.prepare_checkpoints(run, (checkpoint,))).checkpoints
        assert await archive.checkpoint_count(agent_run_id="run") == 0
        assert not await archive.verify_checkpoint_range(agent_run_id="run", after_sequence=0, checkpoints=(prepared,))
        await archive.materialize_checkpoint(run, checkpoint, execution_id="execution")
        await archive.materialize_checkpoint(run, checkpoint, execution_id="execution")
        assert await archive.checkpoint_count(agent_run_id="run") == 1
        assert await archive.verify_checkpoint_range(agent_run_id="run", after_sequence=0, checkpoints=(prepared,))
        assert not await archive.verify_checkpoint_range(agent_run_id="run", after_sequence=1, checkpoints=(prepared,))
        changed = replace(prepared, stored=replace(prepared.stored, step_index=2))
        assert not await archive.verify_checkpoint_range(agent_run_id="run", after_sequence=0, checkpoints=(changed,))
        await archive._mutate(lambda tx: tx.reserve_sequence(archive._sequence("run", "interaction"), 3))
        assert await archive.model_interaction_count(agent_run_id="run") == 3
        await archive._mutate(lambda tx: tx.delete_fact_streams(archive._agent_run_key("run")))
        assert await archive.checkpoint_count(agent_run_id="run") == 1
        assert not await archive.verify_checkpoint_range(agent_run_id="run", after_sequence=0, checkpoints=(prepared,))
    finally:
        await state.close()


async def test_trace_checkpoint_proof_uses_one_bounded_read(tmp_path: Path) -> None:
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="progress-io", tenant_id="tenant")
    try:
        archive = await _trace_archive(state)
        run = AgentRunRecord("run")
        checkpoint = AgentRunCheckpoint("run", 1, [], transcript_message_count_before=0)
        prepared, = (await archive.prepare_checkpoints(run, (checkpoint,))).checkpoints
        await archive.materialize_checkpoint(run, checkpoint, execution_id="execution")
        statements = []
        engine = archive.state_store.storage_group.context.engine.sync_engine

        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement)

        event.listen(engine, "before_cursor_execute", capture)
        try:
            assert await archive.verify_checkpoint_range(agent_run_id="run", after_sequence=0, checkpoints=(prepared,))
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        assert len([sql for sql in statements if sql.startswith("SELECT")]) == 1
        assert not any(sql.startswith(("INSERT", "UPDATE", "DELETE")) for sql in statements)
        assert "LIMIT" in next(sql for sql in statements if sql.startswith("SELECT"))
    finally:
        await state.close()


async def test_trace_checkpoint_proof_rejects_history_lock_before_storage_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.persistence.runs import _AgentRunHistoryLock
    from linktools.ai.persistence.state import StateLockOrderError

    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="progress-io", tenant_id="tenant")
    try:
        archive = await _trace_archive(state)
        prepared, = (await archive.prepare_checkpoints(
            AgentRunRecord("run"), (AgentRunCheckpoint("run", 1, [], transcript_message_count_before=0),),
        )).checkpoints

        async def unexpected(*args, **kwargs):
            raise AssertionError("storage must not be opened under a history lock")

        monkeypatch.setattr(archive.state_store, "read", unexpected)
        async with _AgentRunHistoryLock().hold("run"):
            with pytest.raises(StateLockOrderError):
                await archive.verify_checkpoint_range(agent_run_id="run", after_sequence=0, checkpoints=(prepared,))
    finally:
        await state.close()


async def test_trace_checkpoint_proof_does_not_swallow_storage_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="progress-io", tenant_id="tenant")
    try:
        archive = await _trace_archive(state)
        prepared, = (await archive.prepare_checkpoints(
            AgentRunRecord("run"), (AgentRunCheckpoint("run", 1, [], transcript_message_count_before=0),),
        )).checkpoints

        async def fail(*args, **kwargs):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        monkeypatch.setattr(archive.state_store, "read", fail)
        with pytest.raises(AIError) as raised:
            await archive.verify_checkpoint_range(agent_run_id="run", after_sequence=0, checkpoints=(prepared,))
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()
