#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace checkpoint preparation reuses only metadata preceding body I/O."""

from linktools.ai.persistence import Persistence
import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence._step_contracts import AgentRunRecord
from linktools.ai.persistence.state import active_state_scope

from .test_checkpoint_metadata_snapshot_io import _checkpoint, _trace_archive
from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("backend", ("memory", "sqlite",))
@pytest.mark.parametrize("fault", ("append", "cancel", "failure"))
async def test_run_preparation_body_io_keeps_fresh_commit_and_releases_cancelled_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, fault: str,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="run-preparation", tenant_id="tenant")
    archive = await _trace_archive(state)
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
            await archive.materialize_checkpoint(run, _checkpoint(1), execution_id="execution")
        return await prepare(*args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(archive.transcript_repository, "prepare_model_projection", prepare_with_fault)
            if fault == "append":
                prepared, = (await archive.prepare_checkpoints(run, (_checkpoint(2),))).checkpoints
            else:
                with pytest.raises(asyncio.CancelledError if fault == "cancel" else OSError):
                    await archive.prepare_checkpoints(run, (_checkpoint(2),))
        assert entered and active_state_scope() is None
        if fault == "append":
            assert prepared.chunks[0].first_message_index == 0
            with pytest.raises(AIError) as raised:
                await archive._mutate(
                    lambda tx: archive.materialize_checkpoint_in_transaction(tx, run, prepared, execution_id="execution")
                )
            assert raised.value.code is ErrorCode.STORAGE_CONFLICT
            latest = await archive.latest_checkpoint(agent_run_id="run")
            assert latest is not None and latest.messages == _checkpoint(1).messages
        else:
            assert await archive.get_agent_run(agent_run_id="run") is None
        head = await archive.transcript_repository.get_head("run")
        before = 0 if head is None else head.message_count
        prepared, = (await archive.prepare_checkpoints(
            run, (replace(_checkpoint(2), transcript_message_count_before=before),),
        )).checkpoints
        await archive._mutate(
            lambda tx: archive.materialize_checkpoint_in_transaction(tx, run, prepared, execution_id="execution")
        )
        latest = await archive.latest_checkpoint(agent_run_id="run")
        assert latest is not None and latest.messages == _checkpoint(2).messages
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "sqlite",))
async def test_run_preparation_rejects_corrupt_head_without_run(
    tmp_path: Path, backend: str,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="run-preparation", tenant_id="tenant")
    archive = await _trace_archive(state)
    run = AgentRunRecord("run")
    try:
        await archive.materialize_checkpoint(run, _checkpoint(1), execution_id="execution")
        key = archive.transcript_repository.head_key("run")
        async def corrupt(transaction):
            await transaction.delete_record(archive._agent_run_key("run"))
            record = await transaction.get_record(key)
            assert record is not None
            await transaction.replace_record(
                replace(record, data={}, storage_revision=record.storage_revision + 1),
                expected_storage_revision=record.storage_revision,
            )
        await archive._mutate(corrupt)
        with pytest.raises(AIError) as raised:
            await archive.prepare_checkpoints(run, (_checkpoint(2),))
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert active_state_scope() is None
    finally:
        await state.close()
