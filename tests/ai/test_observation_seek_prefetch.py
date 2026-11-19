#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Observation seek metadata shares its publication transaction's existing read."""
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from sqlalchemy import event

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence
from linktools.ai.persistence._step_contracts import AgentRunRecord
from .test_incremental_projection_boundaries import _storage
from .test_checkpoint_metadata_snapshot_io import _trace_archive

async def _open(tmp_path: Path):
    state = Persistence(_storage('sqlite', tmp_path))
    await state.initialize(namespace='observation-seek', tenant_id='tenant')
    archive = await _trace_archive(state)
    run = AgentRunRecord('run')
    await archive.sync_projection(run, events=(), checkpoints=(), execution_id='execution')
    return state, archive, run

async def _prepare(archive, run, values, first=0):
    messages = tuple(ModelRequest(parts=[UserPromptPart(value, timestamp=run.started_at)]) for value in values)
    prepared = await archive.transcript_repository.prepare_observation('run', messages,
        first_message_index=first, pending=None, pending_keys=())
    return messages, prepared

async def _publish(archive, run, prepared):
    await archive.sync_prepared_projection(run, events=(), checkpoints=(), observation=prepared,
        execution_id='execution', producer_generation=1)

@pytest.mark.asyncio
@pytest.mark.parametrize('count', [1, 260])
async def test_observation_only_prefetch_batches_all_seek_blocks(tmp_path, monkeypatch, count):
    counts = []
    for prefetch in (False, True):
        root = tmp_path / str(prefetch)
        root.mkdir()
        state, archive, run = await _open(root)
        try:
            messages, prepared = await _prepare(archive, run, [str(i) for i in range(count)])
            repository = archive.transcript_repository
            engine = archive.state_store.context.engine.sync_engine
            verbs = []
            def capture(_conn, _cursor, statement, _parameters, _context, _many):
                verbs.append(statement.split()[0].upper())
            with monkeypatch.context() as patch:
                if not prefetch:
                    patch.setattr(repository, 'checkpoint_record_keys',
                        lambda owner, chunks: (repository.head_key(owner), repository.projection_key(owner)))
                event.listen(engine, 'before_cursor_execute', capture)
                try:
                    await _publish(archive, run, prepared)
                finally:
                    event.remove(engine, 'before_cursor_execute', capture)
            counts.append({v: verbs.count(v) for v in ('SELECT', 'INSERT', 'UPDATE', 'DELETE')})
            # Independent reads prove committed transcript and every expected seek key.
            assert await repository.load_messages('run') == messages
            keys = tuple(repository._seek_key('run', index) for index in range(0, count, 128))
            records = await archive.state_store.read(lambda tx: tx.get_records(keys))
            assert set(records) == set(keys)
        finally:
            await state.close()
    assert counts[0]['SELECT'] == counts[1]['SELECT'] + 1
    for verb in ('INSERT', 'UPDATE', 'DELETE'):
        assert counts[0][verb] == counts[1][verb]

@pytest.mark.asyncio
async def test_prefetch_does_not_reuse_preparation_before_independent_commit(tmp_path):
    state, archive, run = await _open(tmp_path)
    try:
        _, stale = await _prepare(archive, run, ['stale'])
        winner_messages, winner = await _prepare(archive, run, ['winner'])
        await _publish(archive, run, winner)
        with pytest.raises(AIError) as error:
            await _publish(archive, run, stale)
        assert error.value.code is ErrorCode.STORAGE_CONFLICT
        assert await archive.transcript_repository.load_messages('run') == winner_messages
        later_messages, later = await _prepare(archive, run, ['later'], first=1)
        await _publish(archive, run, later)
        assert await archive.transcript_repository.load_messages('run') == winner_messages + later_messages
    finally:
        await state.close()

@pytest.mark.asyncio
async def test_rolled_back_seek_prefetch_cannot_leak_dirty_state(tmp_path, monkeypatch):
    state, archive, run = await _open(tmp_path)
    try:
        messages, prepared = await _prepare(archive, run, ['only after commit'])
        repository = archive.transcript_repository
        commit = repository.commit_observation
        async def abort(tx, value):
            await commit(tx, value)
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        with monkeypatch.context() as patch:
            patch.setattr(repository, 'commit_observation', abort)
            with pytest.raises(AIError):
                await _publish(archive, run, prepared)
        assert await repository.load_messages('run') == ()
        assert await archive.state_store.read(lambda tx: tx.get_record(repository._seek_key('run', 0))) is None
        await _publish(archive, run, prepared)
        assert await repository.load_messages('run') == messages
    finally:
        await state.close()
