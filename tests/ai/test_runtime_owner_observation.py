#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Only retained turn ownership can replace a fresh source control observation."""
from pathlib import Path
from collections.abc import Mapping, Sequence
from linktools.ai.persistence.state import StoredRecord
import pytest
from linktools.ai.storage import InMemoryCacheBackend, _write_cache
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import runs as runs_module
from .test_model_control_observation import _state, _seed, _advance_execution
from .test_runtime_checkpoint_publication_reads import _checkpoint

@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_unowned_control_reads_each_boundary_and_rejects_changed_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    clock = [100.0]
    monkeypatch.setattr(_write_cache, "time", lambda: clock[0])
    monkeypatch.setattr(runs_module, "monotonic", lambda: clock[0])
    async with _state(backend, tmp_path, InMemoryCacheBackend(max_bytes=1000000)) as state:
        await _seed(state, restart=True)
        repository = state.execution.executions
        read = repository.state_store.refresh_records
        reads = []
        async def counted(keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord]:
            reads.append(tuple(keys))
            return await read(keys)
        monkeypatch.setattr(repository.state_store, "refresh_records", counted)
        for _ in range(10):
            await state.run_store.require_active_producer("run")
        assert reads == [(repository.execution_record_key("execution"),)] * 10
        await _advance_execution(repository)
        with pytest.raises(AIError) as error:
            await state.run_store.require_active_producer("run")
        assert error.value.code is ErrorCode.STORAGE_CONFLICT
        await state.run_store.save_checkpoint(_checkpoint(1), execution_id="execution", producer_generation=1)
        with pytest.raises(AIError) as error:
            await state.run_store.require_active_producer("run")
        assert error.value.code is ErrorCode.STORAGE_CONFLICT
        clock[0] += 4
        with pytest.raises(AIError) as error:
            await state.run_store.require_active_producer("run")
        assert error.value.code is ErrorCode.STORAGE_CONFLICT

@pytest.mark.asyncio
async def test_foreign_generation_change_is_seen_at_next_control_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [100.0]
    monkeypatch.setattr(_write_cache, "time", lambda: clock[0])
    monkeypatch.setattr(runs_module, "monotonic", lambda: clock[0])
    async with _state('sqlite', tmp_path, InMemoryCacheBackend(max_bytes=1000000)) as owner:
        await _seed(owner, restart=True)
        await owner.run_store.require_active_producer('run')
        async with _state('sqlite', tmp_path, InMemoryCacheBackend(max_bytes=1000000)) as replacement:
            await _advance_execution(replacement.execution.executions)
            # The other Runtime cannot invalidate this process's content cache.
            with pytest.raises(AIError) as error:
                await owner.run_store.require_active_producer('run')
            assert error.value.code is ErrorCode.STORAGE_CONFLICT
            await owner.run_store.save_checkpoint(_checkpoint(1), execution_id='execution', producer_generation=1)
            with pytest.raises(AIError) as error:
                await owner.run_store.require_active_producer('run')
            assert error.value.code is ErrorCode.STORAGE_CONFLICT
            clock[0] = 104.0
            with pytest.raises(AIError) as error:
                await owner.run_store.require_active_producer('run')
            assert error.value.code is ErrorCode.STORAGE_CONFLICT

@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ('late', 'rebind', 'owner_lost', 'cancel'))
async def test_owner_observation_cannot_publish_stale_local_permission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    import asyncio
    clock = [100.0]
    monkeypatch.setattr(runs_module, 'monotonic', lambda: clock[0])
    async with _state('memory', tmp_path) as state:
        await _seed(state, restart=True)
        store = state.run_store
        repository = state.execution.executions
        original = repository.state_store.refresh_records
        entered, finish = asyncio.Event(), asyncio.Event()
        async def delayed(keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord]:
            value = await original(keys)
            entered.set()
            await finish.wait()
            return value
        monkeypatch.setattr(repository.state_store, 'refresh_records', delayed)
        active = [True]
        def guard(*_args):
            if not active[0]:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return False
        store.bind_pending_producer_guard(guard)
        request = asyncio.create_task(store.require_active_producer('run'))
        await entered.wait()
        if outcome == 'late':
            clock[0] += 4
        elif outcome == 'owner_lost':
            active[0] = False
        elif outcome == 'rebind':
            store.bind_execution_producer('run', execution_id='execution', producer_generation=2, restart_checkpoints=True, durable_trace=False)
        else:
            request.cancel()
        finish.set()
        if outcome == 'late':
            # No arbitrary source-read expiry independent of the host scope.
            await request
        else:
            with pytest.raises((AIError, asyncio.CancelledError)):
                await request
        active[0] = True
        monkeypatch.setattr(repository.state_store, 'refresh_records', original)
        if outcome != 'rebind':
            await store.require_active_producer('run')
            await _advance_execution(repository)
            with pytest.raises(AIError) as error:
                await store.require_active_producer('run')
            assert error.value.code is ErrorCode.STORAGE_CONFLICT


@pytest.mark.asyncio
async def test_retained_producer_guard_does_not_cache_authority_after_it_releases(tmp_path: Path) -> None:
    async with _state('memory', tmp_path) as state:
        await _seed(state, restart=True)
        retained = True
        state.run_store.bind_pending_producer_guard(lambda execution_id, generation, allow_cancelling: retained)
        await _advance_execution(state.execution.executions)
        await state.run_store.require_active_producer('run')
        retained = False
        with pytest.raises(AIError) as error:
            await state.run_store.require_active_producer('run')
        assert error.value.code is ErrorCode.STORAGE_CONFLICT
