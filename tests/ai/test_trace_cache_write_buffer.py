#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Trace payload errors preserve authoritative source and cancellation boundaries."""
from ._runtime_test_helpers import model_features

import asyncio

import pytest

from linktools.ai.capability import TraceCapability
from linktools.ai.core import CommitPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.storage import InMemoryCacheBackend
from .test_observation_coalescing import _observe, _setup
from .test_explicit_runtime_features import _FeatureModels, _definitions


class ContentCache:
    async def get(self, key):
        return None

    async def put(self, key, value):
        pass

    async def delete(self, key):
        pass


async def _wait_until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)
    await asyncio.wait_for(wait(), 2)


@pytest.mark.asyncio
@pytest.mark.parametrize('cache', [None, ContentCache()])
async def test_trace_without_write_cache_commits_each_boundary(tmp_path, cache):
    state = await _setup(tmp_path, 'sqlite', cache=cache)
    try:
        store = state.run_store
        store.trace_persistence.configure('run', buffer=store.trace_persistence._caches['run'], batch_size=128, flush_interval=60)
        archive = store.read_store(PersistenceDomain.TRACE)
        for index in range(1, 4):
            await _observe(store, index)
            assert len(await archive.list_events(agent_run_id='run')) == index
        assert not store.trace_persistence._buffers
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_trace_rejects_partial_write_buffer():
    class Partial:
        async def append(self, *args, **kwargs):
            return True
    features = _definitions()
    with pytest.raises(TypeError, match='buffer'):
        features.capability(TraceCapability(commit=CommitPolicy.batch(buffer=Partial(), batch_size=128, flush_interval=3)), scope='runtime')
        async with Persistence.open('partial-trace-cache', storage=PersistenceStorage.in_memory()) as persistence_62, Runtime.attach(persistence=persistence_62, features=model_features(_FeatureModels(), features)):
            pytest.fail('partial write buffers must be rejected before runtime admission')


@pytest.mark.asyncio
@pytest.mark.parametrize('code', [ErrorCode.STORAGE_COMMIT_UNKNOWN, ErrorCode.STORAGE_INTEGRITY_ERROR])
async def test_trace_callback_failure_surfaces_original_source_error(tmp_path, code):
    state = await _setup(tmp_path, 'sqlite')
    store = state.run_store
    owner = store.trace_persistence
    original = owner._publish
    error = AIError(code)
    attempted = asyncio.Event()
    async def fail(*args, **kwargs):
        attempted.set()
        raise error
    owner.configure('run', buffer=owner._caches['run'], batch_size=1)
    owner._publish = fail
    try:
        await _observe(store, 1)
        await asyncio.wait_for(attempted.wait(), 2)
        with pytest.raises(AIError) as captured:
            await store.flush_execution_projection('run', execution_id='execution', producer_generation=1)
        assert captured.value is error
        assert 'run' in owner.dirty
        assert await store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id='run') == []
        assert owner._buffers['run'].pending
        with pytest.raises(AIError) as close_error:
            await state.close()
        assert close_error.value is error
    finally:
        owner._publish = original


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel_waiter', [False, True])
async def test_trace_payload_owner_settles_source_before_barrier_cancellation(tmp_path, monkeypatch, cancel_waiter):
    state = await _setup(tmp_path, 'sqlite')
    store = state.run_store
    owner = store.trace_persistence
    original = owner._publish
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0
    async def gated(*args, **kwargs):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        await original(*args, **kwargs)
    owner._publish = gated
    owner.configure('run', buffer=owner._caches['run'], batch_size=1)
    try:
        await _observe(store, 1)
        await asyncio.wait_for(entered.wait(), 2)
        buffer = owner._buffers['run']
        assert buffer.inflight
        barrier = asyncio.create_task(store.flush_execution_projection(
            'run', execution_id='execution', producer_generation=1,
        ))
        await asyncio.sleep(0)
        if cancel_waiter:
            barrier.cancel()
        await asyncio.sleep(0)
        assert not barrier.done()
        release.set()
        if cancel_waiter:
            with pytest.raises(asyncio.CancelledError):
                await barrier
        else:
            await barrier
        assert calls == 1
        assert not buffer.pending
        assert not store._durability_flights
        assert len(await store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id='run')) == 1
    finally:
        release.set()
        owner._publish = original
        await state.close()


@pytest.mark.asyncio
async def test_trace_stop_before_dispatch_publishes_real_payload_before_ack(tmp_path):
    cache = InMemoryCacheBackend(max_bytes=65536)
    state = await _setup(tmp_path, 'sqlite', cache=cache)
    try:
        store = state.run_store
        owner = store.trace_persistence
        owner.configure('run', buffer=owner._caches['run'], batch_size=128, flush_interval=60)
        await _observe(store, 1)
        assert cache._pending_count == 1
        await owner.stop('run')
        assert 'run' not in owner.dirty
        await _wait_until(lambda: cache._pending_count == 0)
        assert len(await store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id='run')) == 1
        await store.flush_execution_projection('run', execution_id='execution', producer_generation=1)
        assert len(await store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id='run')) == 1
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_trace_unknown_commit_keeps_authoritative_flight_and_buffer_payload(tmp_path, monkeypatch):
    state = await _setup(tmp_path, 'memory')
    store = state.run_store
    owner = store.trace_persistence
    archive = store.read_store(PersistenceDomain.TRACE)
    original_head = archive.execution_history_head_record
    original_flush = owner._publish
    entered, release = asyncio.Event(), asyncio.Event()
    failed = False
    async def write_failed(*args, **kwargs):
        nonlocal failed
        failed = True
        raise RuntimeError('write outcome unavailable')
    async def readback_failed(*args, **kwargs):
        if failed:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
        return await original_head(*args, **kwargs)
    async def gated(*args, **kwargs):
        entered.set()
        await release.wait()
        await original_flush(*args, **kwargs)
    monkeypatch.setattr(archive, 'sync_prepared_projection', write_failed)
    monkeypatch.setattr(archive, 'execution_history_head_record', readback_failed)
    owner._publish = gated
    owner.configure('run', buffer=owner._caches['run'], batch_size=1)
    await _observe(store, 1)
    await asyncio.wait_for(entered.wait(), 2)
    buffer = owner._buffers['run']
    assert buffer.inflight
    release.set()
    await _wait_until(lambda: buffer.error is not None)
    assert isinstance(buffer.error, AIError)
    assert buffer.error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
    flight = store._durability_flights['run']
    with pytest.raises(AIError) as caught:
        await store.flush_execution_projection('run', execution_id='execution', producer_generation=1)
    assert caught.value is buffer.error
    assert store._durability_flights['run'] is flight
    assert 'run' in owner.dirty
    with pytest.raises(AIError) as close_error:
        await state.close()
    assert close_error.value is buffer.error
