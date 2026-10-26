#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Trace observation batching is per run and never weakens explicit barriers."""
from ._runtime_test_helpers import model_features

from linktools.ai.persistence import Persistence
import asyncio

import pytest

from linktools.ai.capability import TraceCapability
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.core import CommitPolicy
from .test_observation_coalescing import _add_run, _observe, _setup
from linktools.ai.persistence import PersistenceDomain


@pytest.mark.parametrize("options", [
    {"batch_size": 0}, {"batch_size": True}, {"batch_size": 1.5},
    {"flush_interval": 0}, {"flush_interval": True},
    {"flush_interval": float("inf")}, {"flush_interval": float("nan")},
])
def test_trace_batch_configuration_rejects_invalid_values(options):
    with pytest.raises(ValueError):
        TraceCapability(commit=CommitPolicy.batch(**{"batch_size": 128, "flush_interval": 3.0, **options}))


def test_trace_batch_configuration_keeps_defaults():
    feature = TraceCapability()
    from linktools.ai.core import DEFAULT_TRACE_BATCH_SIZE, DEFAULT_TRACE_FLUSH_INTERVAL

    assert feature.commit.batch_size == DEFAULT_TRACE_BATCH_SIZE == 128
    assert feature.commit.flush_interval == DEFAULT_TRACE_FLUSH_INTERVAL == 3.0


@pytest.mark.asyncio
async def test_trace_custom_pressure_and_required_flush_are_independent(tmp_path):
    state = await _setup(tmp_path, "memory")
    try:
        store = state.run_store
        store.trace_persistence.configure("run", buffer=InMemoryCacheBackend(max_bytes=65536), batch_size=2, flush_interval=60.0)
        archive = store.read_store(PersistenceDomain.TRACE)
        await _observe(store, 1)
        assert await archive.list_events(agent_run_id="run") == []
        await _observe(store, 2)
        async def wait_pressure():
            while len(await archive.list_events(agent_run_id="run")) < 2:
                await asyncio.sleep(0)
        await asyncio.wait_for(wait_pressure(), 2)
        assert len(await archive.list_events(agent_run_id="run")) == 2
        await _observe(store, 3)
        assert len(await archive.list_events(agent_run_id="run")) == 2
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert len(await archive.list_events(agent_run_id="run")) == 3
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_trace_scheduler_respects_each_runs_deadline(tmp_path):
    state = await _setup(tmp_path, "memory")
    try:
        await _add_run(state, "slow", "slow-execution")
        store = state.run_store
        store.trace_persistence.configure("run", buffer=store.trace_persistence._caches["run"], batch_size=100, flush_interval=0.01)
        store.trace_persistence.configure("slow", buffer=store.trace_persistence._caches["run"], batch_size=100, flush_interval=60.0)
        archive = store.read_store(PersistenceDomain.TRACE)
        await _observe(store, 1, run_id="slow", execution_id="slow-execution")
        await _observe(store, 1)
        async def wait_fast():
            while not await archive.list_events(agent_run_id="run"):
                await asyncio.sleep(0.001)
        await asyncio.wait_for(wait_fast(), 1)
        assert await archive.list_events(agent_run_id="slow") == []
        await store.flush_execution_projection("slow", execution_id="slow-execution", producer_generation=1)
        assert len(await archive.list_events(agent_run_id="slow")) == 1
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("options,expected", [({}, (128, 3.0)), ({"batch_size": 7, "flush_interval": 0.25}, (7, 0.25))])
async def test_runtime_binds_trace_batch_options_before_model_work(options, expected):
    from pydantic_ai.messages import ModelResponse, TextPart
    from linktools.ai.runtime import Runtime
    from linktools.ai.persistence import PersistenceStorage
    from .test_explicit_runtime_features import _FeatureFunctionModel, _FeatureModels, _features

    entered, release = asyncio.Event(), asyncio.Event()

    async def model(messages, info):
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("done")])

    buffer = InMemoryCacheBackend(max_bytes=65536)
    feature = TraceCapability(commit=CommitPolicy.batch(buffer=buffer, **options))
    storage = PersistenceStorage.in_memory()
    async with Persistence.open('trace-options', storage=storage) as persistence_98, Runtime.open(persistence=persistence_98, features=model_features(_FeatureModels(_FeatureFunctionModel(model)), _features(feature))) as runtime:
        execution = await runtime.agents.get().start("question")
        try:
            await asyncio.wait_for(entered.wait(), 2)
            offsets = tuple(runtime._run_store.trace_persistence.offsets.values())
            assert len(offsets) == 1
            assert offsets[0].observation_batch_size == expected[0]
            assert offsets[0].observation_flush_interval == expected[1]
            assert all(value is buffer for value in runtime._run_store.trace_persistence._caches.values())
            assert feature._service is None
        finally:
            release.set()
        await execution.wait(timeout_seconds=5)


@pytest.mark.asyncio
async def test_trace_policy_survives_restoring_existing_projection_offsets(tmp_path):
    from .test_incremental_projection_boundaries import _storage

    state = await _setup(tmp_path, "sqlite")
    run = await state.run_store.get_agent_run(agent_run_id="run")
    assert run is not None
    await _observe(state.run_store, 1)
    await state.run_store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
    await state.close()
    restored = _storage("sqlite", tmp_path)
    restored = Persistence(restored)
    await restored.initialize(
        namespace="observation-batches", tenant_id="tenant",
    )
    try:
        store = restored.run_store
        store.trace_persistence.configure("run", buffer=InMemoryCacheBackend(max_bytes=65536), batch_size=2, flush_interval=60.0)
        store.bind_execution_producer("run", execution_id="execution", producer_generation=1)
        await store.register_agent_run(run, execution_id="execution")
        offset = store.trace_persistence.offsets["run"]
        assert offset.events == 1
        assert offset.observation_batch_size == 2
        assert offset.observation_flush_interval == 60.0
        await _observe(store, 2)
        archive = store.read_store(PersistenceDomain.TRACE)
        assert len(await archive.list_events(agent_run_id="run")) == 1
        await _observe(store, 3)
        async def wait_pressure():
            while len(await archive.list_events(agent_run_id="run")) < 3:
                await asyncio.sleep(0)
        await asyncio.wait_for(wait_pressure(), 2)
        assert len(await archive.list_events(agent_run_id="run")) == 3
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_independent_runtime_stores_borrow_one_buffer_without_draining_each_other(tmp_path):
    shared = InMemoryCacheBackend(max_bytes=65536)
    first = await _setup(tmp_path / "first", "memory", cache=shared)
    second = await _setup(tmp_path / "second", "memory", cache=shared)
    try:
        await _observe(first.run_store, 1)
        await _observe(second.run_store, 1)
        await first.close()
        assert len(await second.run_store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id="run")) == 0
        await second.run_store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert len(await second.run_store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id="run")) == 1
        await shared.put("still-borrowed", b"usable")
        assert await shared.get("still-borrowed") == b"usable"
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_trace_uses_buffer_without_hot_cache_protocol(tmp_path):
    class Buffer:
        def __init__(self):
            self.backend = InMemoryCacheBackend(max_bytes=65536)

        async def append(self, scope, id, key, content, *, wait):
            return await self.backend.append(scope, id, key, content, wait=wait)

        async def read_batch(self, scope, *, limit):
            return await self.backend.read_batch(scope, limit=limit)

        async def ack(self, scope, tokens):
            return await self.backend.ack(scope, tokens)

    state = await _setup(tmp_path, "memory", cache=Buffer())
    try:
        await _observe(state.run_store, 1)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        assert await archive.list_events(agent_run_id="run") == []
        await state.run_store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert len(await archive.list_events(agent_run_id="run")) == 1
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_trace_route_cache_does_not_supply_implicit_write_buffer():
    from linktools.ai.runtime import Runtime
    from linktools.ai.persistence import PersistenceStorage
    from linktools.ai.storage import StorageRoute
    from .test_explicit_runtime_features import _FeatureModels, _features

    class ReadCache(InMemoryCacheBackend):
        async def append(self, *args, **kwargs):
            pytest.fail("hot-cache route must not schedule writes")

    from dataclasses import replace
    feature = TraceCapability()
    storage = PersistenceStorage(replace(PersistenceStorage.in_memory().plan,
                                         trace=StorageRoute.memory(cache=ReadCache(max_bytes=65536))))
    async with Persistence.open('trace-hot-cache', storage=storage) as persistence_208, Runtime.open(persistence=persistence_208, features=model_features(_FeatureModels(), _features(feature))) as runtime:
        execution = await runtime.agents.get().start("question")
        assert (await execution.wait(timeout_seconds=5)).result is not None


@pytest.mark.asyncio
async def test_shared_trace_capability_keeps_runtime_lifetimes_independent():
    from pydantic_ai.messages import ModelResponse, TextPart
    from linktools.ai.runtime import Runtime
    from linktools.ai.persistence import PersistenceStorage
    from .test_explicit_runtime_features import _FeatureFunctionModel, _FeatureModels, _features

    class Buffer(InMemoryCacheBackend):
        closed = False

        async def close(self):
            self.closed = True

        async def append(self, *args, **kwargs):
            assert not self.closed
            return await super().append(*args, **kwargs)

    buffer = Buffer(max_bytes=65536)
    feature = TraceCapability(commit=CommitPolicy.batch(buffer=buffer, batch_size=128, flush_interval=60))
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked(messages, info):
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("second")])

    async with Persistence.open('shared-trace', storage=PersistenceStorage.in_memory()) as persistence_240, Runtime.open(persistence=persistence_240, features=model_features(_FeatureModels(_FeatureFunctionModel(blocked)), _features(feature))) as second:
        pending = await second.agents.get().start("second")
        try:
            await asyncio.wait_for(entered.wait(), 2)
            async with Persistence.open('shared-trace', storage=PersistenceStorage.in_memory()) as persistence_246, Runtime.open(persistence=persistence_246, features=model_features(_FeatureModels(), _features(feature))) as first:
                execution = await first.agents.get().start("first")
                assert (await execution.wait(timeout_seconds=5)).result is not None
            assert not buffer.closed
        finally:
            release.set()
        assert (await pending.wait(timeout_seconds=5)).result.output == {"text": "second"}
    assert not buffer.closed
