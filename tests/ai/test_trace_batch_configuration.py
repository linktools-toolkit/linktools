#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace observation batching is per run and never weakens explicit barriers."""

import asyncio

import pytest

from linktools.ai.capability import TraceCapability
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.core import CommitPolicy
from .test_observation_coalescing import _add_run, _observe, _setup
from linktools.ai.runtime.state import RuntimeDomain


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
        store.trace_persistence.configure("run", batch_size=2, flush_interval=60.0)
        archive = store.read_store(RuntimeDomain.TRACE)
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
        store.trace_persistence.configure("run", batch_size=100, flush_interval=0.01)
        store.trace_persistence.configure("slow", batch_size=100, flush_interval=60.0)
        archive = store.read_store(RuntimeDomain.TRACE)
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
async def test_runtime_binds_trace_batch_options_before_model_work():
    from pydantic_ai.messages import ModelResponse, TextPart
    from linktools.ai.runtime import Runtime, RuntimeStorage
    from .test_explicit_runtime_features import _FeatureFunctionModel, _FeatureModels, _features

    entered, release = asyncio.Event(), asyncio.Event()

    async def model(messages, info):
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("done")])

    feature = TraceCapability(commit=CommitPolicy.batch(batch_size=7, flush_interval=0.25))
    storage = RuntimeStorage.in_memory()
    async with Runtime.open(
        "trace-options", features=_features(feature), storage=storage,
        models=_FeatureModels(_FeatureFunctionModel(model)),
    ) as runtime:
        execution = await runtime.agents.get().start("question")
        try:
            await asyncio.wait_for(entered.wait(), 2)
            offsets = tuple(storage.run_store.trace_persistence.offsets.values())
            assert len(offsets) == 1
            assert offsets[0].observation_batch_size == 7
            assert offsets[0].observation_flush_interval == 0.25
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
    await restored.initialize(
        namespace="observation-batches", tenant_id="tenant",
        module_routes={RuntimeDomain.TRACE: restored.plan.route(RuntimeDomain.TRACE).with_cache(
            InMemoryCacheBackend(max_bytes=65536),
        )},
    )
    try:
        store = restored.run_store
        store.trace_persistence.configure("run", batch_size=2, flush_interval=60.0)
        store.bind_execution_producer("run", execution_id="execution", producer_generation=1)
        await store.register_agent_run(run, execution_id="execution")
        offset = store.trace_persistence.offsets["run"]
        assert offset.events == 1
        assert offset.observation_batch_size == 2
        assert offset.observation_flush_interval == 60.0
        await _observe(store, 2)
        archive = store.read_store(RuntimeDomain.TRACE)
        assert len(await archive.list_events(agent_run_id="run")) == 1
        await _observe(store, 3)
        async def wait_pressure():
            while len(await archive.list_events(agent_run_id="run")) < 3:
                await asyncio.sleep(0)
        await asyncio.wait_for(wait_pressure(), 2)
        assert len(await archive.list_events(agent_run_id="run")) == 3
    finally:
        await restored.close()
