#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime scheduling combines pressure without combining durable owners."""

import asyncio
from dataclasses import replace

import pytest

from linktools.ai.capability import TraceCapability
from linktools.ai.core import CommitPolicy
from linktools.ai.observe import Metrics
from linktools.ai.observe._memory import InMemoryMetricStore
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._step_contracts import AgentRunRecord
from linktools.ai.storage import InMemoryCacheBackend
from .test_explicit_runtime_features import _FeatureModels, _features
from .test_observation_coalescing import _observe
from .test_runtime_metric_buffer import _observation
from .test_step_projection import _execution


async def _prepare_trace(runtime, policy):
    store = runtime._state.run_store
    repository = runtime._state.execution.executions
    execution = replace(_execution(), revision=1)
    await repository.create_with_history_head(execution)
    await repository.state_store.mutate(
        lambda transaction: repository.admit_history_producer_in_transaction(transaction, execution)
    )
    store.trace_persistence.configure(
        "run", buffer=policy.buffer, batch_size=policy.batch_size,
        flush_interval=policy.flush_interval,
    )
    store.bind_execution_producer("run", execution_id="execution", producer_generation=1)
    await store.register_agent_run(AgentRunRecord("run"), execution_id="execution")
    return store


async def _wait_for(predicate):
    async def wait():
        while not await predicate():
            await asyncio.sleep(0.001)
    await asyncio.wait_for(wait(), 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("sharing", ("same", "different_buffer", "different_timing"))
async def test_runtime_metrics_and_trace_share_only_matching_buffer_and_timing(sharing):
    buffer = InMemoryCacheBackend(max_bytes=65536)
    metric_policy = CommitPolicy.batch(buffer=buffer, batch_size=2, flush_interval=60)
    trace_policy = CommitPolicy.batch(
        buffer=InMemoryCacheBackend(max_bytes=65536) if sharing == "different_buffer" else buffer,
        batch_size=2, flush_interval=30 if sharing == "different_timing" else 60,
    )
    metrics = Metrics.in_memory(commit=metric_policy)
    async with Runtime.open(
        "shared-flush", storage=RuntimeStorage.in_memory(), models=_FeatureModels(),
        features=_features(TraceCapability(commit=trace_policy)), metrics=metrics,
    ) as runtime:
        store = await _prepare_trace(runtime, trace_policy)
        archive = store.read_store(RuntimeDomain.TRACE)
        assert await runtime._metric_control.try_record(_observation("metric"))
        assert await metrics.get_observation("metric") is None
        await _observe(store, 1)
        if sharing == "same":
            await _wait_for(lambda: metrics.get_observation("metric"))
            await _wait_for(lambda: archive.list_events(agent_run_id="run"))
            assert runtime.metrics.status().persisted == 1
        else:
            await asyncio.sleep(0.02)
            assert await metrics.get_observation("metric") is None
            assert await archive.list_events(agent_run_id="run") == []
            await runtime.metrics.flush()
            assert await metrics.get_observation("metric") is not None
            assert await archive.list_events(agent_run_id="run") == []


@pytest.mark.asyncio
async def test_same_metrics_dataset_and_buffer_do_not_share_pressure_across_runtimes():
    buffer = InMemoryCacheBackend(max_bytes=65536)
    policy = CommitPolicy.batch(buffer=buffer, batch_size=2, flush_interval=60)
    metrics = Metrics.in_memory(commit=policy)
    feature = TraceCapability(commit=policy)
    async with Runtime.open(
        "same-name", storage=RuntimeStorage.in_memory(), models=_FeatureModels(),
        features=_features(feature), metrics=metrics,
    ) as first, Runtime.open(
        "same-name", storage=RuntimeStorage.in_memory(), models=_FeatureModels(),
        features=_features(feature), metrics=metrics,
    ) as second:
        store = await _prepare_trace(first, policy)
        assert await first._metric_control.try_record(_observation("first"))
        assert await second._metric_control.try_record(_observation("second"))
        await asyncio.sleep(0.02)
        assert await metrics.get_observation("first") is None
        assert await metrics.get_observation("second") is None
        await _observe(store, 1)
        await _wait_for(lambda: metrics.get_observation("first"))
        assert await metrics.get_observation("second") is None
        assert second.metrics.status().pending == 1


@pytest.mark.asyncio
async def test_trace_stop_does_not_flush_other_owner_pending_metrics():
    buffer = InMemoryCacheBackend(max_bytes=65536)
    policy = CommitPolicy.batch(buffer=buffer, batch_size=10, flush_interval=60)
    metrics = Metrics.in_memory(commit=policy)
    async with Runtime.open(
        "local-stop", storage=RuntimeStorage.in_memory(), models=_FeatureModels(),
        features=_features(TraceCapability(commit=policy)), metrics=metrics,
    ) as runtime:
        store = await _prepare_trace(runtime, policy)
        assert await runtime._metric_control.try_record(_observation("pending"))
        await _observe(store, 1)
        await store.trace_persistence.stop("run")
        assert await metrics.get_observation("pending") is None
        assert runtime.metrics.status().pending == 1
        await runtime.metrics.flush()
        assert await metrics.get_observation("pending") is not None


@pytest.mark.asyncio
async def test_metric_source_failure_does_not_prevent_shared_trace_flush():
    class FailingStore(InMemoryMetricStore):
        async def put_observations(self, namespace, observations):
            raise RuntimeError("metrics source unavailable")

    buffer = InMemoryCacheBackend(max_bytes=65536)
    policy = CommitPolicy.batch(buffer=buffer, batch_size=2, flush_interval=60)
    metrics = Metrics.from_store(FailingStore(), commit=policy)
    async with Runtime.open(
        "independent-errors", storage=RuntimeStorage.in_memory(), models=_FeatureModels(),
        features=_features(TraceCapability(commit=policy)), metrics=metrics,
    ) as runtime:
        store = await _prepare_trace(runtime, policy)
        assert await runtime._metric_control.try_record(_observation("failed"))
        await _observe(store, 1)
        await _wait_for(lambda: store.read_store(RuntimeDomain.TRACE).list_events(agent_run_id="run"))
        await runtime.metrics.flush()
        assert runtime.metrics.status().write_failures == 1
        assert runtime.metrics.status().unconfirmed == 1
        assert await metrics.get_observation("failed") is None
