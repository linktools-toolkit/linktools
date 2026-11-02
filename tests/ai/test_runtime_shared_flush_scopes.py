#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Metrics recording and Trace batching have independent publication lifetimes."""
from ._runtime_test_helpers import model_features

import asyncio
from dataclasses import replace

import pytest

from linktools.ai.capability import TraceCapability
from linktools.ai.core import CommitPolicy
from linktools.ai.observe import Metrics
from linktools.ai.observe._memory import InMemoryMetricStore
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence._step_contracts import AgentRunRecord
from linktools.ai.storage import InMemoryCacheBackend
from .test_explicit_runtime_features import _FeatureModels, _features
from .test_observation_coalescing import _observe
from .test_runtime_metric_buffer import _observation
from .test_step_projection import _execution


async def _prepare_trace(runtime, policy):
    store = runtime._run_store
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
async def test_metrics_writes_and_flush_do_not_publish_pending_trace():
    buffer = InMemoryCacheBackend(max_bytes=65536)
    policy = CommitPolicy.batch(buffer=buffer, batch_size=2, flush_interval=60)
    metrics = Metrics.in_memory()
    async with Persistence.open("independent-flush", storage=PersistenceStorage.in_memory()) as persistence, Runtime.attach(
        persistence=persistence, features=model_features(_FeatureModels(), _features(TraceCapability(commit=policy))), metrics=metrics,
    ) as runtime:
        store = await _prepare_trace(runtime, policy)
        archive = store.read_store(PersistenceDomain.TRACE)
        await _observe(store, 1)
        assert await archive.list_events(agent_run_id="run") == []
        assert await runtime._metric_control.try_record(_observation("metric"))
        assert await metrics.get_observation("metric") is not None
        assert (await runtime.metrics.flush()).completed
        assert await archive.list_events(agent_run_id="run") == []
        await _observe(store, 2)
        await _wait_for(lambda: archive.list_events(agent_run_id="run"))
        assert runtime.metrics.status().persisted == 1


@pytest.mark.asyncio
async def test_shared_metrics_does_not_share_trace_pressure_across_runtimes():
    buffer = InMemoryCacheBackend(max_bytes=65536)
    policy = CommitPolicy.batch(buffer=buffer, batch_size=2, flush_interval=60)
    metrics = Metrics.in_memory()
    feature = TraceCapability(commit=policy)
    async with Persistence.open("first", storage=PersistenceStorage.in_memory()) as first_persistence, Persistence.open(
        "second", storage=PersistenceStorage.in_memory(),
    ) as second_persistence, Runtime.attach(
        persistence=first_persistence, features=model_features(_FeatureModels(), _features(feature)), metrics=metrics,
    ) as first, Runtime.attach(
        persistence=second_persistence, features=model_features(_FeatureModels(), _features(feature)), metrics=metrics,
    ) as second:
        first_store = await _prepare_trace(first, policy)
        second_store = await _prepare_trace(second, policy)
        await _observe(first_store, 1)
        await _observe(second_store, 1)
        assert await first._metric_control.try_record(_observation("first"))
        assert await second._metric_control.try_record(_observation("second"))
        assert await metrics.get_observation("first") is not None
        assert await metrics.get_observation("second") is not None
        assert await first_store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id="run") == []
        assert await second_store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id="run") == []
        await _observe(first_store, 2)
        await _wait_for(lambda: first_store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id="run"))
        assert await second_store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id="run") == []


@pytest.mark.asyncio
async def test_trace_stop_does_not_close_metric_recorder():
    buffer = InMemoryCacheBackend(max_bytes=65536)
    policy = CommitPolicy.batch(buffer=buffer, batch_size=10, flush_interval=60)
    metrics = Metrics.in_memory()
    async with Persistence.open("local-stop", storage=PersistenceStorage.in_memory()) as persistence, Runtime.attach(
        persistence=persistence, features=model_features(_FeatureModels(), _features(TraceCapability(commit=policy))), metrics=metrics,
    ) as runtime:
        store = await _prepare_trace(runtime, policy)
        await _observe(store, 1)
        await store.trace_persistence.stop("run")
        assert runtime.metrics.status().accepting
        assert await runtime._metric_control.try_record(_observation("after-trace-stop"))
        assert await metrics.get_observation("after-trace-stop") is not None


@pytest.mark.asyncio
async def test_metric_source_failure_does_not_prevent_trace_flush():
    class FailingStore(InMemoryMetricStore):
        async def put_observations(self, namespace, observations):
            raise RuntimeError("metrics source unavailable")

    buffer = InMemoryCacheBackend(max_bytes=65536)
    policy = CommitPolicy.batch(buffer=buffer, batch_size=2, flush_interval=60)
    metrics = Metrics.from_store(FailingStore())
    async with Persistence.open("independent-errors", storage=PersistenceStorage.in_memory()) as persistence, Runtime.attach(
        persistence=persistence, features=model_features(_FeatureModels(), _features(TraceCapability(commit=policy))), metrics=metrics,
    ) as runtime:
        store = await _prepare_trace(runtime, policy)
        assert not await runtime._metric_control.try_record(_observation("failed"))
        await _observe(store, 1)
        await _observe(store, 2)
        await _wait_for(lambda: store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id="run"))
        await runtime.metrics.flush()
        assert runtime.metrics.status().write_failures == 1
        assert runtime.metrics.status().unconfirmed == 1
        assert await metrics.get_observation("failed") is None
