#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Automatic Agent metrics are explicit and independently owned."""
import asyncio
import pytest
from pydantic_ai.models.test import TestModel
from linktools.ai.capability import AgentContext, RuntimeFeatures
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.observe import Metrics
from linktools.ai.observe._memory import InMemoryMetricStore
from linktools.ai.runtime import Runtime, RuntimeStorage, MetricsCapability, CheckpointCapability, TraceCapability
from linktools.ai.storage import InMemoryCacheBackend
from .test_explicit_runtime_features import _FeatureModels, _features, _definitions


class _RecordingStore(InMemoryMetricStore):
    def __init__(self):
        super().__init__()
        self.observations = []
    async def put_observations(self, namespace, observations):
        await super().put_observations(namespace, observations)
        self.observations.extend(observations)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("cached", (False, True))
async def test_agent_metrics_require_capability_and_keep_batching(enabled, cached):
    store = _RecordingStore()
    metrics = Metrics.from_store(store, cache=InMemoryCacheBackend(max_bytes=2**20) if cached else None)
    features = _features(MetricsCapability()) if enabled else _definitions()
    async with Runtime.open("metrics-explicit", storage=RuntimeStorage.in_memory(), models=_FeatureModels(), metrics=metrics, runtime_features=features) as runtime:
        execution = await runtime.agents.get().start("question")
        await execution.wait(timeout_seconds=5)
        if cached or not enabled:
            assert store.observations == []
        await runtime.metrics.flush()
        kinds = {item.kind for item in store.observations}
        if enabled:
            assert {"linktools.model.request", "linktools.execution.terminal"} <= kinds
        else:
            assert not kinds
        ids = [item.observation_id for item in store.observations]
        assert len(ids) == len(set(ids))


@pytest.mark.asyncio
async def test_metrics_capability_requires_sink():
    with pytest.raises(AIError) as missing:
        async with Runtime.open("missing-sink", storage=RuntimeStorage.in_memory(), models=_FeatureModels(), runtime_features=_features(MetricsCapability())):
            pytest.fail("Metrics capability without sink must fail at open")
    assert missing.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY


@pytest.mark.asyncio
async def test_trace_and_checkpoint_do_not_enable_metrics():
    store = _RecordingStore()
    async with Runtime.open("trace-only", storage=RuntimeStorage.in_memory(), models=_FeatureModels(), metrics=Metrics.from_store(store), runtime_features=_features(TraceCapability(), CheckpointCapability())) as runtime:
        execution = await runtime.agents.get().start("question")
        await execution.wait(timeout_seconds=5)
        assert (await execution.model_interactions()).items
        await runtime.metrics.flush()
        assert not store.observations


@pytest.mark.asyncio
async def test_shared_metrics_survive_other_runtime_close():
    store = _RecordingStore()
    metrics = Metrics.from_store(store, cache=InMemoryCacheBackend(max_bytes=2**20))
    for namespace in ("first", "second"):
        async with Runtime.open(namespace, storage=RuntimeStorage.in_memory(), models=_FeatureModels(), metrics=metrics, runtime_features=_features(MetricsCapability())) as runtime:
            execution = await runtime.agents.get().start("question")
            await execution.wait(timeout_seconds=5)
    assert {item.source_namespace for item in store.observations} == {"first", "second"}


@pytest.mark.asyncio
async def test_tool_metrics_are_not_duplicated():
    store = _RecordingStore()
    async def lookup(context: AgentContext[None]) -> str:
        return "value"
    features = RuntimeFeatures("tool-metrics")
    features.tool(lookup, effect_policy="none")
    features.agent("default", model="default", allow_tools=("lookup",))
    features.capability(MetricsCapability(), scope="runtime")
    async with Runtime.open("tool-metrics", storage=RuntimeStorage.in_memory(), models=_FeatureModels(TestModel(call_tools=["lookup"])), metrics=Metrics.from_store(store), runtime_features=features) as runtime:
        execution = await runtime.agents.get().start("lookup")
        await execution.wait(timeout_seconds=5)
    tools = [item for item in store.observations if item.kind == "linktools.tool.execution"]
    assert len(tools) == 1


@pytest.mark.asyncio
async def test_concurrent_runs_keep_metric_execution_identity():
    store = _RecordingStore()
    async with Runtime.open("concurrent", storage=RuntimeStorage.in_memory(), models=_FeatureModels(), metrics=Metrics.from_store(store), runtime_features=_features(MetricsCapability())) as runtime:
        executions = await asyncio.gather(*(runtime.agents.get().start(str(i)) for i in range(3)))
        await asyncio.gather(*(execution.wait(timeout_seconds=5) for execution in executions))
    models = [item for item in store.observations if item.kind == "linktools.model.request"]
    assert len(models) == 3
    assert {item.correlation["linktools.execution_id"] for item in models} == {execution.execution_id for execution in executions}
    assert len({item.observation_id for item in models}) == 3
