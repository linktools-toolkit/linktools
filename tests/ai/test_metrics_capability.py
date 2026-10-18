#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Automatic Agent metrics are explicit and independently owned."""
import asyncio
import pytest
from pydantic_ai.models.test import TestModel
from linktools.ai.capability import AgentContext, RuntimeFeatures
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
    features = _features(MetricsCapability(metrics=metrics)) if enabled else _definitions()
    async with Runtime.open("metrics-explicit", storage=RuntimeStorage.in_memory(), models=_FeatureModels(), features=features) as runtime:
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
            assert not runtime.metrics.status().enabled
        ids = [item.observation_id for item in store.observations]
        assert len(ids) == len(set(ids))


def test_metrics_capability_requires_sink():
    with pytest.raises(TypeError):
        MetricsCapability()
    with pytest.raises(TypeError, match="metrics must be Metrics"):
        MetricsCapability(metrics=None)


@pytest.mark.asyncio
async def test_trace_and_checkpoint_do_not_enable_metrics():
    store = _RecordingStore()
    async with Runtime.open("trace-only", storage=RuntimeStorage.in_memory(), models=_FeatureModels(), features=_features(TraceCapability(), CheckpointCapability())) as runtime:
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
        async with Runtime.open(namespace, storage=RuntimeStorage.in_memory(), models=_FeatureModels(), features=_features(MetricsCapability(metrics=metrics))) as runtime:
            execution = await runtime.agents.get().start("question")
            await execution.wait(timeout_seconds=5)
    assert {item.source_namespace for item in store.observations} == {"first", "second"}


@pytest.mark.asyncio
async def test_tool_metrics_are_not_duplicated():
    store = _RecordingStore()
    metrics = Metrics.from_store(store)
    async def lookup(context: AgentContext[None]) -> str:
        return "value"
    features = RuntimeFeatures("tool-metrics")
    features.tool(lookup, effect_policy="none")
    features.agent("default", model="default", allow_tools=("lookup",))
    features.capability(MetricsCapability(metrics=metrics), scope="runtime")
    async with Runtime.open("tool-metrics", storage=RuntimeStorage.in_memory(), models=_FeatureModels(TestModel(call_tools=["lookup"])), features=features) as runtime:
        execution = await runtime.agents.get().start("lookup")
        await execution.wait(timeout_seconds=5)
    tools = [item for item in store.observations if item.kind == "linktools.tool.execution"]
    assert len(tools) == 1


@pytest.mark.asyncio
async def test_concurrent_runs_keep_metric_execution_identity():
    store = _RecordingStore()
    metrics = Metrics.from_store(store)
    async with Runtime.open("concurrent", storage=RuntimeStorage.in_memory(), models=_FeatureModels(), features=_features(MetricsCapability(metrics=metrics))) as runtime:
        executions = await asyncio.gather(*(runtime.agents.get().start(str(i)) for i in range(3)))
        await asyncio.gather(*(execution.wait(timeout_seconds=5) for execution in executions))
    models = [item for item in store.observations if item.kind == "linktools.model.request"]
    assert len(models) == 3
    assert {item.correlation["linktools.execution_id"] for item in models} == {execution.execution_id for execution in executions}
    assert len({item.observation_id for item in models}) == 3


@pytest.mark.asyncio
async def test_absent_capability_does_not_open_recording_and_business_metrics_stay_independent(monkeypatch):
    from datetime import datetime, timezone
    from linktools.ai.observe import Observation

    metrics = Metrics.in_memory()
    def forbidden_recorder(self):
        pytest.fail("Runtime without MetricsCapability must not initialize recording")
    monkeypatch.setattr(Metrics, "recorder", forbidden_recorder)
    async with Runtime.open(
        "no-metric-owner", storage=RuntimeStorage.in_memory(), models=_FeatureModels(),
        features=_definitions(),
    ) as runtime:
        await runtime.agents.get().run("question", timeout_seconds=5)
        assert not runtime.metrics.status().enabled
        assert (await runtime.metrics.flush()).completed
    observation = Observation(
        version=1, observation_id="business", kind="business.sample",
        occurred_at=datetime.now(timezone.utc), source_namespace="business", tenant_id="tenant",
        status="SUCCEEDED", error_code=None, correlation={}, dimensions={}, measurements=(),
    )
    await metrics.record_observations((observation,))
    assert await metrics.get_observation("business") == observation


def test_runtime_open_has_no_second_metrics_configuration_source():
    import inspect
    assert "metrics" not in inspect.signature(Runtime.open).parameters
    with pytest.raises(TypeError, match="metrics"):
        Runtime.open("duplicate", storage=RuntimeStorage.in_memory(), models=_FeatureModels(), metrics=Metrics.in_memory())


@pytest.mark.asyncio
async def test_two_metrics_dataset_capabilities_are_rejected_before_recording(monkeypatch):
    from linktools.ai.errors import AIError, ErrorCode

    first = MetricsCapability(metrics=Metrics.in_memory())
    second = MetricsCapability(metrics=Metrics.in_memory())
    first.id, second.id = "metrics-first", "metrics-second"
    features = _definitions()
    features.capability(first, scope="runtime")
    features.capability(second, scope="runtime")
    def forbidden_recorder(self):
        pytest.fail("conflicting datasets must be rejected before opening a recorder")
    monkeypatch.setattr(Metrics, "recorder", forbidden_recorder)
    with pytest.raises(AIError) as duplicate:
        async with Runtime.open(
            "duplicate-metric-datasets", models=_FeatureModels(),
            storage=RuntimeStorage.in_memory(), features=features,
        ):
            pytest.fail("multiple metric datasets accepted")
    assert duplicate.value.code is ErrorCode.CAPABILITY_CONFLICT
