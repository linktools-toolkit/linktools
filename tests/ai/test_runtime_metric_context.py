#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-scoped bounded metric dimension regressions."""

from ._runtime_test_helpers import workspace_features

from datetime import datetime, timedelta, timezone

import pytest
from linktools.ai.core import CommitPolicy
from linktools.ai.runtime.state import RuntimeState
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.observe import (
    MetricAggregation,
    MetricDefinition,
    MetricQuery,
    MetricSource,
    MetricType,
    MetricWindow,
    Metrics,
    Observation,
)
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime._context import RuntimeContext
from linktools.ai.runtime._metrics_service import MetricsService
from linktools.ai.workspace import Workspace


def test_runtime_context_normalizes_and_preserves_metric_dimensions() -> None:
    context = RuntimeContext(
        None,
        tenant_id="tenant",
        correlation={"request": "root"},
        metric_dimensions={"environment": "prod", "region": "us-west"},
    )

    assert dict(context.metric_dimensions) == {
        "environment": "prod",
        "region": "us-west",
    }
    overlaid = context.overlay({"request": "child"})
    assert dict(overlaid.correlation) == {"request": "child"}
    assert dict(overlaid.metric_dimensions) == dict(context.metric_dimensions)


@pytest.mark.parametrize(
    "dimensions",
    [
        {"execution_id": "execution"},
        {"linktools.owner": "value"},
        {"context.environment": "prod"},
        {f"key{index}": "value" for index in range(9)},
    ],
)
def test_runtime_context_rejects_unbounded_or_reserved_metric_dimensions(
    dimensions: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        RuntimeContext(None, metric_dimensions=dimensions)


@pytest.mark.asyncio
async def test_runtime_metric_dimensions_flow_into_automatic_observations_and_queries(
    tmp_path,
) -> None:
    workspace = Workspace.load(tmp_path)
    metrics = Metrics.in_memory(namespace="runtime-context")
    models = ModelRegistry.openai(model="gpt-test")
    context = RuntimeContext(
        None,
        metric_dimensions={"environment": "prod", "region": "us-west"},
    )
    occurred_at = datetime.now(timezone.utc)
    observation = Observation(
        version=1,
        observation_id="runtime-context-observation",
        kind="linktools.model.request",
        occurred_at=occurred_at,
        source_namespace="default",
        tenant_id="default",
        status="SUCCEEDED",
        error_code=None,
        correlation={},
        dimensions={"agent_id": "default"},
        measurements=(),
    )

    features = workspace_features("workspace", workspace=workspace)
    async with Runtime.open(
        "default",
        metrics=metrics, context=context,
        models=models,
        storage=RuntimeStorage.in_memory(),
        features=features,
    ) as runtime:
        control = runtime._metric_control  # type: ignore[attr-defined]
        assert isinstance(control, MetricsService)
        assert await control.try_record(observation) is True
        flushed = await runtime.metrics.flush()
        assert flushed.completed is True

    stored = await metrics.get_observation(observation.observation_id)
    assert stored is not None
    assert stored.source_namespace == "default"
    assert dict(stored.dimensions) == {
        "context.environment": "prod",
        "context.region": "us-west",
        "agent_id": "default",
    }

    result = await metrics.query(
        MetricQuery(
            "linktools.model.request.count",
            MetricWindow.between(
                occurred_at - timedelta(seconds=1),
                occurred_at + timedelta(seconds=1),
            ),
            filters={"context.environment": "prod"},
            group_by=("context.region",),
        )
    )
    assert len(result.points) == 1
    assert result.points[0].dimensions == (("context.region", "us-west"),)
    assert result.points[0].value == 1


@pytest.mark.asyncio
async def test_custom_metric_does_not_gain_implicit_context_query_fields() -> None:
    metrics = Metrics.in_memory(namespace="custom-context")
    await metrics.define(
        MetricDefinition(
            name="app.request.count",
            revision=1,
            observation_kind="app.request",
            source=MetricSource.observation_count(),
            metric_type=MetricType.COUNTER,
            unit="1",
            default_aggregation=MetricAggregation.SUM,
        )
    )

    with pytest.raises(AIError) as raised:
        await metrics.query(
            MetricQuery(
                "app.request.count",
                MetricWindow.recent(minutes=1),
                filters={"context.environment": "prod"},
            )
        )
    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID


@pytest.mark.asyncio
async def test_runtime_recorder_reports_later_source_failure_after_enrichment_failure() -> None:
    from linktools.ai.observe._memory import InMemoryMetricStore

    class FailingStore(InMemoryMetricStore):
        async def put_observations(
            self, namespace: str, observations: tuple[Observation, ...],
        ) -> None:
            raise OSError("source unavailable")

    recorder = MetricsService(Metrics.from_store(FailingStore()))
    assert not await recorder.try_record(object())  # type: ignore[arg-type]
    assert recorder.status().last_failure_code == "ENRICHMENT_FAILED"
    observation = Observation(
        version=1, observation_id="later-source-error", kind="test.metric",
        occurred_at=datetime.now(timezone.utc), source_namespace="default",
        tenant_id="default", status="SUCCEEDED", error_code=None,
        correlation={}, dimensions={}, measurements=(),
    )
    assert not await recorder.try_record(observation)
    assert recorder.status().last_failure_code == "WRITE_FAILED"
    assert recorder.status().rejected == 1
    await recorder.close()


@pytest.mark.asyncio
async def test_runtimes_sharing_metrics_close_only_their_own_write_scopes() -> None:
    from linktools.ai.storage import InMemoryCacheBackend

    metrics = Metrics.in_memory(commit=CommitPolicy.batch(batch_size=128, flush_interval=60.0, buffer=InMemoryCacheBackend(max_bytes=1024 * 1024)))
    features = await RuntimeFeatures("shared-metrics").capture()
    async with Runtime.open(
        "first", metrics=metrics, models=ModelRegistry(), storage=RuntimeStorage.in_memory(), features=features,
    ) as first:
        async with Runtime.open(
            "second", metrics=metrics, models=ModelRegistry(), storage=RuntimeStorage.in_memory(), features=features,
        ) as second:
            observation = Observation(
                version=1, observation_id="second-pending", kind="test.metric",
                occurred_at=datetime.now(timezone.utc), source_namespace="second",
                tenant_id="default", status="SUCCEEDED", error_code=None,
                correlation={}, dimensions={}, measurements=(),
            )
            assert await second._metric_control.try_record(observation)
            await first.close()
            assert second.metrics.status().accepting
            assert second.metrics.status().pending == 1
            assert await metrics.get_observation("second-pending") is None
            assert (await second.metrics.flush()).completed
    assert await metrics.get_observation("second-pending") is not None


@pytest.mark.asyncio
async def test_runtime_cancelled_close_keeps_storage_open_until_metric_sink_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from linktools.ai.observe._memory import InMemoryMetricStore
    from linktools.ai.storage import InMemoryCacheBackend
    from linktools.ai.storage import _write_cache as write_cache

    monkeypatch.setattr(write_cache, "_CLOSE_DEADLINE_SECONDS", 0.01)
    entered = asyncio.Event()
    cancelling = asyncio.Event()
    release = asyncio.Event()
    returned = asyncio.Event()
    storage_closed = asyncio.Event()

    class CancellingStore(InMemoryMetricStore):
        async def put_observations(
            self, namespace: str, observations: tuple[Observation, ...],
        ) -> None:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelling.set()
                await release.wait()
            await super().put_observations(namespace, observations)
            returned.set()

    metrics = Metrics.from_store(CancellingStore(), commit=CommitPolicy.batch(batch_size=128, flush_interval=60.0, buffer=InMemoryCacheBackend(max_bytes=1024 * 1024)))
    storage = RuntimeStorage.in_memory()
    original_close = RuntimeState.close

    async def close_storage(owner) -> None:
        assert returned.is_set()
        storage_closed.set()
        await original_close(owner)

    monkeypatch.setattr(RuntimeState, "close", close_storage)
    features = RuntimeFeatures("close-metrics")
    async with Runtime.open(
        "close-owner", metrics=metrics, models=ModelRegistry(), storage=storage, features=features,
    ) as runtime:
        observation = Observation(
            version=1, observation_id="close-inflight", kind="test.metric",
            occurred_at=datetime.now(timezone.utc), source_namespace="close-owner",
            tenant_id="default", status="SUCCEEDED", error_code=None,
            correlation={}, dimensions={}, measurements=(),
        )
        assert await runtime._metric_control.try_record(observation)
        await runtime.metrics.flush(timeout_seconds=0)
        await asyncio.wait_for(entered.wait(), 1)
        closing = asyncio.create_task(runtime.close())
        try:
            await asyncio.wait_for(cancelling.wait(), 1)
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
            assert not returned.is_set()
            assert not storage_closed.is_set()
            assert runtime.metrics.status().pending == 1
            assert runtime.metrics.status().queue_size == 1
        finally:
            release.set()
            await asyncio.wait_for(runtime.close(), 1)
    assert storage_closed.is_set()
    assert runtime.metrics.status().persisted == 1
    assert runtime.metrics.status().pending == 0


@pytest.mark.asyncio
async def test_metrics_resources_do_not_enter_captured_agent_binding_identity() -> None:
    from linktools.ai.agent import AgentCompiler

    features = RuntimeFeatures("metric-resource-identity")
    features.agent("default")
    capture = await features.capture()
    models = ModelRegistry.openai(model="unused-offline-model")
    bindings = []
    for metrics in (None, Metrics.in_memory(), Metrics.in_memory()):
        async with Runtime.open(
            "metric-resource-identity", models=models,
            storage=RuntimeStorage.in_memory(), features=capture, metrics=metrics,
        ) as runtime:
            compiled = runtime.agents.get().derive(instructions=("fixed binding",)).compiled
            assert compiled is not None
            compiler = AgentCompiler(
                model_resolver=models.capture(), candidates=(),
                agents={compiled.spec.id: compiled.spec},
            )
            bindings.append(compiler.bind(compiled).binding_contract.to_payload())
            assert runtime.metrics.status().enabled is (metrics is not None)
    assert bindings[0] == bindings[1] == bindings[2]
