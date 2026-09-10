#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-scoped bounded metric dimension regressions."""

from datetime import datetime, timedelta, timezone

import pytest
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
from linktools.ai.runtime import Runtime
from linktools.ai.runtime._context import RuntimeContext
from linktools.ai.runtime._metrics import _RuntimeMetricBuffer
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
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
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
        source_namespace=workspace.workspace_id,
        tenant_id="default",
        status="SUCCEEDED",
        error_code=None,
        correlation={},
        dimensions={"agent_id": "default"},
        measurements=(),
    )

    async with Runtime.open(
        workspace,
        context=context,
        models=models,
        metrics=metrics,
    ) as runtime:
        control = runtime._metric_control  # type: ignore[attr-defined]
        assert isinstance(control, _RuntimeMetricBuffer)
        assert control.try_record(observation) is True
        flushed = await runtime.flush_metrics()
        assert flushed.completed is True

    stored = await metrics.get_observation(observation.observation_id)
    assert stored is not None
    assert stored.source_namespace == workspace.workspace_id
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
