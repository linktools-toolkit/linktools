#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metric query safety, empty-population, and namespace edge semantics."""

from datetime import datetime, timedelta, timezone

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.observe import (
    MetricAggregation,
    MetricDefinition,
    MetricQuery,
    MetricSource,
    MetricType,
    MetricWindow,
    Metrics,
)
from linktools.ai.observe._memory import InMemoryMetricStore


def _definition(
    name: str,
    metric_type: MetricType,
    default: MetricAggregation,
) -> MetricDefinition:
    return MetricDefinition(
        name=name,
        revision=1,
        observation_kind=f"{name}.sample",
        source=MetricSource.measurement("value"),
        metric_type=metric_type,
        unit="1",
        default_aggregation=default,
    )


@pytest.mark.asyncio
async def test_empty_population_values_follow_aggregation_semantics() -> None:
    metrics = Metrics.in_memory(namespace="empty-semantics")
    counter = _definition("business.empty.counter", MetricType.COUNTER, MetricAggregation.SUM)
    gauge = _definition("business.empty.gauge", MetricType.GAUGE, MetricAggregation.MEAN)
    distribution = _definition(
        "business.empty.distribution",
        MetricType.DISTRIBUTION,
        MetricAggregation.MEAN,
    )
    ratio = _definition("business.empty.ratio", MetricType.RATIO, MetricAggregation.MEAN)
    for definition in (counter, gauge, distribution, ratio):
        await metrics.define(definition)

    start = datetime(2026, 9, 5, 5, 0, tzinfo=timezone.utc)
    window = MetricWindow.between(start, start + timedelta(minutes=1))

    for aggregation in (
        MetricAggregation.COUNT,
        MetricAggregation.SUM,
        MetricAggregation.RATE,
    ):
        result = await metrics.query(
            MetricQuery(counter.name, window, aggregation=aggregation)
        )
        assert result.points[0].value == 0
        assert result.points[0].sample_count == 0

    for metric, aggregation, percentile in (
        (gauge.name, MetricAggregation.MEAN, None),
        (gauge.name, MetricAggregation.LATEST, None),
        (distribution.name, MetricAggregation.PERCENTILE, 0.95),
        (ratio.name, MetricAggregation.MEAN, None),
    ):
        result = await metrics.query(
            MetricQuery(
                metric,
                window,
                aggregation=aggregation,
                percentile=percentile,
            )
        )
        assert result.points[0].value is None
        assert result.points[0].sample_count == 0


@pytest.mark.asyncio
async def test_query_rejects_aggregation_outside_metric_type_matrix() -> None:
    metrics = Metrics.in_memory(namespace="aggregation-matrix")
    counter = _definition(
        "business.matrix.counter",
        MetricType.COUNTER,
        MetricAggregation.SUM,
    )
    await metrics.define(counter)
    start = datetime(2026, 9, 5, 5, 0, tzinfo=timezone.utc)

    with pytest.raises(AIError) as raised:
        await metrics.query(
            MetricQuery(
                counter.name,
                MetricWindow.between(start, start + timedelta(minutes=1)),
                aggregation=MetricAggregation.MEAN,
            )
        )
    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID


@pytest.mark.asyncio
async def test_bucket_query_limit_fails_explicitly_before_scan() -> None:
    metrics = Metrics.in_memory(namespace="bucket-limit")
    counter = _definition(
        "business.bucket.counter",
        MetricType.COUNTER,
        MetricAggregation.SUM,
    )
    await metrics.define(counter)
    start = datetime(2026, 9, 5, 5, 0, tzinfo=timezone.utc)

    with pytest.raises(AIError) as raised:
        await metrics.query(
            MetricQuery(
                counter.name,
                MetricWindow.between(start, start + timedelta(seconds=10_001)),
                bucket=timedelta(seconds=1),
            )
        )
    assert raised.value.code is ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED


@pytest.mark.asyncio
async def test_same_observation_id_is_isolated_by_metrics_namespace() -> None:
    store = InMemoryMetricStore()
    left = Metrics.from_store(store, namespace="namespace-left")
    right = Metrics.from_store(store, namespace="namespace-right")
    definition = _definition(
        "business.namespace.counter",
        MetricType.COUNTER,
        MetricAggregation.SUM,
    )
    await left.define(definition)
    await right.define(definition)
    occurred_at = datetime(2026, 9, 5, 5, 0, tzinfo=timezone.utc)

    await left.record(
        definition.name,
        1,
        observation_id="shared-id",
        occurred_at=occurred_at,
    )
    await right.record(
        definition.name,
        2,
        observation_id="shared-id",
        occurred_at=occurred_at,
    )

    window = MetricWindow.between(
        occurred_at - timedelta(seconds=1),
        occurred_at + timedelta(seconds=1),
    )
    left_result = await left.query(MetricQuery(definition.name, window))
    right_result = await right.query(MetricQuery(definition.name, window))

    assert left_result.points[0].value == 1
    assert left_result.points[0].sample_count == 1
    assert right_result.points[0].value == 2
    assert right_result.points[0].sample_count == 1
