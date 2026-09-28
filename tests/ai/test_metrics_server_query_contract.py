#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exercise driver-bound aggregation through the public Metrics facade."""

import math
from datetime import timedelta

import pytest
from linktools.ai.observe import (
    MetricAggregation, MetricDefinition, MetricQuery, MetricSource, MetricType, Metrics,
)
from linktools.ai.observe._sql import SqlMetricStore
from unittest.mock import AsyncMock

from .test_metrics_server_read_integrity import (
    _NAMESPACE, _ServerMetrics, _WINDOW, _observation,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("percentile", "sample_count"), [
    (percentile, 100) for percentile in (
        1e-12, .07, .14, .28, .5, .56, .58, .95, .99, 1.0,
        math.nextafter(.07, 0.0), math.nextafter(.07, 1.0),
        math.nextafter(.58, 0.0), math.nextafter(.58, 1.0),
        math.nextafter(1.0, 0.0),
    )
] + [
    (percentile, sample_count)
    for sample_count in (0, 1) for percentile in (1e-12, .5, 1.0)
])
async def test_driver_percentile_preserves_binary_nearest_rank(
    server_metrics: _ServerMetrics, monkeypatch: pytest.MonkeyPatch,
    percentile: float, sample_count: int,
) -> None:
    metrics, _, _ = server_metrics
    definition = MetricDefinition(
        name="business.integrity.distribution", revision=1,
        observation_kind="business.integrity", source=MetricSource.measurement("value"),
        metric_type=MetricType.DISTRIBUTION, unit="1",
        default_aggregation=MetricAggregation.MEAN, query_fields=("group",),
    )
    memory = Metrics.in_memory(namespace=_NAMESPACE)
    observations = tuple(_observation(index) for index in range(sample_count))
    for store in (metrics, memory):
        await store.define(definition)
        if observations:
            await store.record_observations(observations)
    monkeypatch.setattr(SqlMetricStore, "scan_observations", AsyncMock(side_effect=AssertionError("unexpected scan fallback")))
    for group_by, bucket in (((), None), (("group",), timedelta(seconds=1))):
        query = MetricQuery(
            definition.name, _WINDOW, aggregation=MetricAggregation.PERCENTILE,
            percentile=percentile, group_by=group_by, bucket=bucket,
            correlation_filters={"attempt": 2**53 + 1},
        )
        expected = await memory.query(query)
        actual = await metrics.query(query)
        assert actual == expected
        if sample_count:
            assert actual.points[0].value == math.ceil(percentile * sample_count)
            if bucket is not None:
                assert actual.points[1].value is None
        else:
            assert all(point.value is None for point in actual.points)


@pytest.mark.asyncio
@pytest.mark.parametrize(("metric_type", "aggregation"), (
    (MetricType.COUNTER, MetricAggregation.COUNT),
    (MetricType.COUNTER, MetricAggregation.SUM),
    (MetricType.COUNTER, MetricAggregation.RATE),
    (MetricType.GAUGE, MetricAggregation.COUNT),
    (MetricType.GAUGE, MetricAggregation.MEAN),
    (MetricType.GAUGE, MetricAggregation.MIN),
    (MetricType.GAUGE, MetricAggregation.MAX),
    (MetricType.GAUGE, MetricAggregation.LATEST),
    (MetricType.DISTRIBUTION, MetricAggregation.MEAN),
))
async def test_driver_measurement_reduction_preserves_partition_results(
    server_metrics: _ServerMetrics, monkeypatch: pytest.MonkeyPatch,
    metric_type: MetricType, aggregation: MetricAggregation,
) -> None:
    metrics, _, _ = server_metrics
    definition = MetricDefinition(
        name="business.integrity.driver", revision=1,
        observation_kind="business.integrity", source=MetricSource.measurement("value"),
        metric_type=metric_type, unit="1", default_aggregation=MetricAggregation.COUNT,
        query_fields=("group",),
    )
    memory = Metrics.in_memory(namespace=_NAMESPACE)
    observations = tuple(_observation(index) for index in range(5))
    for store in (metrics, memory):
        await store.define(definition)
        await store.record_observations(observations)
    query = MetricQuery(
        definition.name, _WINDOW, aggregation=aggregation,
        group_by=("group",), bucket=timedelta(seconds=1),
    )
    monkeypatch.setattr(SqlMetricStore, "scan_observations", AsyncMock(side_effect=AssertionError("unexpected scan fallback")))
    assert await metrics.query(query) == await memory.query(query)
