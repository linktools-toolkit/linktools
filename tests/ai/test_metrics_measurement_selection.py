#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measurement names remain case-sensitive through SQL extraction."""

from dataclasses import replace

import pytest

from linktools.ai.observe import (
    MetricAggregation, MetricDefinition, MetricMeasurement, MetricQuery,
    MetricSource, MetricType, Metrics,
)

from .test_metrics_server_read_integrity import (
    _NAMESPACE, _ServerMetrics, _WINDOW, _observation, _server, server_metrics,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("aggregation", (
    MetricAggregation.COUNT, MetricAggregation.MEAN, MetricAggregation.MIN,
    MetricAggregation.MAX, MetricAggregation.PERCENTILE,
))
async def test_server_measurement_names_are_distinct(
    server_metrics: _ServerMetrics, aggregation: MetricAggregation,
) -> None:
    metrics, _, _ = server_metrics
    memory = Metrics.in_memory(namespace=_NAMESPACE)
    definition = MetricDefinition(
        name="business.integrity.selection", revision=1,
        observation_kind="business.integrity", source=MetricSource.measurement("value"),
        metric_type=MetricType.DISTRIBUTION, unit="1",
        default_aggregation=MetricAggregation.MEAN,
    )
    record = replace(_observation(), measurements=(
        MetricMeasurement("value", 1, 7),
        MetricMeasurement("Value", 1, 900),
        MetricMeasurement("value", 2, 300),
    ))
    for facade in (metrics, memory):
        await facade.define(definition)
        await facade.record_observations((record,))
    query = MetricQuery(
        definition.name, _WINDOW, aggregation=aggregation,
        percentile=.95 if aggregation is MetricAggregation.PERCENTILE else None,
    )
    assert await metrics.query(query) == await memory.query(query)
