#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metric default aggregation semantics."""

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


def _distribution(name: str, default: MetricAggregation) -> MetricDefinition:
    return MetricDefinition(
        name=name,
        revision=1,
        observation_kind=f"{name}.sample",
        source=MetricSource.measurement("value"),
        metric_type=MetricType.DISTRIBUTION,
        unit="1",
        default_aggregation=default,
    )


@pytest.mark.asyncio
async def test_default_percentile_uses_query_percentile() -> None:
    metrics = Metrics.in_memory(namespace="default-percentile")
    definition = _distribution(
        "business.default.percentile",
        MetricAggregation.PERCENTILE,
    )
    await metrics.define(definition)
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)
    for index, value in enumerate((1, 2, 3), start=1):
        await metrics.record(
            definition.name,
            value,
            observation_id=f"sample-{index}",
            occurred_at=start + timedelta(seconds=index),
        )

    result = await metrics.query(
        MetricQuery(
            definition.name,
            MetricWindow.between(start, start + timedelta(minutes=1)),
            percentile=0.5,
        )
    )

    assert result.aggregation is MetricAggregation.PERCENTILE
    assert result.points[0].value == 2


@pytest.mark.asyncio
async def test_percentile_parameter_requires_percentile_aggregation() -> None:
    metrics = Metrics.in_memory(namespace="invalid-default-percentile")
    definition = _distribution("business.default.mean", MetricAggregation.MEAN)
    await metrics.define(definition)
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)

    with pytest.raises(AIError) as raised:
        await metrics.query(
            MetricQuery(
                definition.name,
                MetricWindow.between(start, start + timedelta(minutes=1)),
                percentile=0.95,
            )
        )

    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
