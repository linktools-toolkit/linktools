#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL measurement aggregation stays database-side and matches scan semantics."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_metrics_database
from linktools.ai.observe import (
    MetricAggregation,
    MetricDefinition,
    MetricMeasurement,
    MetricQuery,
    MetricSource,
    MetricType,
    MetricWindow,
    Metrics,
    Observation,
)
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine


def _definition(
    name: str,
    measurement: str,
    metric_type: MetricType,
    default_aggregation: MetricAggregation,
) -> MetricDefinition:
    return MetricDefinition(
        name=name,
        revision=1,
        observation_kind="business.measurement",
        source=MetricSource.measurement(measurement),
        metric_type=metric_type,
        unit="1",
        default_aggregation=default_aggregation,
        query_fields=("group",),
    )


def _definitions() -> tuple[MetricDefinition, ...]:
    return (
        _definition(
            "business.measurement.counter",
            "counter",
            MetricType.COUNTER,
            MetricAggregation.SUM,
        ),
        _definition(
            "business.measurement.gauge",
            "gauge",
            MetricType.GAUGE,
            MetricAggregation.MEAN,
        ),
        _definition(
            "business.measurement.distribution",
            "distribution",
            MetricType.DISTRIBUTION,
            MetricAggregation.MEAN,
        ),
        _definition(
            "business.measurement.ratio",
            "ratio",
            MetricType.RATIO,
            MetricAggregation.MEAN,
        ),
    )


def _observation(
    observation_id: str,
    occurred_at: datetime,
    *,
    group: str,
    counter: int,
    gauge: int | float,
    distribution: int | float,
    ratio: int | float,
) -> Observation:
    return Observation(
        version=1,
        observation_id=observation_id,
        kind="business.measurement",
        occurred_at=occurred_at,
        source_namespace="workspace",
        tenant_id="tenant",
        status="SUCCEEDED",
        error_code=None,
        correlation={},
        dimensions={"group": group},
        measurements=(
            MetricMeasurement("counter", 1, counter),
            MetricMeasurement("gauge", 1, gauge),
            MetricMeasurement("distribution", 1, distribution),
            MetricMeasurement("ratio", 1, ratio),
        ),
    )


async def _seed(metrics: Metrics, observations: tuple[Observation, ...]) -> None:
    for definition in _definitions():
        await metrics.define(definition)
    for offset in range(0, len(observations), 256):
        await metrics.record_observations(observations[offset : offset + 256])


def _queries(window: MetricWindow) -> tuple[MetricQuery, ...]:
    common = {
        "window": window,
        "group_by": ("group",),
        "bucket": timedelta(seconds=1),
    }
    return (
        MetricQuery(
            "business.measurement.counter",
            aggregation=MetricAggregation.COUNT,
            **common,
        ),
        MetricQuery(
            "business.measurement.counter",
            aggregation=MetricAggregation.SUM,
            **common,
        ),
        MetricQuery(
            "business.measurement.counter",
            aggregation=MetricAggregation.RATE,
            **common,
        ),
        MetricQuery(
            "business.measurement.gauge",
            aggregation=MetricAggregation.COUNT,
            **common,
        ),
        MetricQuery(
            "business.measurement.gauge",
            aggregation=MetricAggregation.MEAN,
            **common,
        ),
        MetricQuery(
            "business.measurement.gauge",
            aggregation=MetricAggregation.MIN,
            **common,
        ),
        MetricQuery(
            "business.measurement.gauge",
            aggregation=MetricAggregation.MAX,
            **common,
        ),
        MetricQuery(
            "business.measurement.gauge",
            aggregation=MetricAggregation.LATEST,
            **common,
        ),
        MetricQuery(
            "business.measurement.distribution",
            aggregation=MetricAggregation.COUNT,
            **common,
        ),
        MetricQuery(
            "business.measurement.distribution",
            aggregation=MetricAggregation.MEAN,
            **common,
        ),
        MetricQuery(
            "business.measurement.distribution",
            aggregation=MetricAggregation.MIN,
            **common,
        ),
        MetricQuery(
            "business.measurement.distribution",
            aggregation=MetricAggregation.MAX,
            **common,
        ),
        MetricQuery(
            "business.measurement.distribution",
            aggregation=MetricAggregation.PERCENTILE,
            percentile=0.95,
            **common,
        ),
        MetricQuery(
            "business.measurement.ratio",
            aggregation=MetricAggregation.COUNT,
            **common,
        ),
        MetricQuery(
            "business.measurement.ratio",
            aggregation=MetricAggregation.MEAN,
            **common,
        ),
    )


@pytest.mark.asyncio
async def test_every_measurement_aggregation_matches_backend_neutral_semantics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "all-measurements.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    memory = Metrics.in_memory(namespace="measurement-pushdown")
    sql = Metrics.sql(engine, namespace="measurement-pushdown")
    start = datetime(2026, 9, 7, 3, 0, tzinfo=timezone.utc)
    observations = (
        _observation(
            "a",
            start,
            group="alpha",
            counter=1,
            gauge=10,
            distribution=1.5,
            ratio=0.25,
        ),
        _observation(
            "b",
            start + timedelta(microseconds=999_999),
            group="alpha",
            counter=2,
            gauge=20.5,
            distribution=2.5,
            ratio=0.5,
        ),
        _observation(
            "c",
            start + timedelta(seconds=1),
            group="alpha",
            counter=3,
            gauge=30,
            distribution=3.5,
            ratio=0.75,
        ),
        _observation(
            "d",
            start + timedelta(seconds=1),
            group="alpha",
            counter=4,
            gauge=40,
            distribution=100,
            ratio=1,
        ),
        _observation(
            "e",
            start + timedelta(seconds=1, microseconds=1),
            group="beta",
            counter=5,
            gauge=-5,
            distribution=-1.25,
            ratio=0,
        ),
    )
    try:
        await provision_metrics_database(engine)
        await _seed(memory, observations)
        await _seed(sql, observations)
        window = MetricWindow.between(start, start + timedelta(seconds=2))

        for query in _queries(window):
            assert await sql.query(query) == await memory.query(query)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metric_type", "value"),
    (
        (MetricType.COUNTER, -1),
        (MetricType.RATIO, 1.5),
    ),
)
async def test_measurement_pushdown_preserves_metric_type_integrity_checks(
    tmp_path: Path,
    metric_type: MetricType,
    value: int | float,
) -> None:
    path = tmp_path / f"invalid-{metric_type.value.lower()}.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    definition = _definition(
        f"business.invalid.{metric_type.value.lower()}",
        "value",
        metric_type,
        MetricAggregation.SUM
        if metric_type is MetricType.COUNTER
        else MetricAggregation.MEAN,
    )
    memory = Metrics.in_memory(namespace="measurement-integrity")
    sql = Metrics.sql(engine, namespace="measurement-integrity")
    occurred_at = datetime(2026, 9, 7, 4, 0, tzinfo=timezone.utc)
    observation = Observation(
        version=1,
        observation_id="invalid",
        kind="business.measurement",
        occurred_at=occurred_at,
        source_namespace="workspace",
        tenant_id="tenant",
        status=None,
        error_code=None,
        correlation={},
        dimensions={},
        measurements=(MetricMeasurement("value", 1, value),),
    )
    try:
        await provision_metrics_database(engine)
        for metrics in (memory, sql):
            await metrics.define(definition)
            await metrics.record_observations((observation,))

        query = MetricQuery(
            definition.name,
            MetricWindow.between(
                occurred_at - timedelta(seconds=1),
                occurred_at + timedelta(seconds=1),
            ),
            aggregation=MetricAggregation.COUNT,
        )
        for metrics in (memory, sql):
            with pytest.raises(AIError) as raised:
                await metrics.query(query)
            assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_every_measurement_aggregation_uses_bounded_sql_round_trips(
    tmp_path: Path,
) -> None:
    path = tmp_path / "measurement-statements.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    metrics = Metrics.sql(engine, namespace="measurement-statements")
    start = datetime(2026, 9, 7, 5, 0, tzinfo=timezone.utc)
    observations = tuple(
        _observation(
            f"sample-{index}",
            start + timedelta(microseconds=index),
            group="alpha",
            counter=index + 1,
            gauge=index + 0.5,
            distribution=index + 0.25,
            ratio=(index % 4) / 4,
        )
        for index in range(1500)
    )
    statements: list[str] = []

    def before_cursor_execute(
        _conn: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        normalized = statement.lower()
        if (
            "ai_metric_observations" in normalized
            and normalized.lstrip().startswith(("select", "with"))
        ):
            statements.append(normalized)

    try:
        await provision_metrics_database(engine)
        await _seed(metrics, observations)
        event.listen(engine.sync_engine, "before_cursor_execute", before_cursor_execute)
        window = MetricWindow.between(start, start + timedelta(seconds=1))

        for query in _queries(window):
            statements.clear()
            await metrics.query(query)
            assert len(statements) == 2
            assert all(
                " limit ?" in statement or "with filtered" in statement
                for statement in statements
            )
    finally:
        if event.contains(
            engine.sync_engine,
            "before_cursor_execute",
            before_cursor_execute,
        ):
            event.remove(
                engine.sync_engine,
                "before_cursor_execute",
                before_cursor_execute,
            )
        await engine.dispose()
