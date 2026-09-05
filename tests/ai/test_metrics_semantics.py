#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metric revision, query, percentile, and retention semantics."""

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
from linktools.ai.observe._codec import observation_digest
from linktools.ai.observe._memory import InMemoryMetricStore
from sqlalchemy.ext.asyncio import create_async_engine


def _measurement_definition(
    name: str,
    *,
    kind: str,
    measurement: str,
    metric_type: MetricType,
    aggregation: MetricAggregation,
    revision: int = 1,
    measurement_revision: int = 1,
) -> MetricDefinition:
    return MetricDefinition(
        name=name,
        revision=revision,
        observation_kind=kind,
        source=MetricSource.measurement(
            measurement,
            revision=measurement_revision,
        ),
        metric_type=metric_type,
        unit="1",
        default_aggregation=aggregation,
        query_fields=("group", "status"),
    )


@pytest.mark.asyncio
async def test_window_rate_latest_and_custom_accuracy_semantics() -> None:
    metrics = Metrics.in_memory(namespace="query-semantics")
    counter = _measurement_definition(
        "business.counter",
        kind="business.counter.sample",
        measurement="value",
        metric_type=MetricType.COUNTER,
        aggregation=MetricAggregation.SUM,
    )
    latest = _measurement_definition(
        "business.latest",
        kind="business.latest.sample",
        measurement="value",
        metric_type=MetricType.GAUGE,
        aggregation=MetricAggregation.LATEST,
    )
    accuracy = _measurement_definition(
        "agent.accuracy",
        kind="agent.accuracy.sample",
        measurement="accuracy",
        metric_type=MetricType.RATIO,
        aggregation=MetricAggregation.MEAN,
    )
    for definition in (counter, latest, accuracy):
        await metrics.define(definition)

    start = datetime(2026, 9, 5, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(seconds=10)
    await metrics.record(
        "business.counter",
        10,
        observation_id="counter-start",
        occurred_at=start,
    )
    await metrics.record(
        "business.counter",
        20,
        observation_id="counter-middle",
        occurred_at=start + timedelta(seconds=5),
    )
    await metrics.record(
        "business.counter",
        30,
        observation_id="counter-end",
        occurred_at=end,
    )

    summed = await metrics.query(
        MetricQuery(
            "business.counter",
            MetricWindow.between(start, end),
        )
    )
    assert summed.points[0].value == 30
    assert summed.points[0].sample_count == 2

    rate = await metrics.query(
        MetricQuery(
            "business.counter",
            MetricWindow.between(start, end),
            aggregation=MetricAggregation.RATE,
        )
    )
    assert rate.unit == "1/s"
    assert rate.points[0].value == 3
    assert rate.points[0].sample_count == 2

    tie_time = start + timedelta(seconds=1)
    await metrics.record(
        "business.latest",
        11,
        observation_id="latest-a",
        occurred_at=tie_time,
    )
    await metrics.record(
        "business.latest",
        22,
        observation_id="latest-b",
        occurred_at=tie_time,
    )
    latest_result = await metrics.query(
        MetricQuery(
            "business.latest",
            MetricWindow.between(start, end),
        )
    )
    expected_latest = max(
        (
            (observation_digest(metrics.namespace, "latest-a"), 11),
            (observation_digest(metrics.namespace, "latest-b"), 22),
        )
    )[1]
    assert latest_result.points[0].value == expected_latest
    assert latest_result.points[0].sample_count == 2

    for index, value in enumerate((1, 1, 0, 1)):
        await metrics.record(
            "agent.accuracy",
            value,
            observation_id=f"accuracy-{index}",
            occurred_at=start + timedelta(seconds=index),
        )
    accuracy_result = await metrics.query(
        MetricQuery(
            "agent.accuracy",
            MetricWindow.between(start, end),
        )
    )
    assert accuracy_result.points[0].value == 0.75
    assert accuracy_result.points[0].sample_count == 4


@pytest.mark.asyncio
async def test_latest_metric_revision_does_not_mix_measurement_revisions() -> None:
    metrics = Metrics.in_memory(namespace="revision-semantics")
    v1 = _measurement_definition(
        "business.revisioned",
        kind="business.revisioned.sample",
        measurement="score",
        metric_type=MetricType.GAUGE,
        aggregation=MetricAggregation.MEAN,
        revision=1,
        measurement_revision=1,
    )
    v2 = _measurement_definition(
        "business.revisioned",
        kind="business.revisioned.sample",
        measurement="score",
        metric_type=MetricType.GAUGE,
        aggregation=MetricAggregation.MEAN,
        revision=2,
        measurement_revision=2,
    )
    await metrics.define(v1)
    await metrics.define(v2)
    occurred_at = datetime(2026, 9, 5, 1, 0, tzinfo=timezone.utc)
    await metrics.record_observations(
        (
            Observation(
                version=1,
                observation_id="revision-sample",
                kind="business.revisioned.sample",
                occurred_at=occurred_at,
                source_namespace=None,
                tenant_id=None,
                status=None,
                error_code=None,
                correlation={},
                dimensions={},
                measurements=(
                    MetricMeasurement("score", 1, 10),
                    MetricMeasurement("score", 2, 20),
                ),
            ),
        )
    )
    window = MetricWindow.between(
        occurred_at - timedelta(seconds=1),
        occurred_at + timedelta(seconds=1),
    )

    latest = await metrics.query(MetricQuery("business.revisioned", window))
    old = await metrics.query(
        MetricQuery("business.revisioned", window, revision=1)
    )

    assert latest.revision == 2
    assert latest.points[0].value == 20
    assert latest.points[0].sample_count == 1
    assert old.revision == 1
    assert old.points[0].value == 10
    assert old.points[0].sample_count == 1


@pytest.mark.asyncio
async def test_definition_identity_is_idempotent_and_semantic_change_conflicts() -> None:
    metrics = Metrics.in_memory(namespace="definition-semantics")
    base = _measurement_definition(
        "business.definition",
        kind="business.definition.sample",
        measurement="value",
        metric_type=MetricType.GAUGE,
        aggregation=MetricAggregation.MEAN,
    )
    first = await metrics.define(base)
    replay = await metrics.define(base)
    assert replay == first == base

    conflict = MetricDefinition(
        name=base.name,
        revision=base.revision,
        observation_kind=base.observation_kind,
        source=MetricSource.measurement("other"),
        metric_type=base.metric_type,
        unit=base.unit,
        default_aggregation=base.default_aggregation,
        query_fields=base.query_fields,
    )
    with pytest.raises(AIError) as raised:
        await metrics.define(conflict)
    assert raised.value.code is ErrorCode.STORAGE_CONFLICT


@pytest.mark.asyncio
async def test_prune_is_namespace_scoped_and_preserves_definitions() -> None:
    store = InMemoryMetricStore()
    metrics_a = Metrics.from_store(store, namespace="retention-a")
    metrics_b = Metrics.from_store(store, namespace="retention-b")
    definition = _measurement_definition(
        "business.retained",
        kind="business.retained.sample",
        measurement="value",
        metric_type=MetricType.COUNTER,
        aggregation=MetricAggregation.SUM,
    )
    await metrics_a.define(definition)
    await metrics_b.define(definition)
    boundary = datetime(2026, 9, 5, 2, 0, tzinfo=timezone.utc)
    old = boundary - timedelta(seconds=1)
    new = boundary + timedelta(seconds=1)
    await metrics_a.record(
        definition.name,
        1,
        observation_id="a-old",
        occurred_at=old,
    )
    await metrics_a.record(
        definition.name,
        1,
        observation_id="a-new",
        occurred_at=new,
    )
    await metrics_b.record(
        definition.name,
        1,
        observation_id="b-old",
        occurred_at=old,
    )

    assert await metrics_a.prune(before=boundary) == 1
    window = MetricWindow.between(old - timedelta(seconds=1), new + timedelta(seconds=1))
    a_result = await metrics_a.query(MetricQuery(definition.name, window))
    b_result = await metrics_b.query(MetricQuery(definition.name, window))

    assert a_result.points[0].value == 1
    assert a_result.points[0].sample_count == 1
    assert b_result.points[0].value == 1
    assert b_result.points[0].sample_count == 1


@pytest.mark.asyncio
async def test_exact_percentiles_match_memory_sqlite_and_sql(tmp_path: Path) -> None:
    definition = _measurement_definition(
        "business.latency",
        kind="business.latency.sample",
        measurement="latency",
        metric_type=MetricType.DISTRIBUTION,
        aggregation=MetricAggregation.MEAN,
    )
    occurred_at = datetime(2026, 9, 5, 3, 0, tzinfo=timezone.utc)
    values = (1, 2, 3, 4, 100)

    memory = Metrics.in_memory(namespace="percentile-memory")

    sqlite_path = tmp_path / "percentile-sqlite.db"
    sqlite_engine = create_async_engine(f"sqlite+aiosqlite:///{sqlite_path}")
    sqlite = Metrics.sqlite(sqlite_path, namespace="percentile-sqlite")

    sql_path = tmp_path / "percentile-sql.db"
    sql_engine = create_async_engine(f"sqlite+aiosqlite:///{sql_path}")
    sql = Metrics.sql(sql_engine, namespace="percentile-sql")

    try:
        await provision_metrics_database(sqlite_engine)
        await provision_metrics_database(sql_engine)
        for metrics in (memory, sqlite, sql):
            await metrics.define(definition)
            for index, value in enumerate(values):
                await metrics.record(
                    definition.name,
                    value,
                    observation_id=f"sample-{index}",
                    occurred_at=occurred_at + timedelta(microseconds=index),
                )

        window = MetricWindow.between(
            occurred_at - timedelta(seconds=1),
            occurred_at + timedelta(seconds=1),
        )
        expected = {0.50: 3, 0.95: 100, 0.99: 100}
        for percentile, value in expected.items():
            observed = []
            for metrics in (memory, sqlite, sql):
                result = await metrics.query(
                    MetricQuery(
                        definition.name,
                        window,
                        aggregation=MetricAggregation.PERCENTILE,
                        percentile=percentile,
                    )
                )
                observed.append(result.points[0].value)
                assert result.points[0].sample_count == len(values)
            assert observed == [value, value, value]
    finally:
        await sqlite_engine.dispose()
        await sql_engine.dispose()
