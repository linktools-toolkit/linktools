#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metrics SQL query pushdown must preserve query semantics and bound I/O."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from linktools.ai.migrate import provision_metrics_database
from linktools.ai.observe import (
    MetricAggregation,
    MetricDefinition,
    MetricMeasurement,
    MetricQuery,
    MetricSource,
    MetricSourceKind,
    MetricType,
    MetricWindow,
    Metrics,
    Observation,
)
from linktools.ai.observe._sql_query import _base_params, _measurement_sql
from linktools.ai.observe._store import _MetricQueryPushdownPlan
from sqlalchemy import event, text
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.ext.asyncio import create_async_engine


def _count_definition() -> MetricDefinition:
    return MetricDefinition(
        name="business.request.count",
        revision=1,
        observation_kind="business.request",
        source=MetricSource.observation_count(),
        metric_type=MetricType.COUNTER,
        unit="1",
        default_aggregation=MetricAggregation.SUM,
        query_fields=("status", "route"),
    )


def _failure_ratio_definition() -> MetricDefinition:
    return MetricDefinition(
        name="business.request.failure_ratio",
        revision=1,
        observation_kind="business.request",
        source=MetricSource.indicator("status", ("FAILED",)),
        metric_type=MetricType.RATIO,
        unit="1",
        default_aggregation=MetricAggregation.MEAN,
        query_fields=("status", "route"),
    )


def _latency_definition() -> MetricDefinition:
    return MetricDefinition(
        name="business.request.latency",
        revision=1,
        observation_kind="business.request",
        source=MetricSource.measurement("latency_ms"),
        metric_type=MetricType.DISTRIBUTION,
        unit="ms",
        default_aggregation=MetricAggregation.MEAN,
        query_fields=("status", "route"),
    )


def _observation(
    observation_id: str,
    occurred_at: datetime,
    *,
    route: str,
    status: str,
    correlation_value: str | int,
    latency: int | float | None = None,
) -> Observation:
    measurements = (
        ()
        if latency is None
        else (MetricMeasurement("latency_ms", 1, latency),)
    )
    return Observation(
        version=1,
        observation_id=observation_id,
        kind="business.request",
        occurred_at=occurred_at,
        source_namespace="workspace",
        tenant_id="tenant",
        status=status,
        error_code=None,
        correlation={"attempt": correlation_value},
        dimensions={"route": route},
        measurements=measurements,
    )


async def _seed(
    memory: Metrics,
    sql: Metrics,
    observations: tuple[Observation, ...],
) -> None:
    definitions = (
        _count_definition(),
        _failure_ratio_definition(),
        _latency_definition(),
    )
    for metrics in (memory, sql):
        for definition in definitions:
            await metrics.define(definition)
        await metrics.record_observations(observations)


@pytest.mark.asyncio
async def test_sql_pushdown_matches_memory_for_count_ratio_filters_groups_and_buckets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pushdown.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    memory = Metrics.in_memory(namespace="memory")
    sql = Metrics.sql(engine, namespace="sql")
    start = datetime(2026, 9, 7, 0, 0, tzinfo=timezone.utc)
    observations = (
        _observation("a", start, route="alpha", status="SUCCEEDED", correlation_value=1),
        _observation(
            "b",
            start + timedelta(microseconds=999_999),
            route="alpha",
            status="FAILED",
            correlation_value=1,
        ),
        _observation(
            "c",
            start + timedelta(seconds=1),
            route="beta",
            status="FAILED",
            correlation_value=1,
        ),
        _observation(
            "d",
            start + timedelta(seconds=1, microseconds=1),
            route="beta",
            status="SUCCEEDED",
            correlation_value="1",
        ),
    )
    try:
        await provision_metrics_database(engine)
        await _seed(memory, sql, observations)
        window = MetricWindow.between(start, start + timedelta(seconds=2))

        queries = (
            MetricQuery(
                "business.request.count",
                window,
                correlation_filters={"attempt": 1},
                group_by=("route",),
                bucket=timedelta(seconds=1),
            ),
            MetricQuery(
                "business.request.failure_ratio",
                window,
                correlation_filters={"attempt": 1},
                group_by=("route",),
                bucket=timedelta(seconds=1),
            ),
        )
        for query in queries:
            assert await sql.query(query) == await memory.query(query)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_sql_pushdown_percentile_matches_nearest_rank_with_groups_and_buckets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "percentile.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    memory = Metrics.in_memory(namespace="memory-percentile")
    sql = Metrics.sql(engine, namespace="sql-percentile")
    start = datetime(2026, 9, 7, 1, 0, tzinfo=timezone.utc)
    observations = []
    for bucket, route in enumerate(("alpha", "beta")):
        for index, latency in enumerate((1, 2, 3, 4, 100)):
            observations.append(
                _observation(
                    f"{route}-{index}",
                    start + timedelta(seconds=bucket, microseconds=index),
                    route=route,
                    status="SUCCEEDED",
                    correlation_value=1,
                    latency=latency,
                )
            )
    try:
        await provision_metrics_database(engine)
        await _seed(memory, sql, tuple(observations))
        query = MetricQuery(
            "business.request.latency",
            MetricWindow.between(start, start + timedelta(seconds=2)),
            aggregation=MetricAggregation.PERCENTILE,
            percentile=0.95,
            group_by=("route",),
            bucket=timedelta(seconds=1),
        )

        result = await sql.query(query)
        assert result == await memory.query(query)
        nonempty = [point for point in result.points if point.sample_count]
        assert [point.value for point in nonempty] == [100, 100]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("sample_count", (
    3,
    pytest.param(1500, marks=pytest.mark.manual, id="stress-1500"),
))
async def test_sql_pushdown_bounds_observation_select_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_count: int,
) -> None:
    path = tmp_path / "statements.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    metrics = Metrics.sql(engine, namespace="statements")
    occurred_at = datetime(2026, 9, 7, 2, 0, tzinfo=timezone.utc)
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
        if "ai_metric_observations" in normalized and normalized.lstrip().startswith(("select", "with")):
            statements.append(normalized)

    try:
        await provision_metrics_database(engine)
        await metrics.define(_count_definition())
        await metrics.define(_latency_definition())
        observations = tuple(
            _observation(
                f"sample-{index}",
                occurred_at + timedelta(microseconds=index),
                route="alpha",
                status="SUCCEEDED",
                correlation_value=1,
                latency=index + 1,
            )
            for index in range(sample_count)
        )
        for offset in range(0, len(observations), 256):
            await metrics.record_observations(observations[offset : offset + 256])

        async def reject_client_scan(*args, **kwargs):
            raise AssertionError("SQL aggregation must not materialize raw observation pages")

        monkeypatch.setattr(metrics._store, "scan_observations", reject_client_scan)
        event.listen(engine.sync_engine, "before_cursor_execute", before_cursor_execute)
        window = MetricWindow.between(
            occurred_at - timedelta(seconds=1),
            occurred_at + timedelta(seconds=1),
        )

        result = await metrics.query(MetricQuery("business.request.count", window))
        assert result.points[0].value == sample_count
        assert len(statements) == 1

        statements.clear()
        result = await metrics.query(
            MetricQuery(
                "business.request.count",
                window,
                group_by=("route",),
            )
        )
        assert result.points[0].value == sample_count
        assert len(statements) == 1

        statements.clear()
        result = await metrics.query(
            MetricQuery(
                "business.request.latency",
                window,
                aggregation=MetricAggregation.PERCENTILE,
                percentile=0.95,
            )
        )
        assert result.points[0].value == (95 * sample_count + 99) // 100
        assert len(statements) == 1
    finally:
        if event.contains(engine.sync_engine, "before_cursor_execute", before_cursor_execute):
            event.remove(engine.sync_engine, "before_cursor_execute", before_cursor_execute)
        await engine.dispose()


@pytest.mark.parametrize("dialect_name", ("mysql", "postgresql", "sqlite"))
def test_measurement_queries_compile_for_supported_sql_dialects(dialect_name: str) -> None:
    dialect = {"mysql": mysql, "postgresql": postgresql, "sqlite": sqlite}[dialect_name].dialect()
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)
    for aggregation in (MetricAggregation.SUM, MetricAggregation.MEAN, MetricAggregation.PERCENTILE):
        plan = _MetricQueryPushdownPlan(
            observation_kind="business.request", source_kind=MetricSourceKind.MEASUREMENT,
            metric_type=MetricType.COUNTER if aggregation is MetricAggregation.SUM else MetricType.DISTRIBUTION,
            measurement_name="latency_ms", measurement_revision=1,
            indicator_field=None, indicator_values=(), aggregation=aggregation,
            percentile=0.95 if aggregation is MetricAggregation.PERCENTILE else None,
            start=start, end=start + timedelta(seconds=2),
            filters=(("status", "SUCCEEDED"),), correlation_filters=(("attempt", 2**53 + 1),),
            group_by=("route",), bucket_microseconds=1_000_000, bucket_count=2,
            max_scanned_observations=100000, max_extracted_samples=100000,
            max_groups=256, max_result_points=16384,
        )
        statement, params = _measurement_sql(
            dialect_name, plan, _base_params("ns", dialect_name, plan),
        )
        query = text(statement)
        assert set(query.compile(dialect=dialect).params) == params.keys()
        query.bindparams(**params).compile(
            dialect=dialect, compile_kwargs={"literal_binds": True},
        )
