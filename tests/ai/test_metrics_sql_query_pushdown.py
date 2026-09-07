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
    MetricType,
    MetricWindow,
    Metrics,
    Observation,
)
from linktools.ai.observe._sql_query import _percentile_pick_sql
from sqlalchemy import event
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
async def test_sql_pushdown_bounds_observation_select_round_trips(tmp_path: Path) -> None:
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
            for index in range(1500)
        )
        for offset in range(0, len(observations), 256):
            await metrics.record_observations(observations[offset : offset + 256])

        event.listen(engine.sync_engine, "before_cursor_execute", before_cursor_execute)
        window = MetricWindow.between(
            occurred_at - timedelta(seconds=1),
            occurred_at + timedelta(seconds=1),
        )

        await metrics.query(MetricQuery("business.request.count", window))
        assert len(statements) == 1

        statements.clear()
        await metrics.query(
            MetricQuery(
                "business.request.count",
                window,
                group_by=("route",),
            )
        )
        assert len(statements) == 1

        statements.clear()
        await metrics.query(
            MetricQuery(
                "business.request.latency",
                window,
                aggregation=MetricAggregation.PERCENTILE,
                percentile=0.95,
            )
        )
        assert len(statements) == 1
    finally:
        if event.contains(engine.sync_engine, "before_cursor_execute", before_cursor_execute):
            event.remove(engine.sync_engine, "before_cursor_execute", before_cursor_execute)
        await engine.dispose()


def test_percentile_strategy_uses_native_only_when_semantics_match() -> None:
    postgresql = _percentile_pick_sql("postgresql", [])
    mysql = _percentile_pick_sql("mysql", [])
    sqlite = _percentile_pick_sql("sqlite", [])

    assert "percentile_disc" in postgresql
    assert "row_number()" in mysql.lower()
    assert "row_number()" in sqlite.lower()
    assert "percentile_disc" not in mysql
    assert "percentile_disc" not in sqlite
