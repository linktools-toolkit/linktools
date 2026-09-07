#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL query limits, capability selection, and bounded execution plans."""

from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from linktools.ai.errors import AIError, ErrorCode
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
from linktools.ai.observe import _query
from linktools.ai.observe._sql import SqlMetricStore
from linktools.ai.observe._sql_query import _sql_features_available
from linktools.ai.observe._store import _MetricQueryPushdownPlan
from sqlalchemy import event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

_START = datetime(2026, 9, 7, tzinfo=timezone.utc)
_NAMES = ("business.review.count", "business.review.ratio", "business.review.value")
_SqlMetrics = tuple[Metrics, AsyncEngine, MetricWindow]


@pytest_asyncio.fixture
async def sql_metrics(tmp_path: Path) -> AsyncIterator[_SqlMetrics]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'limits.db'}")
    metrics = Metrics.sql(engine, namespace="query-limits")
    sources = (
        (MetricSource.observation_count(), MetricType.COUNTER),
        (MetricSource.indicator("status", ("FAILED",)), MetricType.RATIO),
        (MetricSource.measurement("value"), MetricType.DISTRIBUTION),
    )
    try:
        await provision_metrics_database(engine)
        for name, (source, metric_type) in zip(_NAMES, sources, strict=True):
            await metrics.define(MetricDefinition(
                name=name, revision=1, observation_kind="business.review",
                source=source, metric_type=metric_type, unit="1",
                default_aggregation=MetricAggregation.COUNT,
                query_fields=("group", "status"),
            ))
        yield metrics, engine, MetricWindow.between(_START, _START + timedelta(seconds=2))
    finally:
        await engine.dispose()


async def _seed(metrics: Metrics, count: int, *, offset: int = 0, groups: int = 1) -> None:
    observations = tuple(
        Observation(
            version=1, observation_id=f"sample-{i}", kind="business.review",
            occurred_at=_START + timedelta(microseconds=i),
            source_namespace=None, tenant_id=None, status="SUCCEEDED", error_code=None,
            correlation={}, dimensions={"group": f"g{i % groups}"},
            measurements=(MetricMeasurement("value", 1, i),),
        )
        for i in range(offset, offset + count)
    )
    for start in range(0, count, 256):
        await metrics.record_observations(observations[start:start + 256])


@pytest.mark.asyncio
@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize("count", (0, 3, 4))
async def test_scan_limit_is_applied_before_filters(
    sql_metrics: _SqlMetrics, monkeypatch: pytest.MonkeyPatch, name: str, count: int,
) -> None:
    metrics, _, window = sql_metrics
    await _seed(metrics, count)
    monkeypatch.setattr(_query, "_MAX_SCANNED_OBSERVATIONS", 3)
    query = MetricQuery(name, window, filters={"status": "FAILED"})
    if count > 3:
        with pytest.raises(AIError) as raised:
            await metrics.query(query)
        assert raised.value.code is ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED
    else:
        result = await metrics.query(query)
        assert result.points[0].sample_count == 0


@pytest.mark.asyncio
async def test_concurrent_insert_before_aggregate_cannot_bypass_scan_limit(
    sql_metrics: _SqlMetrics, monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics, _, window = sql_metrics
    await _seed(metrics, 2)
    monkeypatch.setattr(_query, "_MAX_SCANNED_OBSERVATIONS", 3)
    execute = AsyncSession.execute
    inserted = False

    async def execute_with_writer(session: AsyncSession, statement: object, *args: object, **kwargs: object) -> object:
        nonlocal inserted
        sql = str(statement).lstrip().lower()
        if not inserted and sql.startswith("with ") and "ai_metric_observations" in sql:
            inserted = True
            await _seed(metrics, 4, offset=2)
        return await execute(session, statement, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "execute", execute_with_writer)
    with pytest.raises(AIError) as raised:
        await metrics.query(MetricQuery(_NAMES[2], window, aggregation=MetricAggregation.MIN))
    assert inserted
    assert raised.value.code is ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize("aggregation", (MetricAggregation.COUNT, MetricAggregation.PERCENTILE))
@pytest.mark.parametrize("count", (256, 257, 300))
async def test_group_limit_sentinel_is_not_storage_corruption(
    sql_metrics: _SqlMetrics, aggregation: MetricAggregation, count: int,
) -> None:
    metrics, _, window = sql_metrics
    await _seed(metrics, count, groups=count)
    query = MetricQuery(
        _NAMES[2], window, aggregation=aggregation,
        percentile=0.95 if aggregation is MetricAggregation.PERCENTILE else None,
        group_by=("group",),
    )
    if count > 256:
        with pytest.raises(AIError) as raised:
            await metrics.query(query)
        assert raised.value.code is ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED
    else:
        result = await metrics.query(query)
        assert len(result.points) == count
        assert sum(point.sample_count for point in result.points) == count


@pytest.mark.asyncio
@pytest.mark.parametrize("groups", (2, 3))
async def test_result_grid_budget_includes_empty_buckets(
    sql_metrics: _SqlMetrics, monkeypatch: pytest.MonkeyPatch, groups: int,
) -> None:
    metrics, _, window = sql_metrics
    await _seed(metrics, groups, groups=groups)
    monkeypatch.setattr(_query, "_MAX_RESULT_POINTS", 4)
    query = MetricQuery(_NAMES[2], window, group_by=("group",), bucket=timedelta(seconds=1))
    if groups > 2:
        with pytest.raises(AIError) as raised:
            await metrics.query(query)
        assert raised.value.code is ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED
    else:
        assert len((await metrics.query(query)).points) == 4


@pytest.mark.asyncio
async def test_oversized_window_does_not_expand_payload_json(
    sql_metrics: _SqlMetrics, monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics, engine, window = sql_metrics
    await _seed(metrics, 4)
    async with engine.begin() as connection:
        await connection.execute(text("UPDATE ai_metric_observations SET payload_json = 'invalid-json'"))
    monkeypatch.setattr(_query, "_MAX_SCANNED_OBSERVATIONS", 3)
    with pytest.raises(AIError) as raised:
        await metrics.query(MetricQuery(
            _NAMES[2], window, aggregation=MetricAggregation.PERCENTILE, percentile=0.95,
        ))
    assert raised.value.code is ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize("aggregation", (MetricAggregation.MIN, MetricAggregation.MAX))
async def test_sqlite_extrema_expand_once_without_order_sort(
    sql_metrics: _SqlMetrics, aggregation: MetricAggregation,
) -> None:
    metrics, engine, window = sql_metrics
    await _seed(metrics, 1500)
    statements: list[tuple[str, object]] = []

    def record_statement(
        connection: object, cursor: object, statement: str, parameters: object,
        context: object, executemany: bool,
    ) -> None:
        if statement.lstrip().lower().startswith("with ") and "ai_metric_observations" in statement:
            statements.append((statement, parameters))

    event.listen(engine.sync_engine, "before_cursor_execute", record_statement)
    try:
        result = await metrics.query(MetricQuery(_NAMES[2], window, aggregation=aggregation))
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", record_statement)
    assert result.points[0].value == (0 if aggregation is MetricAggregation.MIN else 1499)
    assert len(statements) == 1
    statement, parameters = statements[0]
    async with engine.connect() as connection:
        plan = await connection.exec_driver_sql("EXPLAIN QUERY PLAN " + statement, parameters)
    details = [str(row[3]) for row in plan]
    assert sum("SEARCH ai_metric_observations" in detail for detail in details) == 1
    assert sum("SCAN m VIRTUAL TABLE" in detail for detail in details) == 1
    assert not any("TEMP B-TREE FOR ORDER BY" in detail for detail in details)
    assert "ROW_NUMBER" not in statement


def _plan() -> _MetricQueryPushdownPlan:
    return _MetricQueryPushdownPlan(
        observation_kind="business.review", source_kind=MetricSourceKind.MEASUREMENT,
        metric_type=MetricType.DISTRIBUTION, measurement_name="value", measurement_revision=1,
        indicator_field=None, indicator_values=(), aggregation=MetricAggregation.PERCENTILE,
        percentile=0.95, start=_START, end=_START + timedelta(seconds=2),
        filters=(), correlation_filters=(), group_by=(), bucket_microseconds=None, bucket_count=1,
        max_scanned_observations=100000, max_extracted_samples=100000,
        max_groups=256, max_result_points=16384,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("dialect", "version", "supported"), (
    ("mysql", (8, 0, 16), False), ("mysql", (8, 0, 17), True),
    ("postgresql", (9, 3), False), ("postgresql", (9, 4), True),
    ("sqlite", (3, 24, 0), False), ("sqlite", (3, 25, 0), True),
))
async def test_query_features_follow_required_server_capabilities(
    dialect: str, version: tuple[int, ...], supported: bool,
) -> None:
    session = AsyncMock(spec=AsyncSession)
    session.connection.return_value = SimpleNamespace(dialect=SimpleNamespace(server_version_info=version))
    assert await _sql_features_available(session, dialect, _plan()) is supported
    assert session.execute.await_count == (1 if dialect == "sqlite" and supported else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(("message", "missing"), (
    ("no such function: json_extract", True),
    ("no such table: json_each", True),
    ("database is locked", False),
    ("near malformed: syntax error", False),
))
async def test_only_missing_sqlite_probe_features_select_scan(
    message: str, missing: bool,
) -> None:
    session = AsyncMock(spec=AsyncSession)
    session.connection.return_value = SimpleNamespace(dialect=SimpleNamespace(server_version_info=(3, 46, 1)))
    error = OperationalError("probe", {}, sqlite3.OperationalError(message))
    session.execute.side_effect = error
    if missing:
        assert await _sql_features_available(session, "sqlite", _plan()) is False
    else:
        with pytest.raises(OperationalError) as raised:
            await _sql_features_available(session, "sqlite", _plan())
        assert raised.value is error


@pytest.mark.asyncio
async def test_aggregate_failure_does_not_start_scan_fallback(
    sql_metrics: _SqlMetrics, monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics, _, window = sql_metrics
    await _seed(metrics, 1)
    execute = AsyncSession.execute
    error = OperationalError("aggregate", {}, sqlite3.OperationalError("database is locked"))

    async def fail_aggregate(session: AsyncSession, statement: object, *args: object, **kwargs: object) -> object:
        if str(statement).lstrip().lower().startswith("with "):
            raise error
        return await execute(session, statement, *args, **kwargs)

    scan = AsyncMock(side_effect=AssertionError("unexpected scan fallback"))
    monkeypatch.setattr(AsyncSession, "execute", fail_aggregate)
    monkeypatch.setattr(SqlMetricStore, "scan_observations", scan)
    with pytest.raises(OperationalError) as raised:
        await metrics.query(MetricQuery(_NAMES[2], window))
    assert raised.value is error
    scan.assert_not_awaited()
