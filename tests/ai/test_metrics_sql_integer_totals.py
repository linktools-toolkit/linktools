#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integer metric totals must not inherit database int64 overflow or float coercion."""

from dataclasses import replace
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
from linktools.ai.observe._sql_query import _coerce_aggregate_number, _measurement_sum_sql
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine


_MAX = 2**63 - 1
_MIN = -(2**63)
_START = datetime(2026, 9, 7, tzinfo=timezone.utc)


def _definition(metric_type: MetricType) -> MetricDefinition:
    return MetricDefinition(
        name="business.exact.total",
        revision=1,
        observation_kind="business.exact.sample",
        source=MetricSource.measurement("value"),
        metric_type=metric_type,
        unit="1",
        default_aggregation=(
            MetricAggregation.SUM
            if metric_type is MetricType.COUNTER
            else MetricAggregation.MEAN
        ),
        query_fields=("group",),
    )


def _observation(index: int, value: int | float, *, group: str = "integers") -> Observation:
    return Observation(
        version=1,
        observation_id=f"{group}-{index}",
        kind="business.exact.sample",
        occurred_at=_START + timedelta(microseconds=index),
        source_namespace=None,
        tenant_id=None,
        status=None,
        error_code=None,
        correlation={},
        dimensions={"group": group},
        measurements=(MetricMeasurement("value", 1, value),),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("path_backed", (False, True))
@pytest.mark.parametrize(
    ("metric_type", "values", "aggregation"),
    (
        (MetricType.COUNTER, (_MAX, 1), MetricAggregation.SUM),
        (MetricType.COUNTER, (_MAX, _MAX), MetricAggregation.RATE),
        (MetricType.COUNTER, (2**53 + 1, 2**53 + 3), MetricAggregation.SUM),
        (MetricType.GAUGE, (_MAX, 1), MetricAggregation.MEAN),
        (MetricType.GAUGE, (_MIN, -1), MetricAggregation.MEAN),
        (MetricType.GAUGE, (_MIN, _MAX, 17), MetricAggregation.MEAN),
        (MetricType.DISTRIBUTION, (_MAX, _MAX, _MIN), MetricAggregation.MEAN),
        (MetricType.COUNTER, (), MetricAggregation.SUM),
        (MetricType.GAUGE, (), MetricAggregation.MEAN),
    ),
)
async def test_integer_totals_match_python_beyond_int64(
    tmp_path: Path,
    path_backed: bool,
    metric_type: MetricType,
    values: tuple[int, ...],
    aggregation: MetricAggregation,
) -> None:
    path = tmp_path / "integer-totals.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    namespace = "integer-totals"
    sql = Metrics.sqlite(path, namespace=namespace) if path_backed else Metrics.sql(
        engine, namespace=namespace
    )
    memory = Metrics.in_memory(namespace=namespace)
    definition = _definition(metric_type)
    observations = tuple(_observation(index, value) for index, value in enumerate(values))
    query = MetricQuery(
        definition.name,
        MetricWindow.between(_START, _START + timedelta(seconds=2)),
        aggregation=aggregation,
    )
    try:
        await provision_metrics_database(engine)
        for metrics in (sql, memory):
            await metrics.define(definition)
            await metrics.record_observations(observations)
        result = await sql.query(query)
        assert result == await memory.query(query)
        if aggregation is MetricAggregation.SUM:
            assert type(result.points[0].value) is int
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_integer_partitions_remain_exact_next_to_float_partitions(tmp_path: Path) -> None:
    path = tmp_path / "partition-totals.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    sql = Metrics.sql(engine, namespace="partition-totals")
    memory = Metrics.in_memory(namespace="partition-totals")
    definition = _definition(MetricType.COUNTER)
    observations = (
        _observation(0, _MAX),
        _observation(1, _MAX),
        _observation(2, 1.5, group="floats"),
        _observation(3, 2, group="floats"),
        replace(_observation(4, _MAX), occurred_at=_START + timedelta(seconds=1)),
        replace(_observation(5, 1), occurred_at=_START + timedelta(seconds=1)),
    )
    try:
        await provision_metrics_database(engine)
        for metrics in (sql, memory):
            await metrics.define(definition)
            await metrics.record_observations(observations)
        for aggregation in (MetricAggregation.SUM, MetricAggregation.RATE):
            query = MetricQuery(
                definition.name,
                MetricWindow.between(_START, _START + timedelta(seconds=2)),
                aggregation=aggregation,
                group_by=("group",),
                bucket=timedelta(seconds=1),
            )
            assert await sql.query(query) == await memory.query(query)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_large_integer_sum_transfers_only_one_aggregate_statement(tmp_path: Path) -> None:
    path = tmp_path / "large-integer-totals.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    metrics = Metrics.sql(engine, namespace="large-integer-totals")
    definition = _definition(MetricType.COUNTER)
    statements: list[str] = []

    def capture(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        if "ai_metric_observations" in statement.lower():
            statements.append(statement)

    try:
        await provision_metrics_database(engine)
        await metrics.define(definition)
        for offset in range(0, 1500, 250):
            await metrics.record_observations(
                tuple(_observation(index, _MAX) for index in range(offset, offset + 250))
            )
        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        result = await metrics.query(
            MetricQuery(
                definition.name,
                MetricWindow.between(_START, _START + timedelta(seconds=1)),
            )
        )
        assert result.points[0].sample_count == 1500
        assert result.points[0].value == _MAX * 1500
        assert type(result.points[0].value) is int
        assert len(statements) == 1
        assert "integer_sum_high" in statements[0]
    finally:
        if event.contains(engine.sync_engine, "before_cursor_execute", capture):
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await engine.dispose()


def test_mysql_integer_total_uses_raw_decimal_not_promoted_numeric_case() -> None:
    statement = _measurement_sum_sql("mysql", MetricAggregation.SUM, [])
    assert "THEN CAST(JSON_UNQUOTE(raw_value) AS DECIMAL(65, 0))" in statement
    assert "ELSE NULL END) AS integer_sum_low" in statement


def test_postgresql_integer_total_keeps_exact_numeric_accumulation() -> None:
    statement = _measurement_sum_sql("postgresql", MetricAggregation.MEAN, [])
    assert "THEN numeric_value ELSE NULL END) AS integer_sum_low" in statement


@pytest.mark.parametrize("value", ("1", "1e3", "NaN", "null", "[1, 2]"))
def test_aggregate_decoder_rejects_unexpected_string_results(value: str) -> None:
    with pytest.raises(AIError) as raised:
        _coerce_aggregate_number(value)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
