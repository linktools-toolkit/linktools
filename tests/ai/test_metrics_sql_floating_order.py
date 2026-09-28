#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL floating reduction preserves chronological Python arithmetic."""

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
from linktools.ai.observe._sql_query import _coerce_aggregate_number, _coerce_sample_number
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine


_START = datetime(2026, 9, 7, tzinfo=timezone.utc)
_VALUES = (
    (1e16, 1.0, -1e16),
    (2**63 - 1, 1, 0.0, -float(2**63)),
    (-2**63, -2**63, 1.0),
    (2**53 + 1, 2**53 + 3, 0.25, -float(2**54)),
    (0.1, 0.2, 0.3),
    (5e-324, 5e-324, -5e-324),
    (1e308, 1e308, -1e308),
    (),
)


def _observation(index: int, value: int | float) -> Observation:
    return Observation(
        version=1,
        observation_id=f"value-{index}",
        kind="business.ordered.sample",
        occurred_at=_START + timedelta(microseconds=index),
        source_namespace=None,
        tenant_id=None,
        status=None,
        error_code=None,
        correlation={},
        dimensions={},
        measurements=(MetricMeasurement("value", 1, value),),
    )


def _definition(metric_type: MetricType) -> MetricDefinition:
    return MetricDefinition(
        name="business.ordered.value",
        revision=1,
        observation_kind="business.ordered.sample",
        source=MetricSource.measurement("value"),
        metric_type=metric_type,
        unit="1",
        default_aggregation=(
            MetricAggregation.SUM if metric_type is MetricType.COUNTER
            else MetricAggregation.MEAN
        ),
        query_fields=("group",),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("path_backed", (False, True))
@pytest.mark.parametrize("values", _VALUES)
async def test_floating_mean_matches_ordered_scan(
    tmp_path: Path, path_backed: bool, values: tuple[int | float, ...],
) -> None:
    path = tmp_path / "floating.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    namespace = "ordered-float"
    metrics = Metrics.sqlite(path, namespace=namespace) if path_backed else Metrics.sql(
        engine, namespace=namespace
    )
    memory = Metrics.in_memory(namespace=namespace)
    definition = _definition(MetricType.GAUGE)
    observations = tuple(_observation(i, value) for i, value in enumerate(values))
    try:
        await provision_metrics_database(engine)
        for target in (metrics, memory):
            await target.define(definition)
            await target.record_observations(tuple(reversed(observations)))
        query = MetricQuery(
            definition.name,
            MetricWindow.between(_START, _START + timedelta(seconds=1)),
        )
        assert await metrics.query(query) == await memory.query(query)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_floating_partitions_and_ties_do_not_share_accumulator(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'partitions.db'}")
    sql = Metrics.sql(engine, namespace="floating-partitions")
    memory = Metrics.in_memory(namespace="floating-partitions")
    definition = _definition(MetricType.GAUGE)
    observations = tuple(
        replace(
            _observation(index, value),
            observation_id=f"{group}-{bucket}-{index}",
            occurred_at=_START + timedelta(seconds=bucket),
            dimensions={} if group is None else {"group": group},
        )
        for group, values in ((None, (1e16, 1.0, -1e16)), ("b", (2**63 - 1, 1)))
        for bucket in (0, 2)
        for index, value in enumerate(values)
    )
    try:
        await provision_metrics_database(engine)
        for target in (sql, memory):
            await target.define(definition)
            await target.record_observations(tuple(reversed(observations)))
        query = MetricQuery(
            definition.name,
            MetricWindow.between(_START, _START + timedelta(seconds=3)),
            group_by=("group",),
            bucket=timedelta(seconds=1),
        )
        assert await sql.query(query) == await memory.query(query)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("aggregation", (MetricAggregation.SUM, MetricAggregation.RATE))
async def test_floating_counter_preserves_integer_prefix(
    tmp_path: Path, aggregation: MetricAggregation,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'counter.db'}")
    sql = Metrics.sql(engine, namespace="floating-counter")
    memory = Metrics.in_memory(namespace="floating-counter")
    definition = _definition(MetricType.COUNTER)
    values = (2**63 - 1, 2**63 - 1, 0.5, 2**53 + 1, 0.1)
    try:
        await provision_metrics_database(engine)
        for target in (sql, memory):
            await target.define(definition)
            await target.record_observations(tuple(_observation(i, v) for i, v in enumerate(values)))
        query = MetricQuery(
            definition.name,
            MetricWindow.between(_START, _START + timedelta(microseconds=7)),
            aggregation=aggregation,
            bucket=timedelta(microseconds=4),
        )
        assert await sql.query(query) == await memory.query(query)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_floating_reduction_uses_one_observation_statement(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'io.db'}")
    metrics = Metrics.sql(engine, namespace="floating-io")
    definition = _definition(MetricType.GAUGE)
    statements: list[str] = []

    def capture(
        _connection: object, _cursor: object, statement: str,
        _parameters: object, _context: object, _executemany: bool,
    ) -> None:
        if "ai_metric_observations" in statement:
            statements.append(statement)

    try:
        await provision_metrics_database(engine)
        await metrics.define(definition)
        for offset in range(0, 1500, 250):
            await metrics.record_observations(tuple(
                _observation(index, index % 17 / 10)
                for index in range(offset, offset + 250)
            ))
        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        result = await metrics.query(MetricQuery(
            definition.name, MetricWindow.between(_START, _START + timedelta(seconds=1)),
        ))
        assert result.points[0].sample_count == 1500
        assert len(statements) == 1
        assert "WITH RECURSIVE totals" in statements[0]
    finally:
        if event.contains(engine.sync_engine, "before_cursor_execute", capture):
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await engine.dispose()


@pytest.mark.parametrize("value", (float("inf"), float("-inf")))
def test_aggregate_overflow_is_not_an_invalid_input_sample(value: float) -> None:
    assert _coerce_aggregate_number(value) == value
    with pytest.raises(AIError) as raised:
        _coerce_sample_number(value)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
