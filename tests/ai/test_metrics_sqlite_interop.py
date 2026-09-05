#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite convenience and SQLAlchemy SQLite interoperability."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from linktools.ai.migrate import provision_metrics_database
from linktools.ai.observe import (
    MetricAggregation,
    MetricDefinition,
    MetricQuery,
    MetricSource,
    MetricType,
    MetricWindow,
    Metrics,
)
from sqlalchemy.ext.asyncio import create_async_engine


@pytest.mark.asyncio
async def test_sqlite_and_sql_backends_share_one_metric_dataset(tmp_path: Path) -> None:
    path = tmp_path / "metrics.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    raw = Metrics.sqlite(path, namespace="interop")
    sql = Metrics.sql(engine, namespace="interop")
    definition = MetricDefinition(
        name="business.interop",
        revision=1,
        observation_kind="business.interop.sample",
        source=MetricSource.measurement("value"),
        metric_type=MetricType.COUNTER,
        unit="1",
        default_aggregation=MetricAggregation.SUM,
    )
    occurred_at = datetime(2026, 9, 5, 4, 0, tzinfo=timezone.utc)
    window = MetricWindow.between(
        occurred_at - timedelta(seconds=1),
        occurred_at + timedelta(seconds=3),
    )

    try:
        await provision_metrics_database(engine)
        await raw.define(definition)
        await raw.record(
            definition.name,
            1,
            observation_id="raw",
            occurred_at=occurred_at,
        )

        await sql.define(definition)
        from_raw = await sql.query(MetricQuery(definition.name, window))
        assert from_raw.points[0].value == 1
        assert from_raw.points[0].sample_count == 1

        await sql.record(
            definition.name,
            2,
            observation_id="sql",
            occurred_at=occurred_at + timedelta(seconds=1),
        )

        raw_result = await raw.query(MetricQuery(definition.name, window))
        sql_result = await sql.query(MetricQuery(definition.name, window))
        for result in (raw_result, sql_result):
            assert result.points[0].value == 3
            assert result.points[0].sample_count == 2
    finally:
        await engine.dispose()
