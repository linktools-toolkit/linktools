#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite Metrics leaves schema diagnostics to explicit deployment calls."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from linktools.ai.migrate import provision_metrics_sqlite, validate_metrics_sqlite
from linktools.ai.observe import MetricAggregation, MetricDefinition, MetricQuery, MetricSource, MetricType, MetricWindow, Metrics
import linktools.ai.storage._database as database_module


@pytest.mark.asyncio
async def test_sqlite_schema_validation_is_explicit_across_operations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "metrics.db"
    await provision_metrics_sqlite(path)
    real_validate = database_module._validate_sql_schema
    calls = 0
    async def counting_validate(engine: object, metadata: object) -> None:
        nonlocal calls
        calls += 1
        await real_validate(engine, metadata)  # type: ignore[arg-type]
    monkeypatch.setattr(database_module, "_validate_sql_schema", counting_validate)
    metrics = Metrics.sqlite(path, namespace="validation-cache")
    definition = MetricDefinition(name="business.validation_cache", revision=1, observation_kind="business.validation_cache.sample", source=MetricSource.measurement("value"), metric_type=MetricType.COUNTER, unit="1", default_aggregation=MetricAggregation.SUM)
    occurred_at = datetime(2026, 9, 6, tzinfo=timezone.utc)
    await metrics.define(definition)
    await metrics.record(definition.name, 1, observation_id="sample", occurred_at=occurred_at)
    result = await metrics.query(MetricQuery(definition.name, MetricWindow.between(occurred_at - timedelta(seconds=1), occurred_at + timedelta(seconds=1))))
    assert result.points[0].value == 1
    assert calls == 0
    await validate_metrics_sqlite(path)
    assert calls == 1
