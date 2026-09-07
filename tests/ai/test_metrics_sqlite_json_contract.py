#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ignored envelope fields cannot bypass SQLite's strict JSON read boundary."""

import json
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
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import create_async_engine


@pytest.mark.asyncio
@pytest.mark.parametrize("path_backed", (False, True))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf"), 1.5))
async def test_ignored_fields_obey_json_syntax_before_aggregation(
    tmp_path: Path, path_backed: bool, value: float,
) -> None:
    path = tmp_path / "json-contract.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    metrics = (Metrics.sqlite(path, namespace="json-contract") if path_backed
               else Metrics.sql(engine, namespace="json-contract"))
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)
    window = MetricWindow.between(start, start + timedelta(seconds=1))
    queries = []
    statements: list[str] = []

    def record_statement(
        connection: object, cursor: object, statement: str, parameters: object,
        context: object, executemany: bool,
    ) -> None:
        if "ai_metric_observations" in statement:
            statements.append(statement)

    try:
        await provision_metrics_database(engine)
        for name, source, metric_type, aggregation in (
            ("business.json.count", MetricSource.observation_count(),
             MetricType.COUNTER, MetricAggregation.COUNT),
            ("business.json.p95", MetricSource.measurement("value"),
             MetricType.DISTRIBUTION, MetricAggregation.PERCENTILE),
        ):
            await metrics.define(MetricDefinition(
                name=name, revision=1, observation_kind="business.json",
                source=source, metric_type=metric_type, unit="1",
                default_aggregation=aggregation, query_fields=("status",),
            ))
            for filters in ({}, {"status": "FAILED"}):
                queries.append(MetricQuery(
                    name, window, aggregation=aggregation,
                    percentile=.95 if aggregation is MetricAggregation.PERCENTILE else None,
                    filters=filters,
                ))
        await metrics.record_observations((Observation(
            version=1, observation_id="one", kind="business.json", occurred_at=start,
            source_namespace=None, tenant_id=None, status="SUCCEEDED", error_code=None,
            correlation={}, dimensions={}, measurements=(MetricMeasurement("value", 1, 12.5),),
        ),))
        async with engine.begin() as connection:
            raw = await connection.scalar(text("SELECT payload_json FROM ai_metric_observations"))
            payload = json.loads(raw)
            payload["future_optional_field"] = {"nested": [value]}
            await connection.execute(
                text("UPDATE ai_metric_observations SET payload_json = :payload"),
                {"payload": json.dumps(payload)},
            )
        event.listen(engine.sync_engine, "before_cursor_execute", record_statement)
        for query in queries:
            statements.clear()
            if value == 1.5:
                result = await metrics.query(query)
                expected = 0 if query.filters else 1
                assert result.points[0].sample_count == expected
            else:
                with pytest.raises(AIError) as caught:
                    await metrics.query(query)
                assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            if not path_backed:
                assert len(statements) == 1
    finally:
        if event.contains(engine.sync_engine, "before_cursor_execute", record_statement):
            event.remove(engine.sync_engine, "before_cursor_execute", record_statement)
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("path_backed", (False, True))
@pytest.mark.parametrize("field", ("observation_id", "status"))
async def test_unencodable_record_text_returns_typed_integrity_error(
    tmp_path: Path, path_backed: bool, field: str,
) -> None:
    path = tmp_path / "invalid-unicode.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    metrics = (Metrics.sqlite(path, namespace="invalid-unicode") if path_backed
               else Metrics.sql(engine, namespace="invalid-unicode"))
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)
    try:
        await provision_metrics_database(engine)
        await metrics.define(MetricDefinition(
            name="business.unicode.count", revision=1,
            observation_kind="business.unicode", source=MetricSource.observation_count(),
            metric_type=MetricType.COUNTER, unit="1",
            default_aggregation=MetricAggregation.COUNT,
        ))
        await metrics.record_observations((Observation(
            version=1, observation_id="one", kind="business.unicode", occurred_at=start,
            source_namespace=None, tenant_id=None, status="SUCCEEDED", error_code=None,
            correlation={}, dimensions={}, measurements=(),
        ),))
        async with engine.begin() as connection:
            raw = await connection.scalar(text("SELECT payload_json FROM ai_metric_observations"))
            payload = json.loads(raw)
            payload["observation"][field] = "bad-\ud800"
            await connection.execute(
                text("UPDATE ai_metric_observations SET payload_json = :payload"),
                {"payload": json.dumps(payload)},
            )
        with pytest.raises(AIError) as caught:
            await metrics.query(MetricQuery(
                "business.unicode.count",
                MetricWindow.between(start, start + timedelta(seconds=1)),
            ))
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await engine.dispose()
