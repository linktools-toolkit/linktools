#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metrics public contracts, query semantics, and persistence regressions."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import build_sql_schema_metadata
from linktools.ai.observe import (
    MetricAggregation,
    MetricDefinition,
    MetricQuery,
    MetricSource,
    MetricType,
    MetricWindow,
    Metrics,
    provision_metrics_database,
)
from sqlalchemy.ext.asyncio import create_async_engine


def _latency_definition() -> MetricDefinition:
    return MetricDefinition(
        name="app.request.latency",
        revision=1,
        observation_kind="app.request",
        source=MetricSource.measurement("latency_ms"),
        metric_type=MetricType.DISTRIBUTION,
        unit="ms",
        default_aggregation=MetricAggregation.MEAN,
        query_fields=("route", "status"),
    )


@pytest.mark.asyncio
async def test_metrics_record_is_idempotent_and_query_is_definition_driven() -> None:
    metrics = Metrics.in_memory(namespace="metrics-contract")
    await metrics.define(_latency_definition())
    occurred_at = datetime(2026, 9, 5, 0, 0, tzinfo=timezone.utc)

    await metrics.record(
        "app.request.latency",
        10,
        observation_id="request-a",
        occurred_at=occurred_at,
        status="SUCCEEDED",
        dimensions={"route": "alpha"},
    )
    await metrics.record(
        "app.request.latency",
        30,
        observation_id="request-b",
        occurred_at=occurred_at + timedelta(seconds=1),
        status="SUCCEEDED",
        dimensions={"route": "beta"},
    )
    await metrics.record(
        "app.request.latency",
        10,
        observation_id="request-a",
        occurred_at=occurred_at,
        status="SUCCEEDED",
        dimensions={"route": "alpha"},
    )

    result = await metrics.query(
        MetricQuery(
            "app.request.latency",
            MetricWindow.between(
                occurred_at - timedelta(seconds=1),
                occurred_at + timedelta(seconds=2),
            ),
            filters={"status": "SUCCEEDED"},
            group_by=("route",),
        )
    )
    assert [
        (point.dimensions["route"], point.value, point.sample_count)
        for point in result.points
    ] == [
        ("alpha", 10.0, 1),
        ("beta", 30.0, 1),
    ]

    with pytest.raises(AIError) as raised:
        await metrics.record(
            "app.request.latency",
            11,
            observation_id="request-a",
            occurred_at=occurred_at,
            status="SUCCEEDED",
            dimensions={"route": "alpha"},
        )
    assert raised.value.code is ErrorCode.STORAGE_CONFLICT


@pytest.mark.asyncio
async def test_sqlite_metrics_require_explicit_provisioning(tmp_path: Path) -> None:
    path = tmp_path / "metrics.db"
    metrics = Metrics.sqlite(path, namespace="metrics-sqlite")

    with pytest.raises(AIError) as raised:
        await metrics.define(_latency_definition())
    assert raised.value.code is ErrorCode.STORAGE_CAPABILITY_MISSING

    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    try:
        await provision_metrics_database(engine)
        await metrics.define(_latency_definition())
        occurred_at = datetime(2026, 9, 5, 1, 0, tzinfo=timezone.utc)
        await metrics.record(
            "app.request.latency",
            25,
            observation_id="sqlite-request",
            occurred_at=occurred_at,
            status="SUCCEEDED",
            dimensions={"route": "alpha"},
        )
        result = await metrics.query(
            MetricQuery(
                "app.request.latency",
                MetricWindow.between(
                    occurred_at - timedelta(seconds=1),
                    occurred_at + timedelta(seconds=1),
                ),
            )
        )
        assert len(result.points) == 1
        assert result.points[0].value == 25.0
        assert result.points[0].sample_count == 1
    finally:
        await engine.dispose()


def test_complete_sql_schema_includes_metrics_tables() -> None:
    metadata = build_sql_schema_metadata()
    assert len(metadata.tables) == 12
    assert "ai_metric_definitions" in metadata.tables
    assert "ai_metric_observations" in metadata.tables
