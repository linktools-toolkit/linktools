#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metrics public contracts, query semantics, and persistence regressions."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import build_sql_schema_metadata, provision_metrics_database
from linktools.ai.observe import (
    MetricAggregation,
    MetricDefinition,
    MetricQuery,
    MetricSource,
    MetricType,
    MetricWindow,
    Metrics,
    Observation,
)
from linktools.ai.observe._codec import (
    decode_definition_envelope,
    decode_observation_envelope,
    definition_envelope,
    observation_envelope,
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
async def test_metrics_record_observations_accepts_sequences() -> None:
    metrics = Metrics.in_memory(namespace="metrics-sequence")
    observation = Observation(
        version=1,
        observation_id="sequence",
        kind="app.sequence",
        occurred_at=datetime(2026, 9, 5, tzinfo=timezone.utc),
        source_namespace="workspace",
        tenant_id="tenant",
        status="SUCCEEDED",
        error_code=None,
        correlation={},
        dimensions={},
        measurements=(),
    )
    await metrics.record_observations([observation])


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


def test_observation_rejects_boolean_version() -> None:
    with pytest.raises(AIError) as raised:
        Observation(
            version=True,  # type: ignore[arg-type]
            observation_id="observation",
            kind="app.request",
            occurred_at=datetime(2026, 9, 5, tzinfo=timezone.utc),
            source_namespace="workspace",
            tenant_id="tenant",
            status="SUCCEEDED",
            error_code=None,
            correlation={},
            dimensions={},
            measurements=(),
        )
    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID


def test_metric_codec_rejects_invalid_persisted_types() -> None:
    definition_payload = definition_envelope("metrics-codec", _latency_definition())
    definition_data = cast(
        "dict[str, object]", definition_payload["definition"]
    )
    definition_data["revision"] = "1"
    with pytest.raises(AIError) as definition_error:
        decode_definition_envelope(
            definition_payload,
            expected_namespace="metrics-codec",
        )
    assert definition_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

    version_payload = definition_envelope("metrics-codec", _latency_definition())
    version_payload["version"] = True
    with pytest.raises(AIError) as version_error:
        decode_definition_envelope(
            version_payload,
            expected_namespace="metrics-codec",
        )
    assert version_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

    observation = Observation(
        version=1,
        observation_id="observation",
        kind="app.request",
        occurred_at=datetime(2026, 9, 5, tzinfo=timezone.utc),
        source_namespace="workspace",
        tenant_id="tenant",
        status="SUCCEEDED",
        error_code=None,
        correlation={},
        dimensions={"route": "alpha"},
        measurements=(),
    )
    observation_payload_value = observation_envelope("metrics-codec", observation)
    observation_data = cast(
        "dict[str, object]", observation_payload_value["observation"]
    )
    observation_data["dimensions"] = {"route": 1}
    with pytest.raises(AIError) as observation_error:
        decode_observation_envelope(
            observation_payload_value,
            expected_namespace="metrics-codec",
        )
    assert observation_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_complete_sql_schema_includes_metrics_tables() -> None:
    metadata = build_sql_schema_metadata()
    assert len(metadata.tables) == 12
    assert "ai_metric_definitions" in metadata.tables
    assert "ai_metric_observations" in metadata.tables
