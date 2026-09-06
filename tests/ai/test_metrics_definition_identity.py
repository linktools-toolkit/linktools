#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metric definition semantic identity across persistence."""

from pathlib import Path

import pytest
from linktools.ai.migrate import provision_metrics_database
from linktools.ai.observe import (
    MetricAggregation,
    MetricDefinition,
    MetricSource,
    MetricType,
    Metrics,
)
from linktools.ai.observe._codec import (
    decode_definition_envelope,
    definition_envelope,
    definition_semantic_digest,
)
from sqlalchemy.ext.asyncio import create_async_engine


@pytest.mark.asyncio
async def test_sql_definition_replay_ignores_query_field_order(tmp_path: Path) -> None:
    path = tmp_path / "metrics-definition-identity.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    metrics = Metrics.sql(engine, namespace="definition-identity")
    first = MetricDefinition(
        name="business.ordered",
        revision=1,
        observation_kind="business.ordered.sample",
        source=MetricSource.measurement("value"),
        metric_type=MetricType.GAUGE,
        unit="1",
        default_aggregation=MetricAggregation.MEAN,
        query_fields=("status", "route"),
    )
    reordered = MetricDefinition(
        name=first.name,
        revision=first.revision,
        observation_kind=first.observation_kind,
        source=first.source,
        metric_type=first.metric_type,
        unit=first.unit,
        default_aggregation=first.default_aggregation,
        query_fields=("route", "status"),
    )

    try:
        await provision_metrics_database(engine)
        stored = await metrics.define(first)
        replay = await metrics.define(reordered)

        assert replay == stored
        assert replay.query_fields == first.query_fields
    finally:
        await engine.dispose()


def test_definition_description_is_nonsemantic_and_v1_optional() -> None:
    first = MetricDefinition(
        name="business.description",
        revision=1,
        observation_kind="business.description.sample",
        source=MetricSource.measurement("value"),
        metric_type=MetricType.GAUGE,
        unit="1",
        default_aggregation=MetricAggregation.MEAN,
        description="first description",
    )
    changed = MetricDefinition(
        name=first.name,
        revision=first.revision,
        observation_kind=first.observation_kind,
        source=first.source,
        metric_type=first.metric_type,
        unit=first.unit,
        default_aggregation=first.default_aggregation,
        description="changed description",
    )

    assert definition_semantic_digest(first) == definition_semantic_digest(changed)

    legacy = definition_envelope(
        "definition-description",
        MetricDefinition(
            name=first.name,
            revision=first.revision,
            observation_kind=first.observation_kind,
            source=first.source,
            metric_type=first.metric_type,
            unit=first.unit,
            default_aggregation=first.default_aggregation,
        ),
    )
    definition = legacy["definition"]
    assert isinstance(definition, dict)
    assert "description" not in definition
    decoded = decode_definition_envelope(
        legacy,
        expected_namespace="definition-description",
    )
    assert decoded.description is None


@pytest.mark.asyncio
async def test_definition_description_replay_keeps_first_persisted_metadata() -> None:
    metrics = Metrics.in_memory(namespace="definition-description-replay")
    first = MetricDefinition(
        name="business.description.replay",
        revision=1,
        observation_kind="business.description.replay.sample",
        source=MetricSource.measurement("value"),
        metric_type=MetricType.GAUGE,
        unit="1",
        default_aggregation=MetricAggregation.MEAN,
        description="first description",
    )
    replayed = MetricDefinition(
        name=first.name,
        revision=first.revision,
        observation_kind=first.observation_kind,
        source=first.source,
        metric_type=first.metric_type,
        unit=first.unit,
        default_aggregation=first.default_aggregation,
        description="changed description",
    )

    stored = await metrics.define(first)
    replay = await metrics.define(replayed)

    assert stored == first
    assert replay == first
    assert replay.description == "first description"
