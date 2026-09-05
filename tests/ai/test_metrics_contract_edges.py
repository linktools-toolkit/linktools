#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metric public contract shape and character-limit regressions."""

from datetime import datetime, timedelta, timezone

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.observe import (
    MetricAggregation,
    MetricDefinition,
    MetricQuery,
    MetricSource,
    MetricSourceKind,
    MetricType,
    MetricWindow,
    Observation,
)


def _assert_invalid(factory: object) -> None:
    with pytest.raises(AIError) as raised:
        factory()  # type: ignore[operator]
    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID


def test_metric_text_limits_count_characters_not_utf8_bytes() -> None:
    value = "测" * 100
    observation = Observation(
        version=1,
        observation_id="unicode-shape",
        kind="business.unicode",
        occurred_at=datetime(2026, 9, 5, tzinfo=timezone.utc),
        source_namespace="workspace",
        tenant_id="default",
        status=value,
        error_code=None,
        correlation={"label": value},
        dimensions={"label": value},
        measurements=(),
    )

    assert observation.status == value
    assert observation.correlation["label"] == value
    assert observation.dimensions["label"] == value


def test_metric_tuple_contracts_reject_bare_strings() -> None:
    _assert_invalid(
        lambda: MetricSource(
            MetricSourceKind.INDICATOR,
            indicator_field="status",
            indicator_values="FAILED",  # type: ignore[arg-type]
        )
    )
    _assert_invalid(
        lambda: MetricDefinition(
            name="business.invalid.fields",
            revision=1,
            observation_kind="business.invalid.fields.sample",
            source=MetricSource.measurement("value"),
            metric_type=MetricType.GAUGE,
            unit="1",
            default_aggregation=MetricAggregation.MEAN,
            query_fields="status",  # type: ignore[arg-type]
        )
    )
    window = MetricWindow.between(
        datetime(2026, 9, 5, tzinfo=timezone.utc),
        datetime(2026, 9, 5, 0, 1, tzinfo=timezone.utc),
    )
    _assert_invalid(
        lambda: MetricQuery(
            "business.invalid.fields",
            window,
            group_by="status",  # type: ignore[arg-type]
        )
    )


def test_metric_window_and_query_reject_wrong_runtime_types() -> None:
    _assert_invalid(lambda: MetricWindow(recent_delta=1))  # type: ignore[arg-type]
    window = MetricWindow.recent(minutes=1)
    _assert_invalid(
        lambda: MetricQuery(
            "business.invalid.bucket",
            window,
            bucket=1,  # type: ignore[arg-type]
        )
    )
    _assert_invalid(
        lambda: MetricQuery(
            "business.invalid.filters",
            window,
            filters=[("status", "FAILED")],  # type: ignore[arg-type]
        )
    )


def test_observation_rejects_non_contract_collection_shapes() -> None:
    now = datetime(2026, 9, 5, tzinfo=timezone.utc)
    _assert_invalid(
        lambda: Observation(
            version=1,
            observation_id="invalid-measurements",
            kind="business.invalid",
            occurred_at=now,
            source_namespace=None,
            tenant_id=None,
            status=None,
            error_code=None,
            correlation={},
            dimensions={},
            measurements=[],  # type: ignore[arg-type]
        )
    )
    _assert_invalid(
        lambda: Observation(
            version=1,
            observation_id="invalid-correlation",
            kind="business.invalid",
            occurred_at=now,
            source_namespace=None,
            tenant_id=None,
            status=None,
            error_code=None,
            correlation=[],  # type: ignore[arg-type]
            dimensions={},
            measurements=(),
        )
    )
