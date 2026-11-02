#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exercise PostgreSQL native order statistics when local server tools exist."""

import json
import subprocess
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import pytest
from linktools.ai.observe import MetricAggregation, MetricSourceKind, MetricType
from linktools.ai.observe._sql_query import (
    _base_params, _decode_measurement_rows, _measurement_sql,
    _percentile_pick_sql, _postgresql_numeric_order,
)
from linktools.ai.observe._store import _MetricQueryPushdownPlan
from sqlalchemy import text
from sqlalchemy.dialects import postgresql

from ._metrics_server_test_helpers import _server_database


@pytest.fixture(scope="module")
def _postgresql(_postgresql_server: tuple[str, list[str]]) -> Iterator[tuple[str, list[str]]]:
    with _server_database(_postgresql_server) as server:
        yield server


def _plan(aggregation: MetricAggregation, percentile: float | None) -> _MetricQueryPushdownPlan:
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)
    return _MetricQueryPushdownPlan(
        observation_kind="business.numeric", source_kind=MetricSourceKind.MEASUREMENT,
        metric_type=MetricType.DISTRIBUTION, measurement_name="value", measurement_revision=1,
        indicator_field=None, indicator_values=(), aggregation=aggregation, percentile=percentile,
        start=start, end=start + timedelta(seconds=1), filters=(), correlation_filters=(),
        group_by=(), bucket_microseconds=None, bucket_count=1,
        max_scanned_observations=100000, max_extracted_samples=100000,
        max_groups=256, max_result_points=16384,
    )


def _query(
    connection: tuple[str, list[str]], values: tuple[int | float, ...],
    aggregation: MetricAggregation, percentile: float | None = None,
) -> int | float | None:
    _name, command = connection
    plan = _plan(aggregation, percentile)
    statement, params = _measurement_sql("postgresql", plan, _base_params("ns", "postgresql", plan))
    dialect = postgresql.dialect()
    sql = str(text(statement).bindparams(**params).compile(dialect=dialect, compile_kwargs={"literal_binds": True}))
    inserts = []
    for index, value in enumerate(values):
        payload = {"observation": {"measurements": [{"name": "value", "revision": 1, "value": value}]}}
        row = text(
            "INSERT INTO ai_metric_observations VALUES ('ns', 'business.numeric', :occurred, :identity, CAST(:payload AS JSON), NULL, :occurred);"
        ).bindparams(
            occurred=plan.start + timedelta(microseconds=index), identity=f"{index:064x}", payload=json.dumps(payload),
        )
        inserts.append(str(row.compile(dialect=dialect, compile_kwargs={"literal_binds": True})))
    script = """BEGIN;
CREATE TEMP TABLE ai_metric_observations (
    namespace_digest TEXT, kind TEXT, occurred_at TIMESTAMPTZ,
    observation_digest TEXT, payload_json JSON, pruned_at TIMESTAMPTZ, active_occurred_at TIMESTAMPTZ
);
""" + "\n".join(inserts) + "\nSELECT COALESCE(json_agg(q), '[]') FROM (" + sql + ") AS q;\nROLLBACK;"
    result = subprocess.run(
        command,
        input=script, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    rows = json.loads(result.stdout)
    decoded = _decode_measurement_rows(rows, plan)
    return None if not decoded.rows else decoded.rows[0].selected_value


@pytest.mark.parametrize("aggregation", (MetricAggregation.MIN, MetricAggregation.MAX, MetricAggregation.PERCENTILE))
@pytest.mark.parametrize("values", (
    (1.0000000000000002e17, 100000000000000018),
    (-1.0000000000000002e17, -100000000000000018),
    (2**63 - 1, float(2**63)),
    (-2**63, float(-2**63)),
    (2**53 + 1, float(2**53), float(2**53 + 2)),
    (5e-324, -5e-324, 0, 1e308, -1e308),
    (1.0, 1, 1.0, 2),
    (),
))
def test_postgresql_selected_values_are_original_samples(
    _postgresql: tuple[str, list[str]], aggregation: MetricAggregation, values: tuple[int | float, ...],
) -> None:
    p = 0.5 if aggregation is MetricAggregation.PERCENTILE else None
    actual = _query(_postgresql, values, aggregation, p)
    if not values:
        assert actual is None
        return
    if aggregation is MetricAggregation.MIN:
        expected = min(values)
    elif aggregation is MetricAggregation.MAX:
        expected = max(values)
    else:
        import math
        expected = sorted(values)[math.ceil(0.5 * len(values)) - 1]
    assert actual == expected
    assert type(actual) is type(expected)


def test_postgresql_native_percentile_returns_raw_sample_text() -> None:
    statement = _percentile_pick_sql("postgresql", [])
    assert "percentile_disc" in statement
    assert "row_to_json" in statement
    assert "->> 'f5'" in statement
    coarse, residual = _postgresql_numeric_order()
    assert "DOUBLE PRECISION" in coarse
    assert "9223372036854775807 - 1" in residual
