#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MySQL order statistics retain the exact mixed-numeric sample ordering."""

import json
import math
import subprocess
from dataclasses import replace
from datetime import timedelta

import pytest
from linktools.ai.observe import MetricAggregation, MetricType
from linktools.ai.observe._sql_query import _base_params, _decode_measurement_rows, _measurement_sql
from sqlalchemy import text
from sqlalchemy.dialects import mysql

from .test_metrics_sql_server_sums import _plan, _server


@pytest.mark.parametrize("_server", ("mysql",), indirect=True)
@pytest.mark.parametrize("values", (
    (1.0000000000000002e17, 100000000000000018),
    (-1.0000000000000002e17, -100000000000000018),
    (2**63 - 3, 2**63 - 2, 2**63 - 1, float(2**63)),
))
@pytest.mark.parametrize("aggregation", (
    MetricAggregation.MIN, MetricAggregation.MAX, MetricAggregation.PERCENTILE,
))
def test_mysql_selected_value_matches_python(
    _server: tuple[str, list[str]], values: tuple[int | float, ...],
    aggregation: MetricAggregation,
) -> None:
    name, command = _server
    plan = replace(
        _plan(), metric_type=MetricType.DISTRIBUTION, aggregation=aggregation,
        percentile=.5 if aggregation is MetricAggregation.PERCENTILE else None,
    )
    dialect = mysql.dialect(paramstyle="named")
    statement, params = _measurement_sql(name, plan, _base_params("ns", name, plan))
    sql = str(text(statement).bindparams(**params).compile(
        dialect=dialect, compile_kwargs={"literal_binds": True},
    ))
    script = """DROP TABLE IF EXISTS ai_metric_observations;
CREATE TABLE ai_metric_observations (
    namespace_digest VARCHAR(64), kind VARCHAR(128), occurred_at DATETIME(6),
    observation_digest VARCHAR(64), payload_json JSON
);
"""
    for index, value in enumerate(values):
        payload = {"observation": {"measurements": [
            {"name": "value", "revision": 1, "value": value},
        ]}}
        insert = text(
            "INSERT INTO ai_metric_observations VALUES "
            "('ns', 'business.numeric', :occurred, :identity, CAST(:payload AS JSON));"
        ).bindparams(
            occurred=plan.start.replace(tzinfo=None) + timedelta(microseconds=index),
            identity=f"{index:064x}", payload=json.dumps(payload),
        )
        script += str(insert.compile(dialect=dialect, compile_kwargs={"literal_binds": True})) + "\n"
    fields = ", ".join(f"'{column}', q.{column}" for column in (
        "scanned_count", "extracted_count", "invalid_count", "sample_count", "selected_value",
    ))
    script += f"SELECT JSON_OBJECT({fields}) FROM ({sql}) AS q;"
    completed = subprocess.run(command, input=script, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    rows = [json.loads(line) for line in completed.stdout.splitlines()]
    result = _decode_measurement_rows(rows, plan).rows
    ordered = sorted(values)
    if aggregation is MetricAggregation.MIN:
        expected = ordered[0]
    elif aggregation is MetricAggregation.MAX:
        expected = ordered[-1]
    else:
        expected = ordered[math.ceil(.5 * len(ordered)) - 1]
    assert len(result) == 1
    assert result[0].sample_count == len(values)
    assert result[0].selected_value == expected
    assert type(result[0].selected_value) is type(expected)
