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

from .test_metrics_sql_server_sums import _plan


@pytest.mark.parametrize("_server", ("mysql",), indirect=True)
@pytest.mark.parametrize("values", (
    (1.0000000000000002e17, 100000000000000018),
    (-1.0000000000000002e17, -100000000000000018),
    (2**63 - 3, 2**63 - 2, 2**63 - 1, float(2**63)),
    (-2**63, -2**63 + 1, float(-2**63)),
    (2**53 - 1, 2**53, 2**53 + 1, float(2**53)),
    (0, -0.0, 1.0, 1),
    (5e-324, 0, -5e-324),
    (1.25, -2.5, 3, 4.5),
))
@pytest.mark.parametrize(("aggregation", "percentile"), (
    (MetricAggregation.MIN, None),
    (MetricAggregation.MAX, None),
    (MetricAggregation.PERCENTILE, .5),
    (MetricAggregation.PERCENTILE, .95),
))
@pytest.mark.parametrize("partitioned", (False, True))
def test_mysql_selected_value_matches_python(
    _server: tuple[str, list[str]], values: tuple[int | float, ...],
    aggregation: MetricAggregation, percentile: float | None, partitioned: bool,
) -> None:
    name, command = _server
    plan = replace(
        _plan(), metric_type=MetricType.DISTRIBUTION, aggregation=aggregation,
        percentile=percentile, group_by=("group",) if partitioned else (),
        bucket_microseconds=1_000_000 if partitioned else None,
        bucket_count=2 if partitioned else 1,
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
    partitions = ((None, 0), ("x", 0), (None, 1), ("x", 1)) if partitioned else ((None, 0),)
    for partition_index, (group, bucket_index) in enumerate(partitions):
        for index, value in enumerate(values):
            payload = {"observation": {
                "dimensions": {} if group is None else {"group": group},
                "measurements": [{"name": "value", "revision": 1, "value": value}],
            }}
            insert = text(
                "INSERT INTO ai_metric_observations VALUES "
                "('ns', 'business.numeric', :occurred, :identity, CAST(:payload AS JSON));"
            ).bindparams(
                occurred=plan.start.replace(tzinfo=None) + timedelta(
                    seconds=bucket_index, microseconds=index,
                ),
                identity=f"{partition_index * len(values) + index:064x}",
                payload=json.dumps(payload),
            )
            script += str(insert.compile(dialect=dialect, compile_kwargs={"literal_binds": True})) + "\n"
    columns = ["scanned_count", "extracted_count", "invalid_count", "sample_count", "selected_value"]
    if partitioned:
        columns.extend(("g0", "bucket_index"))
    fields = ", ".join(f"'{column}', q.{column}" for column in columns)
    script += f"SELECT JSON_OBJECT({fields}) FROM ({sql}) AS q;"
    completed = subprocess.run(command, input=script, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    rows = [json.loads(line) for line in completed.stdout.splitlines()]
    result = _decode_measurement_rows(rows, plan).rows
    if aggregation is MetricAggregation.MIN:
        expected = min(values)
    elif aggregation is MetricAggregation.MAX:
        expected = max(values)
    else:
        assert percentile is not None
        expected = sorted(values)[math.ceil(percentile * len(values)) - 1]
    expected_partitions = {((group,), bucket) for group, bucket in partitions} if partitioned else {((), None)}
    assert {(row.group, row.bucket_index) for row in result} == expected_partitions
    for row in result:
        assert row.sample_count == len(values)
        assert row.selected_value == expected
        assert type(row.selected_value) is type(expected)
