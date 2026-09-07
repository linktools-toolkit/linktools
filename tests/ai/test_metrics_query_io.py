#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare complete SQL queries with the backend-neutral scan executor."""

import json
import time
from dataclasses import replace
from datetime import timedelta
from typing import cast

import pytest
from sqlalchemy import event

from linktools.ai.observe import (
    MetricAggregation, MetricDefinition, MetricQuery, MetricSource, MetricType,
    Metrics, MetricStore,
)
from linktools.ai.observe._sql import SqlMetricStore

from .test_metrics_server_read_integrity import (
    _COUNT, _NAMESPACE, _ServerMetrics, _WINDOW, _observation, _server, server_metrics,
)


class _ScanStore:
    def __init__(self, store: SqlMetricStore) -> None:
        self.get_definition = store.get_definition
        self.latest_definition = store.latest_definition
        self.scan_observations = store.scan_observations


@pytest.mark.asyncio
@pytest.mark.parametrize("sample_count", (1000, 10000, 100000))
async def test_complete_query_io(
    server_metrics: _ServerMetrics, sample_count: int, capsys: pytest.CaptureFixture[str],
) -> None:
    metrics, engine, _ = server_metrics
    definition = MetricDefinition(
        name="business.integrity.io", revision=1,
        observation_kind="business.integrity", source=MetricSource.measurement("value"),
        metric_type=MetricType.DISTRIBUTION, unit="1",
        default_aggregation=MetricAggregation.MEAN, query_fields=("group",),
    )
    await metrics.define(definition)
    for offset in range(0, sample_count, 256):
        await metrics.record_observations(tuple(
            replace(_observation(index), dimensions={"group": f"group-{index % 10}"})
            for index in range(offset, min(offset + 256, sample_count))
        ))
    scan = Metrics.from_store(
        cast(MetricStore, _ScanStore(SqlMetricStore(engine))), namespace=_NAMESPACE,
    )
    queries = {
        "count": MetricQuery(_COUNT, _WINDOW),
        "mean": MetricQuery(definition.name, _WINDOW, group_by=("group",), bucket=timedelta(seconds=1)),
        "p95": MetricQuery(
            definition.name, _WINDOW, aggregation=MetricAggregation.PERCENTILE,
            percentile=.95, group_by=("group",), bucket=timedelta(seconds=1),
        ),
    }
    statements: list[str] = []

    def before_execute(
        connection: object, cursor: object, statement: str, parameters: object,
        context: object, executemany: bool,
    ) -> None:
        if "ai_metric_observations" in statement and statement.lstrip().upper().startswith(("SELECT", "WITH")):
            statements.append(statement)

    # Initialize both adapters before measuring warm query execution.
    await scan.query(queries["count"])
    await metrics.query(queries["count"])
    event.listen(engine.sync_engine, "before_cursor_execute", before_execute)
    try:
        for name, query in queries.items():
            expected = None
            measurements = []
            for iteration in range(2):
                paths = (("scan", scan), ("pushdown", metrics))
                if iteration:
                    paths = tuple(reversed(paths))
                for path, facade in paths:
                    statements.clear()
                    started = time.perf_counter()
                    result = await facade.query(query)
                    elapsed = time.perf_counter() - started
                    if expected is None:
                        expected = result
                    else:
                        assert result == expected
                    assert sum(point.sample_count for point in result.points) == sample_count
                    measurement = {
                        "backend": engine.dialect.name, "rows": sample_count,
                        "query": name, "path": path, "iteration": iteration,
                        "seconds": round(elapsed, 6),
                        "observation_statements": len(statements),
                        "points": len(result.points),
                        "payload_rows_readback": sample_count,
                    }
                    measurements.append(measurement)
                    if path == "pushdown":
                        assert len(statements) <= 2
                    else:
                        assert len(statements) == (sample_count + 511) // 512
            with capsys.disabled():
                for measurement in measurements:
                    print("METRICS_QUERY_IO " + json.dumps(measurement, sort_keys=True), flush=True)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", before_execute)
