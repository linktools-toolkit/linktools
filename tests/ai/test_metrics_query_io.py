#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare complete verified query costs against the scan executor."""

import getpass
import json
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import timedelta
from typing import cast

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from linktools.ai.migrate import provision_metrics_database
from linktools.ai.observe import (
    MetricAggregation, MetricDefinition, MetricQuery, MetricSource, MetricType,
    Metrics, MetricStore, build_metrics_sql_metadata,
)
from linktools.ai.observe import _sql
from linktools.ai.observe._codec import (
    observation_digest, observation_envelope, observation_payload_digest,
)
from linktools.ai.observe._sql import SqlMetricStore
from linktools.ai.storage import namespace_digest

from .test_metrics_server_read_integrity import _COUNT, _NAMESPACE, _WINDOW, _observation
from .test_metrics_sql_server_sums import _server

_IOData = tuple[Metrics, Metrics, AsyncEngine, int]


class _ScanStore:
    def __init__(self, store: SqlMetricStore) -> None:
        self.get_definition = store.get_definition
        self.latest_definition = store.latest_definition
        self.scan_observations = store.scan_observations


@pytest_asyncio.fixture(scope="module", loop_scope="module", params=(1000, 10000, 100000))
async def _io_data(
    _server: tuple[str, list[str]], request: pytest.FixtureRequest,
) -> AsyncIterator[_IOData]:
    name, command = _server
    if name == "postgresql":
        url = URL.create(
            "postgresql+asyncpg", username=getpass.getuser(), database="postgres",
            query={"host": command[command.index("-h") + 1]},
        )
    else:
        socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
        url = URL.create(
            "mysql+asyncmy", username="root", database="metrics_test",
            query={"unix_socket": socket, "charset": "utf8mb4"},
        )
    engine = create_async_engine(url, isolation_level="READ COMMITTED", pool_size=1, max_overflow=0)
    count = int(request.param)
    try:
        metadata = build_metrics_sql_metadata()
        async with engine.begin() as connection:
            await connection.run_sync(metadata.drop_all)
        await provision_metrics_database(engine)
        metrics = Metrics.sql(engine, namespace=_NAMESPACE)
        await metrics.define(MetricDefinition(
            name=_COUNT, revision=1, observation_kind="business.integrity",
            source=MetricSource.observation_count(), metric_type=MetricType.COUNTER,
            unit="1", default_aggregation=MetricAggregation.COUNT,
        ))
        definition = MetricDefinition(
            name="business.integrity.io", revision=1,
            observation_kind="business.integrity", source=MetricSource.measurement("value"),
            metric_type=MetricType.DISTRIBUTION, unit="1",
            default_aggregation=MetricAggregation.MEAN, query_fields=("group",),
        )
        await metrics.define(definition)
        table = metadata.tables["ai_metric_observations"]
        namespace_key = namespace_digest(_NAMESPACE)
        async with engine.begin() as connection:
            for offset in range(0, count, 512):
                rows = []
                for index in range(offset, min(offset + 512, count)):
                    observation = replace(
                        _observation(index), dimensions={"group": f"group-{index % 10}"},
                    )
                    rows.append({
                        "namespace_digest": namespace_key,
                        "observation_digest": observation_digest(_NAMESPACE, observation.observation_id),
                        "payload_digest": observation_payload_digest(_NAMESPACE, observation),
                        "kind": observation.kind, "occurred_at": observation.occurred_at,
                        "payload_json": observation_envelope(_NAMESPACE, observation),
                    })
                await connection.execute(table.insert(), rows)
        store = SqlMetricStore(engine)
        await store.latest_definition(_NAMESPACE, _COUNT)
        scan = Metrics.from_store(cast(MetricStore, _ScanStore(store)), namespace=_NAMESPACE)
        yield metrics, scan, engine, count
    finally:
        await engine.dispose()


async def _database_work(
    engine: AsyncEngine, statements: list[tuple[str, object]],
) -> list[dict[str, object]]:
    measured = []
    async with engine.connect() as connection:
        for statement, parameters in statements:
            if engine.dialect.name == "postgresql":
                row = (await connection.exec_driver_sql(
                    "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + statement, parameters,
                )).scalar_one()
                document = json.loads(row) if isinstance(row, str) else row
                plan = document[0]["Plan"]
                nodes = [plan]
                source_rows = 0
                while nodes:
                    node = nodes.pop()
                    if node.get("Relation Name") == "ai_metric_observations":
                        source_rows += node["Actual Rows"] * node["Actual Loops"]
                    nodes.extend(node.get("Plans", ()))
                measured.append({
                    "source_rows": source_rows,
                    "shared_read_blocks": plan.get("Shared Read Blocks", 0),
                    "shared_hit_blocks": plan.get("Shared Hit Blocks", 0),
                    "temp_read_blocks": plan.get("Temp Read Blocks", 0),
                    "temp_written_blocks": plan.get("Temp Written Blocks", 0),
                })
            else:
                counter = (
                    "SELECT COALESCE(SUM(COUNT_FETCH), 0) FROM "
                    "performance_schema.table_io_waits_summary_by_table "
                    "WHERE OBJECT_SCHEMA = DATABASE() AND OBJECT_NAME = 'ai_metric_observations'"
                )
                before = int((await connection.exec_driver_sql(counter)).scalar_one())
                plan = (await connection.exec_driver_sql(
                    "EXPLAIN ANALYZE " + statement, parameters,
                )).scalar_one()
                after = int((await connection.exec_driver_sql(counter)).scalar_one())
                measured.append({"source_fetches": after - before, "plan": plan})
    return measured


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize("query_name", ("count", "mean", "p95"))
@pytest.mark.parametrize("iteration", (0, 1))
async def test_complete_query_io(
    _io_data: _IOData, query_name: str, iteration: int,
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics, scan, engine, count = _io_data
    queries = {
        "count": MetricQuery(_COUNT, _WINDOW),
        "mean": MetricQuery("business.integrity.io", _WINDOW, group_by=("group",), bucket=timedelta(seconds=1)),
        "p95": MetricQuery(
            "business.integrity.io", _WINDOW, aggregation=MetricAggregation.PERCENTILE,
            percentile=.95, group_by=("group",), bucket=timedelta(seconds=1),
        ),
    }
    statements: list[tuple[str, object]] = []
    verified = 0
    decoder = _sql._decode_observation_record

    def decode(*args: object, **kwargs: object) -> object:
        nonlocal verified
        verified += 1
        return decoder(*args, **kwargs)

    def before_execute(
        connection: object, cursor: object, statement: str, parameters: object,
        context: object, executemany: bool,
    ) -> None:
        if "ai_metric_observations" in statement and statement.lstrip().upper().startswith(("SELECT", "WITH")):
            statements.append((statement, parameters))

    monkeypatch.setattr(_sql, "_decode_observation_record", decode)
    event.listen(engine.sync_engine, "before_cursor_execute", before_execute)
    paths = (("scan", scan), ("pushdown", metrics))
    if iteration:
        paths = tuple(reversed(paths))
    expected = None
    aggregate_statements = []
    try:
        for path, facade in paths:
            statements.clear()
            verified = 0
            started = time.perf_counter()
            cpu_started = time.process_time()
            result = await facade.query(queries[query_name])
            cpu = time.process_time() - cpu_started
            elapsed = time.perf_counter() - started
            if expected is None:
                expected = result
            else:
                assert result == expected
            assert verified == count
            assert sum(point.sample_count for point in result.points) == count
            if path == "pushdown":
                assert len(statements) == (1 if query_name == "count" else 2)
                aggregate_statements = list(statements)
            else:
                assert len(statements) == (count + 511) // 512
            with capsys.disabled():
                print("METRICS_QUERY_IO " + json.dumps({
                    "backend": engine.dialect.name, "rows": count,
                    "query": query_name, "path": path, "iteration": iteration,
                    "seconds": round(elapsed, 6), "python_cpu_seconds": round(cpu, 6),
                    "observation_statements": len(statements),
                    "points": len(result.points), "verified_records": verified,
                }, sort_keys=True), flush=True)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", before_execute)
    if iteration:
        work = await _database_work(engine, aggregate_statements)
        with capsys.disabled():
            print("METRICS_QUERY_PLAN " + json.dumps({
                "backend": engine.dialect.name, "rows": count,
                "query": query_name, "statements": work,
            }, sort_keys=True), flush=True)
