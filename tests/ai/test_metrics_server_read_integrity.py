#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify real server reads and aggregation share one canonical snapshot."""

import asyncio
import getpass
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
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
    build_metrics_sql_metadata,
)
from linktools.ai.observe import _query
from linktools.ai.observe._sql import SqlMetricStore
from sqlalchemy import delete, event, select, text, update
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


_START = datetime(2026, 9, 7, tzinfo=timezone.utc)
_WINDOW = MetricWindow.between(_START, _START + timedelta(seconds=2))
_NAMESPACE = "server-integrity"
_COUNT = "business.integrity.count"
_RATIO = "business.integrity.ratio"
_VALUE = "business.integrity.value"
_ServerMetrics = tuple[Metrics, AsyncEngine, AsyncEngine]


def _observation(index: int = 0) -> Observation:
    return Observation(
        version=1, observation_id=f"observation-{index}", kind="business.integrity",
        occurred_at=_START + timedelta(microseconds=index), source_namespace="workspace",
        tenant_id="tenant", status="SUCCEEDED", error_code=None,
        correlation={"attempt": 2**53 + 1}, dimensions={"group": "中文"},
        measurements=(MetricMeasurement("value", 1, index + 1),),
    )


@pytest_asyncio.fixture
async def server_metrics(_server: tuple[str, list[str]]) -> AsyncIterator[_ServerMetrics]:
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
    writer = create_async_engine(url, isolation_level="READ COMMITTED", pool_size=1, max_overflow=0)
    metadata = build_metrics_sql_metadata()
    try:
        async with engine.begin() as connection:
            await connection.run_sync(metadata.drop_all)
        await provision_metrics_database(engine)
        metrics = Metrics.sql(engine, namespace=_NAMESPACE)
        for name, source, metric_type in (
            (_COUNT, MetricSource.observation_count(), MetricType.COUNTER),
            (_RATIO, MetricSource.indicator("status", ("FAILED",)), MetricType.RATIO),
            (_VALUE, MetricSource.measurement("value"), MetricType.GAUGE),
        ):
            await metrics.define(MetricDefinition(
                name=name, revision=1, observation_kind="business.integrity", source=source,
                metric_type=metric_type, unit="1", default_aggregation=MetricAggregation.COUNT,
                query_fields=("group", "status"),
            ))
        yield metrics, engine, writer
    finally:
        await writer.dispose()
        await engine.dispose()


async def _payload(engine: AsyncEngine) -> dict[str, object]:
    table = build_metrics_sql_metadata().tables["ai_metric_observations"]
    async with engine.connect() as connection:
        return dict((await connection.execute(select(table.c.payload_json))).scalar_one())


async def _write_payload(engine: AsyncEngine, payload: dict[str, object]) -> None:
    table = build_metrics_sql_metadata().tables["ai_metric_observations"]
    async with engine.begin() as connection:
        await connection.execute(update(table).values(payload_json=payload))


@pytest.mark.asyncio
@pytest.mark.parametrize("name", (_COUNT, _RATIO, _VALUE))
@pytest.mark.parametrize("damage", (
    "envelope_version", "observation_version", "namespace", "value", "identity",
    "timestamp", "digest", "duplicate_measurement", "boolean_measurement",
))
async def test_corrupt_record_fails_before_any_filter_or_aggregation(
    server_metrics: _ServerMetrics, name: str, damage: str,
) -> None:
    metrics, engine, _ = server_metrics
    await metrics.record_observations((_observation(),))
    table = build_metrics_sql_metadata().tables["ai_metric_observations"]
    if damage in {"identity", "timestamp", "digest"}:
        values = {
            "identity": {"observation_digest": "0" * 64},
            "timestamp": {"occurred_at": _START + timedelta(microseconds=1)},
            "digest": {"payload_digest": "0" * 64},
        }[damage]
        async with engine.begin() as connection:
            await connection.execute(update(table).values(**values))
    else:
        payload = await _payload(engine)
        observation = payload["observation"]
        if damage == "envelope_version":
            payload["version"] = 999
        elif damage == "observation_version":
            observation["version"] = 999
        elif damage == "namespace":
            payload["namespace"] = "other"
        elif damage == "duplicate_measurement":
            observation["measurements"] *= 2
        else:
            observation["measurements"][0]["value"] = True if damage == "boolean_measurement" else 99
        await _write_payload(engine, payload)
    code = ErrorCode.STORAGE_VERSION_UNSUPPORTED if damage.endswith("version") else ErrorCode.STORAGE_INTEGRITY_ERROR
    # The invalid SUCCEEDED record must not disappear behind this FAILED filter.
    with pytest.raises(AIError) as raised:
        await metrics.query(MetricQuery(name, _WINDOW, filters={"status": "FAILED"}))
    assert raised.value.code is code
    async with engine.connect() as connection:
        assert await connection.get_isolation_level() == "READ COMMITTED"


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", (False, True))
async def test_global_count_and_missing_measurement_do_not_bypass_verification(
    server_metrics: _ServerMetrics, missing: bool,
) -> None:
    metrics, engine, _ = server_metrics
    observation = _observation()
    if missing:
        observation = replace(observation, measurements=())
    await metrics.record_observations((observation,))
    payload = await _payload(engine)
    payload["version"] = 999
    await _write_payload(engine, payload)
    name = _VALUE if missing else _COUNT
    with pytest.raises(AIError) as raised:
        await metrics.query(MetricQuery(name, _WINDOW))
    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


@pytest.mark.asyncio
async def test_scan_limit_precedes_payload_decoding(
    server_metrics: _ServerMetrics, monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics, engine, _ = server_metrics
    await metrics.record_observations(tuple(_observation(index) for index in range(3)))
    table = build_metrics_sql_metadata().tables["ai_metric_observations"]
    async with engine.begin() as connection:
        await connection.execute(update(table).values(payload_json="not an envelope"))
    monkeypatch.setattr(_query, "_MAX_SCANNED_OBSERVATIONS", 2)
    with pytest.raises(AIError) as raised:
        await metrics.query(MetricQuery(_COUNT, _WINDOW))
    assert raised.value.code is ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize("name", (_COUNT, _VALUE))
@pytest.mark.parametrize("change", ("insert", "tamper", "prune"))
async def test_verification_and_aggregation_keep_the_same_snapshot(
    server_metrics: _ServerMetrics, monkeypatch: pytest.MonkeyPatch, change: str, name: str,
) -> None:
    metrics, engine, writer = server_metrics
    await metrics.record_observations((_observation(),))
    verify = SqlMetricStore._verify_query_records
    table = build_metrics_sql_metadata().tables["ai_metric_observations"]

    async def verify_then_write(self: SqlMetricStore, *args: object, **kwargs: object) -> int:
        count = await verify(self, *args, **kwargs)
        if change == "insert":
            await Metrics.sql(writer, namespace=_NAMESPACE).record_observations((_observation(1),))
        else:
            async with writer.begin() as connection:
                statement = delete(table) if change == "prune" else update(table).values(payload_digest="0" * 64)
                await connection.execute(statement)
        return count

    monkeypatch.setattr(SqlMetricStore, "_verify_query_records", verify_then_write)
    result = await asyncio.wait_for(metrics.query(MetricQuery(name, _WINDOW)), timeout=10)
    assert result.points[0].value == 1
    monkeypatch.setattr(SqlMetricStore, "_verify_query_records", verify)
    if change == "tamper":
        with pytest.raises(AIError) as raised:
            await metrics.query(MetricQuery(name, _WINDOW))
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    else:
        result = await metrics.query(MetricQuery(name, _WINDOW))
        assert result.points[0].value == (2 if change == "insert" else 0)
    async with engine.connect() as connection:
        assert await connection.get_isolation_level() == "READ COMMITTED"


@pytest.mark.asyncio
async def test_cancelled_query_releases_its_snapshot(
    server_metrics: _ServerMetrics, monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics, engine, _ = server_metrics
    await metrics.record_observations((_observation(),))
    verify = SqlMetricStore._verify_query_records
    verified = asyncio.Event()
    block = asyncio.Event()

    async def verify_then_block(self: SqlMetricStore, *args: object, **kwargs: object) -> int:
        count = await verify(self, *args, **kwargs)
        verified.set()
        await block.wait()
        return count

    monkeypatch.setattr(SqlMetricStore, "_verify_query_records", verify_then_block)
    task = asyncio.create_task(metrics.query(MetricQuery(_COUNT, _WINDOW)))
    try:
        await asyncio.wait_for(verified.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    monkeypatch.setattr(SqlMetricStore, "_verify_query_records", verify)
    result = await asyncio.wait_for(metrics.query(MetricQuery(_COUNT, _WINDOW)), timeout=10)
    assert result.points[0].value == 1
    async with engine.connect() as connection:
        assert await connection.get_isolation_level() == "READ COMMITTED"


@pytest.mark.asyncio
@pytest.mark.parametrize("count", (0, 270))
async def test_server_query_streams_verification_once_and_still_aggregates_in_sql(
    server_metrics: _ServerMetrics, monkeypatch: pytest.MonkeyPatch, count: int,
) -> None:
    metrics, engine, _ = server_metrics
    observations = tuple(_observation(index) for index in range(count))
    for offset in range(0, count, 128):
        await metrics.record_observations(observations[offset:offset + 128])
    statements: list[str] = []

    def record_statement(
        connection: object, cursor: object, statement: str, parameters: object,
        context: object, executemany: bool,
    ) -> None:
        if "ai_metric_observations" in statement.lower():
            statements.append(statement)

    monkeypatch.setattr(SqlMetricStore, "scan_observations", AsyncMock(side_effect=AssertionError("unexpected scan fallback")))
    event.listen(engine.sync_engine, "before_cursor_execute", record_statement)
    try:
        result = await metrics.query(MetricQuery(_VALUE, _WINDOW, aggregation=MetricAggregation.MAX))
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", record_statement)
    assert result.points[0].value == (count or None)
    assert result.points[0].sample_count == count
    assert len(statements) == 2
    assert "metric_verification" in statements[0]
    assert "reduced" in statements[1]
    assert all("FOR UPDATE" not in statement.upper() for statement in statements)


@pytest.mark.asyncio
@pytest.mark.parametrize("_server", ("postgresql",), indirect=True)
async def test_server_json_text_does_not_erase_duplicate_members(
    server_metrics: _ServerMetrics,
) -> None:
    metrics, engine, _ = server_metrics
    await metrics.record_observations((_observation(),))
    payload = await _payload(engine)
    value = json.dumps(payload)
    value = value[:-1] + ', "version": 1}'
    async with engine.begin() as connection:
        await connection.execute(text("UPDATE ai_metric_observations SET payload_json = CAST(:payload AS JSON)"), {"payload": value})
    with pytest.raises(AIError) as raised:
        await metrics.query(MetricQuery(_COUNT, _WINDOW))
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("count", (0, 1025))
async def test_global_count_reuses_only_fully_verified_sql_count(
    server_metrics: _ServerMetrics, count: int,
) -> None:
    metrics, engine, _ = server_metrics
    for offset in range(0, count, 256):
        await metrics.record_observations(tuple(
            _observation(index) for index in range(offset, min(offset + 256, count))
        ))
    statements: list[str] = []

    def record_statement(
        connection: object, cursor: object, statement: str, parameters: object,
        context: object, executemany: bool,
    ) -> None:
        if "ai_metric_observations" in statement:
            statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", record_statement)
    try:
        for aggregation in (MetricAggregation.COUNT, MetricAggregation.SUM, MetricAggregation.RATE):
            statements.clear()
            result = await metrics.query(MetricQuery(_COUNT, _WINDOW, aggregation=aggregation))
            assert result.points[0].value == (count / 2 if aggregation is MetricAggregation.RATE else count)
            assert result.points[0].sample_count == count
            assert len(statements) == 1
            assert "metric_verification" in statements[0]
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", record_statement)
    if count:
        table = build_metrics_sql_metadata().tables["ai_metric_observations"]
        async with engine.begin() as connection:
            await connection.execute(update(table).where(
                table.c.occurred_at == _START + timedelta(microseconds=count - 1),
            ).values(payload_digest="0" * 64))
        with pytest.raises(AIError) as raised:
            await metrics.query(MetricQuery(_COUNT, _WINDOW))
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
