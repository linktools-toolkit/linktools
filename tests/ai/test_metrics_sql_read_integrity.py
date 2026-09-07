#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite pushdown verifies canonical records without returning raw rows."""

import asyncio
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
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
from sqlalchemy import event, select, text, update
from sqlalchemy.ext.asyncio import create_async_engine
from linktools.ai.observe import _sql as sql_module

_START = datetime(2026, 9, 7, tzinfo=timezone.utc)
_WINDOW = MetricWindow.between(_START, _START + timedelta(seconds=2))


def _definitions() -> tuple[MetricDefinition, ...]:
    return (
        MetricDefinition(
            name="business.integrity.count", revision=1,
            observation_kind="business.integrity",
            source=MetricSource.observation_count(), metric_type=MetricType.COUNTER,
            unit="1", default_aggregation=MetricAggregation.SUM,
            query_fields=("status",),
        ),
        MetricDefinition(
            name="business.integrity.value", revision=1,
            observation_kind="business.integrity",
            source=MetricSource.measurement("value"), metric_type=MetricType.DISTRIBUTION,
            unit="1", default_aggregation=MetricAggregation.MIN,
            query_fields=("status",),
        ),
    )


def _observation(identity: str = "one") -> Observation:
    return Observation(
        version=1, observation_id=identity, kind="business.integrity",
        occurred_at=_START, source_namespace="workspace", tenant_id="tenant",
        status="SUCCEEDED", error_code=None, correlation={}, dimensions={},
        measurements=(MetricMeasurement("value", 1, 12.5),),
    )


async def _seed(metrics: Metrics, count: int = 1) -> None:
    for definition in _definitions():
        await metrics.define(definition)
    observations = tuple(_observation(str(index)) for index in range(count))
    for index in range(0, count, 128):
        await metrics.record_observations(observations[index:index + 128])


@pytest.mark.asyncio
@pytest.mark.parametrize("path_backed", (False, True))
@pytest.mark.parametrize(
    ("damage", "expected"),
    (
        ("envelope_version", ErrorCode.STORAGE_VERSION_UNSUPPORTED),
        ("observation_version", ErrorCode.STORAGE_VERSION_UNSUPPORTED),
        ("digest", ErrorCode.STORAGE_INTEGRITY_ERROR),
        ("value", ErrorCode.STORAGE_INTEGRITY_ERROR),
        ("identity", ErrorCode.STORAGE_INTEGRITY_ERROR),
        ("namespace", ErrorCode.STORAGE_INTEGRITY_ERROR),
        ("kind", ErrorCode.STORAGE_INTEGRITY_ERROR),
        ("time", ErrorCode.STORAGE_INTEGRITY_ERROR),
        ("dimensions", ErrorCode.STORAGE_INTEGRITY_ERROR),
        ("duplicate_measurement", ErrorCode.STORAGE_INTEGRITY_ERROR),
        ("duplicate_key", ErrorCode.STORAGE_INTEGRITY_ERROR),
        ("malformed_json", ErrorCode.STORAGE_INTEGRITY_ERROR),
    ),
)
async def test_pushdown_rejects_corrupt_records_before_filters(
    tmp_path: Path, path_backed: bool, damage: str, expected: ErrorCode,
) -> None:
    path = tmp_path / "integrity.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    metrics = (Metrics.sqlite(path, namespace="integrity") if path_backed
               else Metrics.sql(engine, namespace="integrity"))
    table = build_metrics_sql_metadata().tables["ai_metric_observations"]
    try:
        await provision_metrics_database(engine)
        await _seed(metrics)
        async with engine.begin() as connection:
            original = (await connection.execute(select(table))).mappings().one()
            payload = deepcopy(original["payload_json"])
            values = {}
            if damage == "envelope_version":
                payload["version"] = 999
            elif damage == "observation_version":
                payload["observation"]["version"] = 999
            elif damage == "digest":
                values["payload_digest"] = "0" * 64
            elif damage == "value":
                payload["observation"]["measurements"][0]["value"] = 999.0
            elif damage == "identity":
                payload["observation"]["observation_id"] = "other"
            elif damage == "namespace":
                payload["namespace"] = "other"
            elif damage == "kind":
                payload["observation"]["kind"] = "business.other"
            elif damage == "time":
                payload["observation"]["occurred_at"] = (_START + timedelta(seconds=1)).isoformat()
            elif damage == "dimensions":
                payload["observation"]["dimensions"] = {"other": 123}
            elif damage == "duplicate_measurement":
                payload["observation"]["measurements"] *= 2
            if damage in {"duplicate_key", "malformed_json"}:
                # SQLAlchemy's JSON serializer cannot generate duplicate keys.
                raw = json.dumps(payload)
                raw = raw[:-1] + ', "version":1}' if damage == "duplicate_key" else "{broken"
                await connection.execute(
                    text("UPDATE ai_metric_observations SET payload_json = :payload"),
                    {"payload": raw},
                )
            else:
                values["payload_json"] = payload
                await connection.execute(update(table).values(**values))
        queries = (
            MetricQuery("business.integrity.count", _WINDOW),
            MetricQuery("business.integrity.count", _WINDOW, filters={"status": "FAILED"}),
            MetricQuery("business.integrity.value", _WINDOW),
            MetricQuery("business.integrity.value", _WINDOW,
                        aggregation=MetricAggregation.PERCENTILE, percentile=.95),
        )
        for query in queries:
            with pytest.raises(AIError) as caught:
                await metrics.query(query)
            assert caught.value.code is expected
        # The pooled connection remains usable after a callback error.
        async with engine.begin() as connection:
            await connection.execute(update(table).values(
                payload_json=original["payload_json"], payload_digest=original["payload_digest"],
            ))
        assert (await metrics.query(queries[0])).points[0].value == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_validation_is_once_per_record_and_one_observation_statement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'once.db'}")
    metrics = Metrics.sql(engine, namespace="once")
    calls = []
    statements = []
    original = sql_module._decode_observation_record

    def decode(namespace: str, key: str, row: object) -> Observation:
        calls.append(row)
        return original(namespace, key, row)

    def before_execute(conn: object, cursor: object, statement: str,
                       parameters: object, context: object, many: bool) -> None:
        if "ai_metric_observations" in statement:
            statements.append(statement)

    try:
        await provision_metrics_database(engine)
        await _seed(metrics, 1500)
        monkeypatch.setattr(sql_module, "_decode_observation_record", decode)
        event.listen(engine.sync_engine, "before_cursor_execute", before_execute)
        for name in ("business.integrity.count", "business.integrity.value"):
            for filters in ({}, {"status": "FAILED"}):
                calls.clear()
                statements.clear()
                await metrics.query(MetricQuery(name, _WINDOW, filters=filters))
                assert len(calls) == 1500
                assert len(statements) == 1
    finally:
        if event.contains(engine.sync_engine, "before_cursor_execute", before_execute):
            event.remove(engine.sync_engine, "before_cursor_execute", before_execute)
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("in_memory", (False, True))
async def test_sqlite_validation_is_isolated_between_concurrent_namespaces(tmp_path: Path, in_memory: bool) -> None:
    path = tmp_path / "snapshot.db"
    engine = create_async_engine("sqlite+aiosqlite:///:memory:" if in_memory else f"sqlite+aiosqlite:///{path}")
    metrics = Metrics.sql(engine, namespace="snapshot")
    try:
        await provision_metrics_database(engine)
        await _seed(metrics, 10)
        # Distinct namespaces must not share a connection-local verifier closure.
        other = Metrics.sql(engine, namespace="other")
        await _seed(other, 3)
        for _ in range(5):
            results = await asyncio.gather(
                metrics.query(MetricQuery("business.integrity.count", _WINDOW)),
                other.query(MetricQuery("business.integrity.count", _WINDOW)),
            )
            assert [result.points[0].value for result in results] == [10, 3]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_verifier_is_reinstalled_after_connection_invalidation(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'reconnect.db'}")
    try:
        await provision_metrics_database(engine)
        # Install onto an engine that already owns pooled connections.
        metrics = Metrics.sql(engine, namespace="reconnect")
        await _seed(metrics)
        query = MetricQuery("business.integrity.count", _WINDOW)
        assert (await metrics.query(query)).points[0].value == 1
        async with engine.connect() as connection:
            await connection.invalidate()
        assert (await metrics.query(query)).points[0].value == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_scan_limit_prevents_record_decoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.ai.observe import _query

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'limit.db'}")
    metrics = Metrics.sql(engine, namespace="limit")
    try:
        await provision_metrics_database(engine)
        await _seed(metrics, 4)
        async with engine.begin() as connection:
            await connection.execute(text("UPDATE ai_metric_observations SET payload_json = '{broken'"))
        monkeypatch.setattr(_query, "_MAX_SCANNED_OBSERVATIONS", 3)

        def decode(*args: object) -> Observation:
            raise AssertionError("over-limit query decoded a record")

        monkeypatch.setattr(sql_module, "_decode_observation_record", decode)
        for name in ("business.integrity.count", "business.integrity.value"):
            with pytest.raises(AIError) as caught:
                await metrics.query(MetricQuery(name, _WINDOW))
            assert caught.value.code is ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_verification_preserves_codec_normalization(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'normalization.db'}")
    metrics = Metrics.sql(engine, namespace="normalization")
    table = build_metrics_sql_metadata().tables["ai_metric_observations"]
    try:
        await provision_metrics_database(engine)
        await _seed(metrics)
        query = MetricQuery("business.integrity.value", _WINDOW)
        expected = await metrics.query(query)
        async with engine.begin() as connection:
            row = (await connection.execute(select(table))).mappings().one()
            payload = deepcopy(row["payload_json"])
            for key in ("error_code", "dimensions", "correlation"):
                payload["observation"].pop(key)
            payload["future_optional_field"] = {"ignored": True}
            await connection.execute(
                text("UPDATE ai_metric_observations SET payload_json = :payload"),
                {"payload": json.dumps(payload, sort_keys=True, indent=2)},
            )
        assert await metrics.query(query) == expected
    finally:
        await engine.dispose()
