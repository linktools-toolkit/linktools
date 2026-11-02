#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retention keeps exact replay proofs without exposing removed observations."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_metrics_database
from linktools.ai.observe import (
    MetricAggregation, MetricDefinition, MetricMeasurement, MetricQuery,
    MetricSource, MetricType, MetricWindow, Metrics, Observation,
    build_metrics_sql_metadata,
)
from linktools.ai.observe import _query
from linktools.ai.observe._memory import InMemoryMetricStore
from linktools.ai.observe._sql import SqlMetricStore
from linktools.ai.observe._sqlite import SQLiteMetricStore
from linktools.ai.observe._store import MetricStore

_START = datetime(2026, 10, 10, tzinfo=timezone.utc)
_BEFORE = _START + timedelta(seconds=1)
_END = _START + timedelta(seconds=2)
_REASON = "metric_observation_pruned"


def _observation(identity: str, *, occurred_at: datetime = _START) -> Observation:
    return Observation(
        version=1, observation_id=identity, kind="app.retention",
        occurred_at=occurred_at, source_namespace="source", tenant_id="tenant",
        status="SUCCEEDED", error_code=None, correlation={}, dimensions={},
        measurements=(MetricMeasurement("value", 1, 3),),
    )


@pytest_asyncio.fixture(params=("memory", "sqlite", "sql"))
async def retention_store(
    request: pytest.FixtureRequest, tmp_path: Path,
) -> AsyncIterator[MetricStore]:
    if request.param == "memory":
        yield InMemoryMetricStore()
        return
    path = tmp_path / "retention.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    try:
        await provision_metrics_database(engine)
        yield SQLiteMetricStore(path) if request.param == "sqlite" else SqlMetricStore(engine)
    finally:
        await engine.dispose()


async def _assert_retention(metrics: Metrics) -> None:
    old, active, late = (_observation("old"), _observation("active", occurred_at=_BEFORE),
                         _observation("late"))
    await metrics.record_observations((old, active))
    assert await metrics.prune(before=_BEFORE) == 1
    assert await metrics.prune(before=_BEFORE) == 0
    assert await metrics.get_observation("old") is None
    assert await metrics.get_observation("active") == active
    with pytest.raises(AIError) as caught:
        await metrics.record_observations((old,))
    assert caught.value.code is ErrorCode.STORAGE_CONFLICT
    assert caught.value.retryable is False
    assert caught.value.safe_details == {"reason": _REASON}
    with pytest.raises(AIError) as conflict:
        await metrics.record_observations((replace(old, status="FAILED"),))
    assert conflict.value.code is ErrorCode.STORAGE_CONFLICT
    assert conflict.value.safe_details.get("reason") != _REASON
    # Prune is not a permanent occurrence-time admission floor.
    await metrics.record_observations((late,))
    assert await metrics.get_observation("late") == late
    assert await metrics.prune(before=_BEFORE) == 1
    # Both an earlier insert and a later digest conflict must retain atomicity.
    fresh = _observation("fresh")
    with pytest.raises(AIError) as batch:
        await metrics.record_observations((fresh, old))
    assert batch.value.safe_details == {"reason": _REASON}
    assert await metrics.get_observation("fresh") is None
    with pytest.raises(AIError) as conflict_batch:
        await metrics.record_observations((old, fresh, replace(active, status="FAILED")))
    assert conflict_batch.value.safe_details.get("reason") != _REASON
    assert await metrics.get_observation("fresh") is None


@pytest.mark.asyncio
async def test_retention_preserves_identity_and_atomic_batch_contract(retention_store: MetricStore) -> None:
    metrics = Metrics.from_store(retention_store, namespace="retention")
    await _assert_retention(metrics)
    other = Metrics.from_store(retention_store, namespace="other")
    await other.record_observations((_observation("old"),))
    assert await other.get_observation("old") == _observation("old")
    page = await retention_store.scan_observations(
        "retention", "app.retention", _START, _END, cursor=None, limit=1,
    )
    assert [item.observation_id for item in page.items] == ["active"]
    assert page.next_cursor is None


@pytest.mark.asyncio
async def test_server_retention_contract(
    server_metrics: tuple[Metrics, AsyncEngine, AsyncEngine], monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics, engine, writer = server_metrics
    await _assert_retention(metrics)
    monkeypatch.setattr(_query, "_MAX_SCANNED_OBSERVATIONS", 1)
    await _assert_query_retention(Metrics.sql(engine, namespace="retention-query"))
    race = _observation("server-race")
    await metrics.record_observations((race,))
    replay = Metrics.sql(writer, namespace=metrics.namespace)
    results = await asyncio.gather(
        metrics.prune(before=_BEFORE), replay.record_observations((race,)),
        return_exceptions=True,
    )
    assert results[0] == 1
    assert results[1] is None or (
        isinstance(results[1], AIError) and results[1].safe_details == {"reason": _REASON}
    )
    assert await replay.get_observation("server-race") is None


@pytest.mark.asyncio
async def test_retention_precedes_scan_limits_and_all_aggregations(
    retention_store: MetricStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = Metrics.from_store(retention_store, namespace="retention")
    monkeypatch.setattr(_query, "_MAX_SCANNED_OBSERVATIONS", 1)
    await _assert_query_retention(metrics)


async def _assert_query_retention(metrics: Metrics) -> None:
    for name, source, metric_type, aggregation in (
        ("count", MetricSource.observation_count(), MetricType.COUNTER, MetricAggregation.COUNT),
        ("value", MetricSource.measurement("value"), MetricType.DISTRIBUTION, MetricAggregation.MEAN),
    ):
        await metrics.define(MetricDefinition(
            name=name, revision=1, observation_kind="app.retention", source=source,
            metric_type=metric_type, unit="1", default_aggregation=aggregation,
        ))
    await metrics.record_observations(tuple(_observation(str(index)) for index in range(3)))
    assert await metrics.prune(before=_BEFORE) == 3
    await metrics.record_observations((_observation("active", occurred_at=_BEFORE),))
    for name, aggregation, expected in (
        ("count", MetricAggregation.COUNT, 1), ("value", MetricAggregation.MEAN, 3),
        ("value", MetricAggregation.PERCENTILE, 3),
    ):
        result = await metrics.query(MetricQuery(
            name, MetricWindow.between(_START, _END), aggregation=aggregation,
            percentile=.95 if aggregation is MetricAggregation.PERCENTILE else None,
        ))
        assert result.points[0].sample_count == 1
        assert result.points[0].value == expected
    assert await metrics.prune(before=_END) == 1
    result = await metrics.query(MetricQuery("count", MetricWindow.between(_START, _END)))
    assert result.points[0].value == 0


@pytest.mark.asyncio
async def test_prune_replay_race_never_reactivates(retention_store: MetricStore) -> None:
    metrics = Metrics.from_store(retention_store, namespace="retention")
    old = _observation("race")
    await metrics.record_observations((old,))
    results = await asyncio.gather(
        metrics.prune(before=_BEFORE), metrics.record_observations((old,)),
        return_exceptions=True,
    )
    assert results[0] == 1
    if results[1] is not None:
        assert isinstance(results[1], AIError)
        assert results[1].safe_details == {"reason": _REASON}
    assert await metrics.get_observation("race") is None
    with pytest.raises(AIError) as caught:
        await metrics.record_observations((old,))
    assert caught.value.safe_details == {"reason": _REASON}


@pytest.mark.asyncio
async def test_sql_retains_minimal_original_receipt_and_first_pruned_time(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'receipt.db'}")
    table = build_metrics_sql_metadata().tables["ai_metric_observations"]
    metrics = Metrics.sql(engine, namespace="retention")
    try:
        await provision_metrics_database(engine)
        await metrics.record_observations((_observation("old"),))
        async with engine.connect() as connection:
            before = (await connection.execute(select(table))).mappings().one()
        assert await metrics.prune(before=_BEFORE) == 1
        async with engine.connect() as connection:
            pruned = (await connection.execute(select(table))).mappings().one()
            assert await connection.scalar(text(
                "SELECT payload_json IS NULL FROM ai_metric_observations"
            ))
        assert pruned["pruned_at"] is not None
        assert pruned["active_occurred_at"] is None
        for field in ("namespace", "observation_id", "namespace_digest", "observation_digest",
                      "payload_digest", "kind", "occurred_at"):
            assert pruned[field] == before[field]
        assert await metrics.prune(before=_END) == 0
        async with engine.connect() as connection:
            assert (await connection.execute(select(table))).mappings().one() == pruned
        reopened = Metrics.sqlite(tmp_path / "receipt.db", namespace="retention")
        with pytest.raises(AIError) as caught:
            await reopened.record_observations((_observation("old"),))
        assert caught.value.safe_details == {"reason": _REASON}
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ("namespace", "identity", "digest", "time", "projection", "payload", "json"))
async def test_prune_cannot_turn_corruption_into_a_deletion_proof(tmp_path: Path, damage: str) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'corrupt.db'}")
    table = build_metrics_sql_metadata().tables["ai_metric_observations"]
    metrics = Metrics.sql(engine, namespace="retention")
    try:
        await provision_metrics_database(engine)
        await metrics.record_observations((_observation("old"),))
        values = {
            "namespace": {"namespace": "other"}, "identity": {"observation_id": "other"},
            "digest": {"payload_digest": "0" * 64}, "time": {"occurred_at": _START + timedelta(microseconds=1)},
            "projection": {"active_occurred_at": _START - timedelta(microseconds=1)},
            "payload": {"payload_json": None},
        }
        async with engine.begin() as connection:
            if damage == "json":
                raw = await connection.scalar(text("SELECT payload_json FROM ai_metric_observations"))
                await connection.execute(text("UPDATE ai_metric_observations SET payload_json = :value"),
                                         {"value": raw[:-1] + ', "version":1}'})
            else:
                await connection.execute(update(table).values(**values[damage]))
        with pytest.raises(AIError) as caught:
            await metrics.prune(before=_BEFORE)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        async with engine.connect() as connection:
            assert await connection.scalar(select(table.c.pruned_at)) is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_paged_prune_is_one_atomic_mutation(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'paged.db'}")
    table = build_metrics_sql_metadata().tables["ai_metric_observations"]
    metrics = Metrics.sql(engine, namespace="retention")
    observations = tuple(_observation(str(index), occurred_at=_START + timedelta(microseconds=index))
                         for index in range(260))
    try:
        await provision_metrics_database(engine)
        await metrics.record_observations(observations[:256])
        await metrics.record_observations(observations[256:])
        async with engine.begin() as connection:
            await connection.execute(update(table).where(table.c.observation_id == "259").values(payload_digest="0" * 64))
        with pytest.raises(AIError):
            await metrics.prune(before=_BEFORE)
        assert await metrics.get_observation("0") == observations[0]
        async with engine.connect() as connection:
            assert (await connection.execute(select(table.c.pruned_at).distinct())).scalars().all() == [None]
        from linktools.ai.observe._codec import observation_payload_digest

        async with engine.begin() as connection:
            await connection.execute(update(table).where(table.c.observation_id == "259").values(
                payload_digest=observation_payload_digest("retention", observations[-1]),
            ))
        assert await metrics.prune(before=_BEFORE) == 260
        assert await metrics.prune(before=_BEFORE) == 0
        assert await metrics.get_observation("259") is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ("namespace", "identity", "kind", "time", "projection", "payload"))
async def test_invalid_tombstone_is_not_exact_pruned_proof(tmp_path: Path, damage: str) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'invalid-receipt.db'}")
    table = build_metrics_sql_metadata().tables["ai_metric_observations"]
    metrics = Metrics.sql(engine, namespace="retention")
    observation = _observation("old")
    try:
        await provision_metrics_database(engine)
        await metrics.record_observations((observation,))
        assert await metrics.prune(before=_BEFORE) == 1
        values = {
            "namespace": {"namespace": "other"}, "identity": {"observation_id": "other"},
            "kind": {"kind": "app.other"}, "time": {"occurred_at": _BEFORE},
            "projection": {"active_occurred_at": _START}, "payload": {"payload_json": {}},
        }
        async with engine.begin() as connection:
            await connection.execute(update(table).values(**values[damage]))
        with pytest.raises(AIError) as caught:
            await metrics.record_observations((observation,))
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert caught.value.safe_details.get("reason") != _REASON
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_unknown_commit_then_prune_requires_original_identity_retry() -> None:
    class UnknownCommitStore(InMemoryMetricStore):
        unknown = True

        async def put_observations(self, namespace: str, observations: tuple[Observation, ...]) -> None:
            await super().put_observations(namespace, observations)
            if self.unknown:
                self.unknown = False
                await self.prune_observations(namespace, before=_BEFORE)
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

    metrics = Metrics.from_store(UnknownCommitStore(), namespace="retention")
    with pytest.raises(AIError) as caught:
        await metrics.record_observations((_observation("old"),))
    assert caught.value.code is ErrorCode.STORAGE_CONFLICT
    assert caught.value.safe_details == {"reason": _REASON}
    assert await metrics.get_observation("old") is None


@pytest.mark.asyncio
async def test_unconfirmed_rollback_does_not_expose_pruned_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy.ext.asyncio import AsyncSessionTransaction

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'unknown.db'}")
    store = SqlMetricStore(engine)
    observation = _observation("old")
    original = AsyncSessionTransaction.__aexit__

    async def rollback_unknown(self: AsyncSessionTransaction, *args: object) -> None:
        await original(self, *args)
        if args[0] is not None:
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

    try:
        await provision_metrics_database(engine)
        await store.put_observations("retention", (observation,))
        await store.prune_observations("retention", before=_BEFORE)
        monkeypatch.setattr(AsyncSessionTransaction, "__aexit__", rollback_unknown)
        with pytest.raises(AIError) as caught:
            await store.put_observations("retention", (observation,))
        assert caught.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert caught.value.safe_details.get("reason") != _REASON
    finally:
        await engine.dispose()
