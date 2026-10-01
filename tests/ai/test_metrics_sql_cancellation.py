#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cancelled buffered metric reads release SQLite locks before returning."""

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from aiosqlite import Cursor
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.migrate import provision_metrics_database
from linktools.ai.observe import (
    MetricAggregation,
    MetricDefinition,
    MetricQuery,
    MetricSource,
    MetricType,
    MetricWindow,
    Metrics,
    Observation,
)
from linktools.ai.observe._sql import SqlMetricStore


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("point", "query"))
async def test_cancelled_metric_read_releases_cursor_before_followup_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'metrics.db'}")
    await provision_metrics_database(engine)
    store = SqlMetricStore(engine)
    metrics = Metrics.from_store(store, namespace="cancelled-read")
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)
    await metrics.define(MetricDefinition(
        name="business.cancel.count", revision=1,
        observation_kind="business.cancel", source=MetricSource.observation_count(),
        metric_type=MetricType.COUNTER, unit="1",
        default_aggregation=MetricAggregation.COUNT,
    ))
    first = Observation(
        version=1, observation_id="one", kind="business.cancel", occurred_at=start,
        source_namespace=None, tenant_id=None, status="SUCCEEDED", error_code=None,
        correlation={}, dimensions={}, measurements=(),
    )
    await metrics.record_observations((first,))
    query = MetricQuery(
        "business.cancel.count",
        MetricWindow.between(start, start + timedelta(seconds=1)),
    )
    fetching = asyncio.Event()
    release = asyncio.Event()
    original_fetchall = Cursor.fetchall

    async def pause_fetchall(cursor: Cursor):
        columns = {column[0] for column in cursor.description or ()}
        selected = (
            "observation_digest" in columns
            if operation == "point"
            else "sample_count" in columns
        )
        if not fetching.is_set() and selected:
            fetching.set()
            await release.wait()
        return await original_fetchall(cursor)

    monkeypatch.setattr(Cursor, "fetchall", pause_fetchall)
    read = (
        store.get_observation("cancelled-read", "one")
        if operation == "point"
        else metrics.query(query)
    )
    reader = asyncio.create_task(read)
    try:
        await asyncio.wait_for(fetching.wait(), 5)
        reader.cancel()
        await asyncio.sleep(0)
        release.set()
        try:
            await reader
        except asyncio.CancelledError:
            await metrics.record_observations((replace(first, observation_id="two"),))
            result = await metrics.query(query)
            assert result.points[0].value == 2
        else:
            pytest.fail("read cancellation was not propagated")
    finally:
        release.set()
        await asyncio.gather(reader, return_exceptions=True)
        await engine.dispose()
