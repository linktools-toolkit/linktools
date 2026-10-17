#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite feature probes follow physical connections, not metric data."""

import asyncio
import sqlite3
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import event
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from linktools.ai.migrate import provision_metrics_database
from linktools.ai.observe import MetricAggregation, MetricQuery, MetricWindow, Metrics
from linktools.ai.observe._sql_query import _sql_features_available
from linktools.ai.web._app import _SUMMARY_METRICS

from .test_metrics_sql_query_limits import _START, _plan


@pytest.mark.asyncio
async def test_success_is_reused_per_connection_and_requirement() -> None:
    session = AsyncMock(spec=AsyncSession)
    session.connection.return_value = SimpleNamespace(
        dialect=SimpleNamespace(server_version_info=(3, 46, 1)), info={},
    )
    plain = replace(_plan(), aggregation=MetricAggregation.MIN, percentile=None)
    assert await _sql_features_available(session, "sqlite", plain)
    assert await _sql_features_available(session, "sqlite", plain)
    assert session.execute.await_count == 1
    assert await _sql_features_available(session, "sqlite", _plan())
    assert await _sql_features_available(session, "sqlite", _plan())
    assert session.execute.await_count == 2
    session.connection.return_value = SimpleNamespace(
        dialect=SimpleNamespace(server_version_info=(3, 46, 1)), info={},
    )
    assert await _sql_features_available(session, "sqlite", _plan())
    assert session.execute.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(("message", "missing"), (
    ("no such function: json_extract", True),
    ("database is locked", False),
))
async def test_failure_does_not_poison_later_probe(message: str, missing: bool) -> None:
    session = AsyncMock(spec=AsyncSession)
    connection = SimpleNamespace(
        dialect=SimpleNamespace(server_version_info=(3, 46, 1)), info={},
    )
    session.connection.return_value = connection
    error = OperationalError("probe", {}, sqlite3.OperationalError(message))
    session.execute.side_effect = [error, None]
    if missing:
        assert not await _sql_features_available(session, "sqlite", _plan())
    else:
        with pytest.raises(OperationalError) as caught:
            await _sql_features_available(session, "sqlite", _plan())
        assert caught.value is error
    assert connection.info == {}
    assert await _sql_features_available(session, "sqlite", _plan())
    assert session.execute.await_count == 2


@pytest.mark.asyncio
async def test_summary_probe_budget_and_recreated_connection(tmp_path: Path) -> None:
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "metrics.db"}', pool_size=1)
    await provision_metrics_database(engine)
    metrics = Metrics.sql(engine)
    await metrics.get_observation("warmup")
    probes: list[str] = []

    def capture(conn, cursor, sql, params, context, many):
        if "FROM json_each('[1]')" in sql:
            probes.append(sql)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    window = MetricWindow.between(_START, _START + timedelta(seconds=2))
    try:
        for _ in range(2):
            for name in _SUMMARY_METRICS:
                await metrics.query(MetricQuery(name, window))
        assert len(probes) == 1
        async with engine.connect() as connection:
            await connection.invalidate()
        await metrics.query(MetricQuery(_SUMMARY_METRICS[0], window))
        assert len(probes) == 2
        # Pool replacement must not inherit capabilities from the old DBAPI connection.
        await engine.dispose()
        await metrics.query(MetricQuery(_SUMMARY_METRICS[0], window))
        assert len(probes) == 3
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await engine.dispose()


@pytest.mark.asyncio
async def test_cancelled_probe_is_not_memoized() -> None:
    session = AsyncMock(spec=AsyncSession)
    connection = SimpleNamespace(
        dialect=SimpleNamespace(server_version_info=(3, 46, 1)), info={},
    )
    session.connection.return_value = connection
    cancelled = asyncio.CancelledError()
    session.execute.side_effect = [cancelled, None]
    with pytest.raises(asyncio.CancelledError) as caught:
        await _sql_features_available(session, "sqlite", _plan())
    assert caught.value is cancelled
    assert connection.info == {}
    assert await _sql_features_available(session, "sqlite", _plan())
    assert session.execute.await_count == 2
