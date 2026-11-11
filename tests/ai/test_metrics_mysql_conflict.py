#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Schema-changing conflict probes own an isolated module database."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_metrics_database
from linktools.ai.observe import Metrics, Observation
from linktools.ai.storage import SQLiteDialect


@pytest.mark.asyncio
@pytest.mark.parametrize("_server", ("mysql",), indirect=True)
@pytest.mark.parametrize("cross_namespace", (False, True))
async def test_mysql_secondary_unique_conflict_cannot_commit_observation_replacement(
    server_metrics: tuple[Metrics, AsyncEngine, AsyncEngine], cross_namespace: bool,
) -> None:
    from sqlalchemy import text

    metrics, engine, _ = server_metrics
    async with engine.begin() as connection:
        await connection.execute(text(
            "ALTER TABLE ai_metric_observations ADD UNIQUE KEY uk_kind (kind)"
        ))
    observation = Observation(
        version=1, observation_id="existing", kind="app.event",
        occurred_at=datetime(2026, 10, 11, tzinfo=timezone.utc),
        source_namespace="source", tenant_id="tenant", status="SUCCEEDED",
        error_code=None, correlation={}, dimensions={}, measurements=(),
    )
    await metrics.record_observations((observation,))
    target = Metrics.sql(engine, namespace="other") if cross_namespace else metrics
    conflicting = observation if cross_namespace else replace(observation, observation_id="different")
    new = replace(observation, observation_id="new", kind="app.other")
    with pytest.raises(AIError) as caught:
        await target.record_observations((new, conflicting))
    assert caught.value.code is ErrorCode.STORAGE_CONFLICT
    assert "reason" not in caught.value.safe_details
    assert await metrics.get_observation("existing") == observation
    assert await target.get_observation(conflicting.observation_id) is None
    assert await target.get_observation("new") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("cross_namespace", (False, True))
async def test_secondary_conflict_readback_preserves_original_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cross_namespace: bool,
) -> None:
    from sqlalchemy import text
    from sqlalchemy.dialects.sqlite import insert

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'secondary-conflict.db'}")
    metrics = Metrics.sql(engine, namespace="secondary-conflict")
    observation = Observation(
        version=1, observation_id="existing", kind="app.event",
        occurred_at=datetime(2026, 10, 11, tzinfo=timezone.utc),
        source_namespace="source", tenant_id="tenant", status="SUCCEEDED",
        error_code=None, correlation={}, dimensions={}, measurements=(),
    )

    async def insert_with_secondary_conflict(self, session, *, table, rows, index_elements):
        # Model MySQL's temporary target-key rewrite after a secondary duplicate.
        statement = insert(table).values([dict(row) for row in rows])
        column = index_elements[0]
        await session.execute(statement.on_conflict_do_update(
            index_elements=("kind",), set_={column: statement.excluded[column]},
        ))
        return None

    try:
        await provision_metrics_database(engine)
        await metrics.record_observations((observation,))
        async with engine.begin() as connection:
            await connection.execute(text("CREATE UNIQUE INDEX uk_kind ON ai_metric_observations (kind)"))
        monkeypatch.setattr(SQLiteDialect, "insert_ignore_conflict_many", insert_with_secondary_conflict)
        target = Metrics.sql(engine, namespace="other") if cross_namespace else metrics
        conflicting = observation if cross_namespace else replace(observation, observation_id="different")
        new = replace(observation, observation_id="new", kind="app.other")
        with pytest.raises(AIError) as caught:
            await target.record_observations((new, conflicting))
        assert caught.value.code is ErrorCode.STORAGE_CONFLICT
        assert "reason" not in caught.value.safe_details
        assert await metrics.get_observation("existing") == observation
        assert await target.get_observation(conflicting.observation_id) is None
        assert await target.get_observation("new") is None
    finally:
        await engine.dispose()
