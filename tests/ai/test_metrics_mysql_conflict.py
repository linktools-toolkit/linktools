#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Schema-changing conflict probes own an isolated module database."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from linktools.ai.errors import AIError
from linktools.ai.observe import Metrics, Observation


@pytest.mark.asyncio
@pytest.mark.parametrize("_server", ("mysql",), indirect=True)
async def test_mysql_secondary_unique_conflict_cannot_commit_observation_replacement(
    server_metrics: tuple[Metrics, AsyncEngine, AsyncEngine],
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
    with pytest.raises(AIError, match="STORAGE_CONFLICT"):
        await metrics.record_observations((replace(observation, observation_id="different"),))
    assert await metrics.get_observation("existing") == observation
    assert await metrics.get_observation("different") is None
