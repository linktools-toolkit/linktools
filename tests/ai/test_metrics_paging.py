#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metric query paging and idempotent commit-unknown replay regressions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from linktools.ai.errors import AIError, ErrorCode
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
from linktools.ai.observe._memory import InMemoryMetricStore

pytestmark = pytest.mark.asyncio


class _CommitUnknownOnceStore(InMemoryMetricStore):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[Observation, ...]] = []
        self._failed = False

    async def put_observations(
        self,
        namespace: str,
        observations: tuple[Observation, ...],
    ) -> None:
        self.calls.append(observations)
        if not self._failed:
            self._failed = True
            await super().put_observations(namespace, observations)
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        await super().put_observations(namespace, observations)


def _count_definition() -> MetricDefinition:
    return MetricDefinition(
        name="business.paged.count",
        revision=1,
        observation_kind="business.paged.event",
        source=MetricSource.observation_count(),
        metric_type=MetricType.COUNTER,
        unit="1",
        default_aggregation=MetricAggregation.SUM,
    )


async def test_query_scans_multiple_store_pages_without_truncation() -> None:
    metrics = Metrics.in_memory(namespace="paged-query")
    await metrics.define(_count_definition())
    start = datetime(2026, 9, 5, tzinfo=timezone.utc)

    observations = tuple(
        Observation(
            version=1,
            observation_id=f"event-{index}",
            kind="business.paged.event",
            occurred_at=start + timedelta(microseconds=index),
            source_namespace="workspace",
            tenant_id="tenant",
            status="SUCCEEDED",
            error_code=None,
            correlation={},
            dimensions={},
            measurements=(),
        )
        for index in range(600)
    )
    for offset in range(0, len(observations), 200):
        await metrics.record_observations(observations[offset : offset + 200])

    result = await metrics.query(
        MetricQuery(
            "business.paged.count",
            MetricWindow.between(start, start + timedelta(seconds=1)),
        )
    )

    assert len(result.points) == 1
    assert result.points[0].value == 600
    assert result.points[0].sample_count == 600


async def test_commit_unknown_replays_the_exact_same_observation_batch_once() -> None:
    store = _CommitUnknownOnceStore()
    metrics = Metrics.from_store(store, namespace="commit-unknown")
    occurred_at = datetime(2026, 9, 5, tzinfo=timezone.utc)
    observations = (
        Observation(
            version=1,
            observation_id="stable-a",
            kind="business.commit.event",
            occurred_at=occurred_at,
            source_namespace="workspace",
            tenant_id="tenant",
            status="SUCCEEDED",
            error_code=None,
            correlation={},
            dimensions={},
            measurements=(),
        ),
        Observation(
            version=1,
            observation_id="stable-b",
            kind="business.commit.event",
            occurred_at=occurred_at,
            source_namespace="workspace",
            tenant_id="tenant",
            status="SUCCEEDED",
            error_code=None,
            correlation={},
            dimensions={},
            measurements=(),
        ),
    )

    await metrics.record_observations(observations)

    assert len(store.calls) == 2
    assert store.calls[0] is store.calls[1]
    assert tuple(item.observation_id for item in store.calls[0]) == (
        "stable-a",
        "stable-b",
    )
