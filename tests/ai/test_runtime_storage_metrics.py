#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Runtime storage-operation Metrics integration regressions."""
from ._runtime_test_helpers import model_features

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus, JsonValue
from linktools.ai.observe import MetricQuery, MetricWindow, Metrics
from linktools.ai.observe._memory import InMemoryMetricStore

from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from pydantic_ai.models.test import TestModel


class _ModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:test"
    model_digest = "d" * 64
    contract: dict[str, JsonValue] = {"provider": "test", "model": "test"}

    def materialize(self) -> TestModel:
        return TestModel(custom_output_text="ok")


class _Models(_ModelBinding):
    pass


def _agent_group() -> RuntimeFeatures[object]:
    group = RuntimeFeatures[object]("application")
    group.agent("default", model="default", allow_tools=())
    return group


@pytest.mark.asyncio
async def test_runtime_projects_storage_operation_metrics(tmp_path: Path) -> None:
    store = InMemoryMetricStore()
    metrics = Metrics.from_store(store, namespace="runtime-storage-metrics")
    start = datetime.now(timezone.utc) - timedelta(seconds=1)

    async with Persistence.open('default', storage=PersistenceStorage.in_memory()) as persistence_47, Runtime.attach(persistence=persistence_47, metrics=metrics, features=model_features(_Models(), _agent_group())) as runtime:
        result = (await runtime.agents.get("default").run("hello", timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED

    end = datetime.now(timezone.utc) + timedelta(seconds=1)
    window = MetricWindow.between(start, end)
    count = await metrics.query(
        MetricQuery(
            "linktools.storage.operation.count",
            window,
            filters={"domain": "execution", "target": "runtime"},
        )
    )
    latency = await metrics.query(
        MetricQuery(
            "linktools.storage.operation.latency",
            window,
            filters={"domain": "execution", "target": "runtime"},
        )
    )
    failure = await metrics.query(
        MetricQuery(
            "linktools.storage.operation.failure_ratio",
            window,
            filters={"domain": "execution", "target": "runtime"},
        )
    )

    assert count.points[0].value == 1
    assert count.points[0].sample_count == 1
    assert isinstance(latency.points[0].value, (int, float))
    assert latency.points[0].value > 0
    assert latency.points[0].sample_count == 1
    assert failure.points[0].value == 0
    assert failure.points[0].sample_count == 1

    page = await store.scan_observations(
        "runtime-storage-metrics",
        "linktools.storage.operation",
        start,
        end,
        cursor=None,
        limit=100,
    )
    assert page.next_cursor is None
    assert len(page.items) == 1
    observation = page.items[0]
    assert observation.status == "SUCCEEDED"
    assert observation.error_code is None
    assert dict(observation.dimensions) == {
        "domain": "execution",
        "target": "runtime",
    }
    assert [(item.name, item.revision) for item in observation.measurements] == [
        ("latency_ns", 1)
    ]
