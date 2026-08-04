#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Usage capture and task-level accounting tests."""

from decimal import Decimal

import pytest

from linktools.ai.errors import UsageRegressionError
from linktools.ai.execution.domain import RunError, RunUsage
from linktools.ai.execution.snapshots import RunUsageCapture
from linktools.ai.tasks.persistence.local import LocalTaskBackend
from linktools.ai.tasks.models import TaskUsage
from linktools.ai.tasks.swarm.engine import TaskGraphEngine

from tests.ai.tasks.swarm._support import (
    NoopGate,
    RecordingRunner,
    make_plan,
    ready_executions,
)
from tests.ai.tasks.swarm.test_engine import _limits


def test_usage_capture_preserves_unknown_cost_and_accepts_authoritative_recovery():
    capture = RunUsageCapture()
    capture.observe_absolute(
        RunUsage(input_tokens=10, output_tokens=4, total_tokens=14, total_cost=Decimal("0.3"))
    )
    capture.observe_absolute(
        RunUsage(input_tokens=12, output_tokens=5, total_tokens=17, total_cost=None)
    )
    assert capture.snapshot().total_cost is None

    capture.observe_absolute(
        RunUsage(input_tokens=12, output_tokens=5, total_tokens=17, total_cost=Decimal("0.4"))
    )
    assert capture.snapshot().total_cost == Decimal("0.4")


@pytest.mark.parametrize(
    "field",
    ("input_tokens", "output_tokens", "total_tokens", "cache_write_tokens", "cache_read_tokens"),
)
def test_usage_capture_rejects_any_cumulative_regression(field):
    capture = RunUsageCapture()
    capture.observe_absolute(
        RunUsage(
            input_tokens=10,
            output_tokens=8,
            total_tokens=18,
            cache_write_tokens=2,
            cache_read_tokens=3,
        )
    )
    values = {
        "input_tokens": 9,
        "output_tokens": 7,
        "total_tokens": 17,
        "cache_write_tokens": 1,
        "cache_read_tokens": 2,
    }
    current = capture.snapshot()
    with pytest.raises(UsageRegressionError):
        capture.observe_absolute(
            RunUsage(
                input_tokens=values["input_tokens"] if field == "input_tokens" else current.input_tokens,
                output_tokens=values["output_tokens"] if field == "output_tokens" else current.output_tokens,
                total_tokens=values["total_tokens"] if field == "total_tokens" else current.total_tokens,
                cache_write_tokens=values["cache_write_tokens"] if field == "cache_write_tokens" else current.cache_write_tokens,
                cache_read_tokens=values["cache_read_tokens"] if field == "cache_read_tokens" else current.cache_read_tokens,
            )
        )


@pytest.mark.asyncio
async def test_task_graph_usage_includes_failed_node_usage():
    store = LocalTaskBackend()
    plan = make_plan(("a",))
    await store.create_plan(plan, ready_executions(plan))
    runner = RecordingRunner(
        usage={"a": TaskUsage(input_tokens=11, output_tokens=6)},
        failures={"a": RunError("failed", "failed")},
    )
    engine = TaskGraphEngine(
        store=store,
        runner=runner,
        gate=NoopGate(),
        limits=_limits(max_concurrency=1),
        owner="scheduler",
        parent_run_id="parent",
    )

    usage = await engine.execute(plan)

    assert usage.input_tokens == 11
    assert usage.output_tokens == 6
