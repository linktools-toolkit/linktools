#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Usage capture and task-level accounting tests."""

from decimal import Decimal

import pytest

from linktools.ai.errors import UsageObservationConflictError
from linktools.ai.execution.domain import RunError
from linktools.ai.execution.snapshots import (
    ModelRequestUsageObservation,
    RequestUsage,
    RunUsageCapture,
)
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
    capture.observe_request(
        ModelRequestUsageObservation(
            "request-1",
            RequestUsage(input_tokens=10, output_tokens=4, total_tokens=14),
            None,
            None,
            provider_request_cost=Decimal("0.3"),
        ),
        pricing_id=None,
        pricing=None,
    )
    capture.observe_request(
        ModelRequestUsageObservation(
            "request-2",
            RequestUsage(input_tokens=2, output_tokens=1, total_tokens=3),
            None,
            None,
        ),
        pricing_id=None,
        pricing=None,
    )
    assert capture.snapshot().total_cost is None

    capture.observe_request(
        ModelRequestUsageObservation(
            "request-3",
            RequestUsage(),
            None,
            None,
            provider_request_cost=Decimal("0.4"),
        ),
        pricing_id=None,
        pricing=None,
    )
    assert capture.snapshot().total_cost is None


def test_usage_capture_rejects_duplicate_request_with_different_observation():
    capture = RunUsageCapture()
    capture.observe_request(
        ModelRequestUsageObservation(
            "regression-base",
            RequestUsage(
                input_tokens=10,
                output_tokens=8,
                cache_write_tokens=2,
                cache_read_tokens=3,
            ),
            None,
            None,
        ),
        pricing_id=None,
        pricing=None,
    )
    with pytest.raises(UsageObservationConflictError):
        capture.observe_request(
            ModelRequestUsageObservation(
                "regression-base",
                RequestUsage(
                    input_tokens=9,
                    output_tokens=8,
                ),
                None,
                None,
            ),
            pricing_id=None,
            pricing=None,
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
        parent_owner="scheduler",
        parent_fence=0,
    )

    usage = await engine.execute(plan)

    assert usage.input_tokens == 11
    assert usage.output_tokens == 6
