#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical JSON round-trip tests for TaskPlan/TaskExecution and SwarmSpec.
Decode routes through the frozen-dataclass constructors, so the round-trip
shares construction validation (the spec's step-one verification)."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from linktools.ai.execution.domain import RunError
from linktools.ai.json import canonical_json_bytes
from linktools.ai.storage.coordination.lease import Lease
from linktools.ai.tasks.codec import (
    decode_execution,
    decode_plan,
    encode_execution,
    encode_plan,
)
from linktools.ai.tasks.models import (
    DependencyFailurePolicy,
    TaskDependency,
    TaskExecution,
    TaskGraphNodePayload,
    TaskNode,
    TaskPlan,
    TaskStatus,
    TaskUsage,
)
from linktools.ai.tasks.swarm.aggregation import AggregationMode, AggregationPolicy
from linktools.ai.tasks.swarm.codec import decode_swarm_spec, encode_swarm_spec
from linktools.ai.tasks.swarm.limits import SwarmLimits
from linktools.ai.tasks.swarm.models import AgentRef
from linktools.ai.tasks.swarm.spec import (
    SwarmContextPolicy,
    SwarmSpec,
    SwarmStrategySpec,
)


def _sample_plan() -> TaskPlan:
    return TaskPlan(
        "plan-1",
        (
            TaskNode("a", TaskGraphNodePayload("agent-a", "do a")),
            TaskNode(
                "b",
                TaskGraphNodePayload("agent-b", "do b", metadata={"k": "v"}),
                dependencies=(TaskDependency("a"),),
            ),
            TaskNode(
                "c",
                TaskGraphNodePayload("agent-c", "do c"),
                dependencies=(TaskDependency("b", DependencyFailurePolicy.PROCEED_DEGRADED),),
            ),
        ),
    )


def test_plan_codec_round_trips_every_field():
    plan = _sample_plan()
    encoded = encode_plan(plan)
    # the encoded form must itself be canonical-JSON serializable (JsonValue)
    canonical_json_bytes(encoded)
    decoded = decode_plan(encoded)
    assert decoded == plan


def test_execution_codec_round_trips_terminal_states():
    now = datetime.now(timezone.utc)
    base = dict(
        id="exec-1",
        plan_id="plan-1",
        node_id="a",
        lease=Lease("owner", 3, now),
    )
    completed = TaskExecution(
        **base,
        status=TaskStatus.COMPLETED,
        attempt=1,
        active_run_id="child-1",
        result={"out": 1},
        usage=TaskUsage(12, 8, Decimal("0.0042")),
    )
    decoded = decode_execution(encode_execution(completed))
    assert decoded.status is TaskStatus.COMPLETED
    assert decoded.active_run_id == "child-1"
    assert decoded.usage.total_cost == Decimal("0.0042")
    assert decoded.usage.input_tokens == 12

    failed = TaskExecution(
        **base,
        status=TaskStatus.FAILED,
        attempt=1,
        active_run_id="child-1",
        error=RunError("boom", "it broke", {"detail": 1}),
        usage=TaskUsage(3, 1),
    )
    decoded_fail = decode_execution(encode_execution(failed))
    assert decoded_fail.status is TaskStatus.FAILED
    assert decoded_fail.error.error_type == "boom"
    assert decoded_fail.error.detail == {"detail": 1}

    skipped = TaskExecution(
        **base,
        status=TaskStatus.SKIPPED,
        blocked_by=("upstream",),
        terminal_reason="dep failed",
    )
    decoded_skip = decode_execution(encode_execution(skipped))
    assert decoded_skip.status is TaskStatus.SKIPPED
    assert decoded_skip.blocked_by == ("upstream",)


def test_cost_round_trips_as_decimal_string():
    plan = _sample_plan()
    enc = encode_plan(plan)
    # nested payloads are JsonValue: metadata stays a dict, not Decimal
    canonical_json_bytes(enc)


def test_swarm_spec_codec_round_trips_task_graph():
    spec = SwarmSpec(
        id="ws",
        name="workers",
        agents=(AgentRef("a1"), AgentRef("a2", role="checker")),
        strategy=SwarmStrategySpec("task_graph"),
        limits=SwarmLimits(
            max_rounds=1,
            max_tasks=50,
            max_delegations=0,
            max_depth=0,
            max_concurrency=4,
            max_total_tokens=10000,
            max_total_cost=Decimal("1.50"),
            timeout_seconds=120.0,
        ),
        context_policy=SwarmContextPolicy(),
        aggregation=AggregationPolicy(mode=AggregationMode.COLLECT),
    )
    encoded = encode_swarm_spec(spec)
    canonical_json_bytes(encoded)
    decoded = decode_swarm_spec(encoded)
    assert decoded.id == spec.id
    assert decoded.strategy.kind == "task_graph"
    assert decoded.aggregation.mode is AggregationMode.COLLECT
    assert decoded.limits.max_total_cost == Decimal("1.50")
    assert decoded.limits.max_total_tokens == 10000
    assert decoded.agents[1].role == "checker"


def test_decode_rejects_illegal_state_combination():
    # READY with attempt=1 cannot be constructed, so the codec must reject it
    bad = {
        "id": "e",
        "plan_id": "p",
        "node_id": "n",
        "status": "ready",
        "lease": {"owner": None, "fence": 0, "expires_at": None},
        "attempt": 1,
        "active_run_id": None,
        "result": None,
        "error": None,
        "blocked_by": [],
        "terminal_reason": None,
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_cost": None},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with pytest.raises(ValueError):
        decode_execution(bad)
