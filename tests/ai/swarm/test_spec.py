#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for swarm.spec/limits/aggregation: the Swarm declaration layer
(SwarmSpec, SwarmLimits, AggregationPolicy/aggregate, SwarmContextPolicy)."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from linktools.ai.agent.spec import MiddlewareRef
from linktools.ai.run.models import RunResult
from linktools.ai.swarm.aggregation import (
    AggregationMode,
    AggregationPolicy,
    aggregate,
)
from linktools.ai.swarm.limits import DEFAULT_SWARM_LIMITS, SwarmLimits
from linktools.ai.swarm.models import (
    AgentRef,
    SwarmTask,
    SwarmTaskStatus,
    TaskInput,
)
from linktools.ai.swarm.spec import (
    SwarmContextPolicy,
    SwarmSpec,
    SwarmStrategySpec,
)


# --- helpers -----------------------------------------------------------------

def _make_task(output) -> SwarmTask:
    """Build a minimal SwarmTask carrying a succeeded RunResult(output=...)."""
    return SwarmTask(
        id="t-1",
        swarm_run_id="sr-1",
        parent_task_id=None,
        assigned_agent_id="agent-1",
        description="d",
        status=SwarmTaskStatus.SUCCEEDED,
        dependencies=(),
        input=TaskInput(prompt="p"),
        result=RunResult(output=output),
        error=None,
        attempts=1,
        version=1,
        claimed_at=None,
        lease_expires_at=None,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _make_failed_task() -> SwarmTask:
    """Build a SwarmTask whose result is None (e.g. failed/cancelled)."""
    return SwarmTask(
        id="t-fail",
        swarm_run_id="sr-1",
        parent_task_id=None,
        assigned_agent_id="agent-1",
        description="d",
        status=SwarmTaskStatus.FAILED,
        dependencies=(),
        input=TaskInput(prompt="p"),
        result=None,
        error=None,
        attempts=1,
        version=1,
        claimed_at=None,
        lease_expires_at=None,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


# --- SwarmLimits -------------------------------------------------------------

def test_swarm_limits_constructs_with_all_fields():
    limits = SwarmLimits(
        max_rounds=5,
        max_tasks=20,
        max_delegations=10,
        max_depth=3,
        max_concurrency=2,
        max_total_tokens=10000,
        max_total_cost=Decimal("1.50"),
        timeout_seconds=30.0,
    )
    assert limits.max_rounds == 5
    assert limits.max_tasks == 20
    assert limits.max_delegations == 10
    assert limits.max_depth == 3
    assert limits.max_concurrency == 2
    assert limits.max_total_tokens == 10000
    assert limits.max_total_cost == Decimal("1.50")
    assert limits.timeout_seconds == 30.0


def test_default_swarm_limits_values():
    assert DEFAULT_SWARM_LIMITS.max_rounds == 10
    assert DEFAULT_SWARM_LIMITS.max_tasks == 50
    assert DEFAULT_SWARM_LIMITS.max_delegations == 20
    assert DEFAULT_SWARM_LIMITS.max_depth == 5
    assert DEFAULT_SWARM_LIMITS.max_concurrency == 4
    assert DEFAULT_SWARM_LIMITS.max_total_tokens is None
    assert DEFAULT_SWARM_LIMITS.max_total_cost is None
    assert DEFAULT_SWARM_LIMITS.timeout_seconds is None


def test_swarm_limits_is_frozen():
    limits = SwarmLimits(
        max_rounds=1, max_tasks=1, max_delegations=1, max_depth=1,
        max_concurrency=1, max_total_tokens=None,
        max_total_cost=None, timeout_seconds=None,
    )
    with pytest.raises(FrozenInstanceError):
        limits.max_rounds = 99  # type: ignore[misc]


# --- AggregationPolicy -------------------------------------------------------

def test_aggregation_policy_defaults_to_concat():
    policy = AggregationPolicy()
    assert policy.mode is AggregationMode.CONCAT


def test_aggregation_mode_is_str_enum():
    assert isinstance(AggregationMode.CONCAT, str)
    assert AggregationMode.CONCAT == "concat"
    assert AggregationMode.FIRST == "first"
    assert AggregationMode.LAST == "last"
    assert AggregationMode.MERGE == "merge"


# --- aggregate() -------------------------------------------------------------

def test_aggregate_concat_joins_outputs():
    tasks = (_make_task("alpha"), _make_task("beta"))
    result = aggregate(AggregationPolicy(AggregationMode.CONCAT), tasks)
    assert result.output == "alpha\nbeta"
    assert result.metadata["task_count"] == 2
    assert result.token_usage == {}


def test_aggregate_first_returns_first_succeeded():
    tasks = (_make_task("alpha"), _make_task("beta"))
    result = aggregate(AggregationPolicy(AggregationMode.FIRST), tasks)
    assert result.output == "alpha"
    assert result.metadata["task_count"] == 2


def test_aggregate_last_returns_last_succeeded():
    tasks = (_make_task("alpha"), _make_task("beta"))
    result = aggregate(AggregationPolicy(AggregationMode.LAST), tasks)
    assert result.output == "beta"
    assert result.metadata["task_count"] == 2


def test_aggregate_merge_merges_dict_outputs():
    tasks = (_make_task({"a": 1}), _make_task({"b": 2}))
    result = aggregate(AggregationPolicy(AggregationMode.MERGE), tasks)
    assert result.output == {"a": 1, "b": 2}
    assert result.metadata["task_count"] == 2


def test_aggregate_merge_skips_non_dict_outputs():
    tasks = (_make_task({"a": 1}), _make_task("not-a-dict"))
    result = aggregate(AggregationPolicy(AggregationMode.MERGE), tasks)
    assert result.output == {"a": 1}


def test_aggregate_skips_tasks_with_none_result():
    tasks = (_make_task("alpha"), _make_failed_task(), _make_task("beta"))
    result = aggregate(AggregationPolicy(AggregationMode.CONCAT), tasks)
    assert result.output == "alpha\nbeta"
    assert result.metadata["task_count"] == 2


def test_aggregate_concat_empty_tasks_returns_empty_string():
    result = aggregate(AggregationPolicy(AggregationMode.CONCAT), ())
    assert result.output == ""
    assert result.metadata["task_count"] == 0


def test_aggregate_first_empty_returns_empty_string():
    result = aggregate(AggregationPolicy(AggregationMode.FIRST), ())
    assert result.output == ""
    assert result.metadata["task_count"] == 0


def test_aggregate_last_empty_returns_empty_string():
    result = aggregate(AggregationPolicy(AggregationMode.LAST), ())
    assert result.output == ""
    assert result.metadata["task_count"] == 0


def test_aggregate_merge_empty_returns_empty_dict():
    result = aggregate(AggregationPolicy(AggregationMode.MERGE), ())
    assert result.output == {}
    assert result.metadata["task_count"] == 0


def test_aggregate_concat_uses_str_on_non_string_outputs():
    tasks = (_make_task(42), _make_task(3.14))
    result = aggregate(AggregationPolicy(AggregationMode.CONCAT), tasks)
    assert result.output == "42\n3.14"


# --- SwarmContextPolicy ------------------------------------------------------

def test_swarm_context_policy_defaults():
    policy = SwarmContextPolicy()
    assert policy.coordinator_reads_session is True
    assert policy.worker_reads_session is False
    assert policy.worker_reads_summary is True
    assert policy.write_aggregate_to_session is True


def test_swarm_context_policy_is_frozen():
    policy = SwarmContextPolicy()
    with pytest.raises(FrozenInstanceError):
        policy.coordinator_reads_session = False  # type: ignore[misc]


# --- SwarmStrategySpec -------------------------------------------------------

def test_swarm_strategy_spec_defaults_config_to_empty_dict():
    strategy = SwarmStrategySpec(kind="parallel_fan_out")
    assert strategy.kind == "parallel_fan_out"
    assert strategy.config == {}


def test_swarm_strategy_spec_accepts_config():
    strategy = SwarmStrategySpec(kind="round_robin", config={"fanout": 4})
    assert strategy.config == {"fanout": 4}


# --- SwarmSpec ---------------------------------------------------------------

def _make_spec() -> SwarmSpec:
    coordinator = AgentRef(agent_id="agent-coord", role="coordinator")
    return SwarmSpec(
        id="swarm-1",
        name="research-swarm",
        agents=(
            coordinator,
            AgentRef(agent_id="agent-1", role="worker"),
        ),
        coordinator=coordinator,
        strategy=SwarmStrategySpec(kind="parallel_fan_out"),
        limits=DEFAULT_SWARM_LIMITS,
        context_policy=SwarmContextPolicy(),
        aggregation=AggregationPolicy(),
    )


def test_swarm_spec_constructs_with_required_fields():
    spec = _make_spec()
    assert spec.id == "swarm-1"
    assert spec.name == "research-swarm"
    assert len(spec.agents) == 2
    assert spec.coordinator.agent_id == "agent-coord"
    assert spec.strategy.kind == "parallel_fan_out"
    assert spec.limits is DEFAULT_SWARM_LIMITS
    assert isinstance(spec.context_policy, SwarmContextPolicy)
    assert isinstance(spec.aggregation, AggregationPolicy)


def test_swarm_spec_defaults_middleware_and_metadata():
    spec = _make_spec()
    assert spec.middleware == ()
    assert spec.metadata == {}


def test_swarm_spec_accepts_middleware_and_metadata():
    spec = SwarmSpec(
        id="swarm-2",
        name="swarm-with-mw",
        agents=(AgentRef(agent_id="a-1"),),
        coordinator=AgentRef(agent_id="a-1"),
        strategy=SwarmStrategySpec(kind="parallel_fan_out"),
        limits=DEFAULT_SWARM_LIMITS,
        context_policy=SwarmContextPolicy(),
        aggregation=AggregationPolicy(),
        middleware=(MiddlewareRef(name="logging"),),
        metadata={"owner": "team-a"},
    )
    assert len(spec.middleware) == 1
    assert spec.middleware[0].name == "logging"
    assert spec.metadata == {"owner": "team-a"}


def test_swarm_spec_is_frozen():
    spec = _make_spec()
    with pytest.raises(FrozenInstanceError):
        spec.name = "changed"  # type: ignore[misc]
